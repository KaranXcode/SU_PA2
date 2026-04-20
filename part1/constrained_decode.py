"""
Task 1.2 -- Constrained decoding with Whisper + N-gram Logit Bias.

Mathematical formulation
------------------------
At each decode step t with Whisper logits z_t in R^V, we produce

    z_t'[v]  =  z_t[v]  +  lambda_bias * 1{v in T}
                         +  lambda_lm  * log P_ng(word_v | history)

where
    T                is the set of BPE tokens that are prefixes / completions
                     of the Speech-Course technical term vocabulary,
    P_ng(. | .)      is the trigram LM (ngram_lm.py),
    lambda_bias      is a constant token-level boost (default 2.0),
    lambda_lm        is the shallow-fusion weight (default 0.3).

The bias is applied ONLY on word boundaries (token is whitespace-initial)
so mid-word sub-tokens are untouched -- this avoids degenerating the ASR
output into spurious technical terms.

Usage:
    python constrained_decode.py --wav data/denoised.wav \
        --lm checkpoints/lm.pkl --corpus syllabus.txt \
        --out data/transcript.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import torch
from transformers import (WhisperForConditionalGeneration, WhisperProcessor,
                          LogitsProcessor, LogitsProcessorList)

from ngram_lm import NgramLM, technical_terms, tokenize

SAMPLE_RATE = 16000


class NgramLogitBias(LogitsProcessor):
    """Adds  lambda_bias * 1{token starts a technical term}  +
             lambda_lm   * log P_ng(last_word | history)
       to the logits at every decode step."""

    def __init__(self, tokenizer, lm: NgramLM, term_vocab,
                 lambda_bias: float = 2.0, lambda_lm: float = 0.3):
        self.tok = tokenizer
        self.lm = lm
        self.term_vocab = set(term_vocab)
        self.lambda_bias = lambda_bias
        self.lambda_lm = lambda_lm

        self.term_token_ids = self._collect_term_token_ids()
        self._word_boundary_re = re.compile(r"^[\sĠ]")   # BPE uses Ġ for space
        # cache of token_id -> decoded piece
        self._piece_cache = {}

    def _collect_term_token_ids(self):
        """Every BPE piece that begins one of the technical terms."""
        ids = set()
        for term in self.term_vocab:
            # encode as a standalone word (leading space => BPE boundary)
            toks = self.tok.encode(" " + term, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
        return ids

    def _piece(self, tid: int) -> str:
        if tid not in self._piece_cache:
            self._piece_cache[tid] = self.tok.decode([tid])
        return self._piece_cache[tid]

    def _history_words(self, input_ids_row: torch.Tensor) -> List[str]:
        text = self.tok.decode(input_ids_row.tolist(), skip_special_tokens=True)
        return tokenize(text)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # (1) Flat token-level bias on technical-term-initial pieces.
        if self.term_token_ids:
            idx = torch.tensor(sorted(self.term_token_ids),
                               device=scores.device, dtype=torch.long)
            scores.index_add_(
                1, idx,
                torch.full((scores.size(0), idx.numel()),
                           self.lambda_bias, device=scores.device)
            )

        # (2) Shallow-fusion LM term: only for tokens that *start* a new word.
        #     We score log P_ng(candidate_word | history_last_{n-1}).
        if self.lambda_lm > 0:
            for b in range(scores.size(0)):
                history = self._history_words(input_ids[b])
                ctx = tuple(history[-(self.lm.n - 1):]) if history else ()
                # Only probe the top-K candidates to stay cheap.
                topk = torch.topk(scores[b], k=50)
                for val, tid in zip(topk.values.tolist(), topk.indices.tolist()):
                    piece = self._piece(tid)
                    if not piece or not self._word_boundary_re.match(piece):
                        continue
                    word = piece.strip().lower()
                    if not word:
                        continue
                    lp = self.lm.logprob_word(word, ctx)
                    scores[b, tid] = scores[b, tid] + self.lambda_lm * lp
        return scores


def transcribe_segment(
    model, processor, wav: np.ndarray, sr: int,
    language: str, logits_processor,
) -> str:
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_feats = inputs.input_features.to(model.device)
    forced = processor.get_decoder_prompt_ids(language=language, task="transcribe")

    # Cap max_new_tokens to the actual segment duration. Hardcoded 220 was
    # letting Whisper burn the whole budget on a 3 s clip as a repetition
    # loop ("के लिएटर के लिएटर ..."). 6 tokens/sec is generous for speech
    # (real delivery is ~2-4 tok/sec), +20 slack for short utterances.
    dur_s = len(wav) / SAMPLE_RATE
    max_new = max(20, min(220, int(dur_s * 6) + 20))

    with torch.no_grad():
        out_ids = model.generate(
            input_feats,
            forced_decoder_ids=forced,
            num_beams=5,
            logits_processor=logits_processor,
            max_new_tokens=max_new,
            # Anti-hallucination guards. no_repeat_ngram_size=3 forbids exact
            # trigram repetition; repetition_penalty softly discourages any
            # re-emitted token. Both standard Whisper practice on low-SNR /
            # out-of-distribution speech (accented Hindi here).
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
    return processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[decode] loading Whisper ({args.model}) on {device}")
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device)
    model.eval()

    with open(args.lm, "rb") as f:
        lm: NgramLM = pickle.load(f)
    terms = technical_terms(Path(args.corpus))

    biaser = NgramLogitBias(processor.tokenizer, lm, terms,
                            lambda_bias=args.lambda_bias,
                            lambda_lm=args.lambda_lm)
    # The N-gram LM and technical-term bias are built from the English course
    # syllabus, so they only make sense for EN segments. Applying them to HI
    # segments was pushing decoded tokens toward English technical terms and
    # tanking the Hindi transcript. HI spans now decode with an empty
    # LogitsProcessorList so Whisper runs unconstrained + language-forced.
    lp_en = LogitsProcessorList([biaser])
    lp_hi = LogitsProcessorList([])

    # Load LID segments (from Task 1.1).
    with open(args.segments, "r", encoding="utf-8") as f:
        segments = json.load(f)

    arr, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    wav = arr[:, 0]
    if sr != SAMPLE_RATE:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    # Whisper's feature extractor pads / truncates to a fixed 30 s window
    # per call. Segments longer than that get silently cropped to the first
    # 30 s, which on the uncropped tail produced junk like "jimmy." for a
    # 180 s EN span. Split long segments into overlapping sub-windows.
    MAX_SUB_S = 28.0      # a hair under 30 so we don't lose tail audio
    OVERLAP_S = 1.5       # re-listen the last 1.5 s of the prev window

    results = []
    for seg in segments:
        if seg["lang"] == "sil":
            continue
        s = int(seg["start"] * sr)
        e = int(seg["end"]   * sr)
        chunk = wav[s:e]
        if len(chunk) < sr * 0.3:                   # skip < 300 ms
            continue
        lang = "hi" if seg["lang"] == "hi" else "en"
        lp = lp_hi if lang == "hi" else lp_en

        seg_dur = len(chunk) / sr
        if seg_dur <= MAX_SUB_S:
            text = transcribe_segment(model, processor, chunk, sr, lang, lp)
        else:
            parts = []
            win = int(MAX_SUB_S * sr)
            hop = int((MAX_SUB_S - OVERLAP_S) * sr)
            off = 0
            while off < len(chunk):
                sub = chunk[off:off + win]
                if len(sub) < sr * 0.3:
                    break
                parts.append(transcribe_segment(model, processor, sub, sr,
                                                lang, lp))
                if off + win >= len(chunk):
                    break
                off += hop
            text = " ".join(p.strip() for p in parts if p.strip())

        print(f"[{seg['start']:7.2f}-{seg['end']:7.2f}] ({lang}) {text}")
        results.append({**seg, "text": text})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Wrote transcript -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--segments", required=True,
                    help="LID segments JSON produced by infer_lid.py")
    ap.add_argument("--lm", required=True)
    ap.add_argument("--corpus", default="syllabus.txt")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--lambda_bias", type=float, default=2.0)
    ap.add_argument("--lambda_lm", type=float, default=0.3)
    ap.add_argument("--out", default="results/transcript.json")
    args = ap.parse_args()
    run(args)
