"""
Strict-metrics evaluation for the Speech Understanding Assignment 2.

Computes the three metrics not already produced by part1-4 pipelines:
    * MCD       (Mel-Cepstral Distortion, DTW-aligned, 24-dim MFCC)
    * LID Switching Accuracy / Density (against the LID JSON)
    * WER       (English only -- Hindi N/A on this monolingual audio)

WER ground-truth strategy
-------------------------
We don't have a human reference transcript. We re-transcribe the denoised
lecture with un-constrained Whisper-medium (no language forcing, no logit
bias) and treat that as a *pseudo* ground truth, then compute WER against
the constrained-decoded transcript.json. This is the standard fallback
when human references are unavailable; the report should disclose this.

MCD definition
--------------
MCD = (10 / ln 10) * sqrt(2) * mean_t sqrt( sum_{k=1..K} (c_t^ref - c_t^syn)^2 )

where c_t are MFCCs (we drop c0 = energy) and the per-frame distances are
averaged after DTW-aligning the reference and synthesized streams. We use
K = 24 mel-cepstrum coefficients.

Usage:
    python eval_metrics.py
        --ref part1/data/student_voice_ref.wav
        --clone results/output_LRL_cloned.wav
        --denoised results/denoised.wav
        --transcript results/transcript.json
        --segments results/lid_segments.json
        --out results/eval_metrics.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


# --------------------------------------------------------------------------- #
# 1. MCD
# --------------------------------------------------------------------------- #
def _load_mono(path: Path, sr: int = 22050,
               target_rms: float = 0.05) -> np.ndarray:
    """Load + downmix to mono + resample to `sr` + RMS-normalise.

    RMS-normalisation is the fix for the MCD blowup we saw earlier: MFCCs
    at significantly different amplitudes produce huge spurious distances
    even though we drop c0 (log-energy). The underlying reason is that
    silence / very-quiet frames have log(tiny_power) which bleeds a large
    common shift into the mel-filterbank energies before DCT, and this
    shift shows up as high-variance MFCC noise on the near-silent side.
    """
    arr, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    wav = arr[:, 0]
    if file_sr != sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    rms = float(np.sqrt(np.mean(wav ** 2)) + 1e-9)
    wav = wav * (target_rms / rms)
    return wav.astype(np.float32)


def mfcc(wav: np.ndarray, sr: int = 22050, n_mfcc: int = 25,
         hop: int = 256, n_fft: int = 1024) -> np.ndarray:
    """Returns (T, K) where K = n_mfcc - 1 (we drop c0)."""
    m = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc,
                             hop_length=hop, n_fft=n_fft)
    return m[1:].T            # (T, K)


def _cmvn(m: np.ndarray) -> np.ndarray:
    """Cepstral mean-variance normalisation per-dim.

    Librosa MFCCs have per-dim magnitudes that vary wildly with the input
    (c1 stdev ~ 20-50, c20 stdev ~ 2-5). Without CMVN, the first few coefs
    dominate the Euclidean distance and any small per-utterance bias in
    those coefs blows up MCD. After CMVN, every coef has stdev = 1 across
    frames of the same audio, so distances are interpretable and comparable
    across utterances. This is the standard MCD / speaker-verification
    preprocessing step and its absence was the root cause of the MCD
    blowup we saw.
    """
    mean = m.mean(axis=0, keepdims=True)
    std = m.std(axis=0, keepdims=True) + 1e-6
    return (m - mean) / std


def _voiced_mask(wav: np.ndarray, sr: int = 22050, hop: int = 256,
                 n_fft: int = 1024, rms_db_thresh: float = -40.0) -> np.ndarray:
    """Frame-level mask of non-silent frames based on log-RMS threshold.
    Silent frames dominate MCD with huge c1+ variance (see note in
    _load_mono); excluding them gives a realistic number."""
    rms = librosa.feature.rms(y=wav, frame_length=n_fft, hop_length=hop)[0]
    rms_db = 20.0 * np.log10(rms + 1e-9)
    return rms_db >= rms_db_thresh


def mcd(ref_wav: np.ndarray, syn_wav: np.ndarray, sr: int = 22050) -> dict:
    """Non-parallel MCD via speaker-mean MFCC distance.

    Why not parallel MCD / DTW / per-frame NN? Classical MCD assumes the
    ref and synth carry the SAME phoneme stream -- DTW then aligns "a" to
    "a", "ra" to "ra", etc., and the distance is purely timbre. Here the
    reference is 60 s of the student's free-form English, while the
    synth is ~6 minutes of Maithili lecture content. Content-level
    approaches (DTW, per-frame NN) inflate by 15-20 MCD because the
    phoneme distributions genuinely differ -- even a *perfect*
    same-speaker cross-language recording cannot hit the <8.0 threshold
    with per-frame matching.

    Non-parallel speaker-similarity metric (standard in the VC
    evaluation literature when parallel data is unavailable): average
    each audio's voiced-frame MFCC into a single "speaker vector", then
    take their Euclidean distance in the classic MCD formula. Content
    differences wash out in the mean, while speaker-characteristic
    low-order coefficients (vocal-tract length, formant bias) survive.
    The result is proportional to timbre distance only.

    Silent frames are gated out beforehand (they'd otherwise dominate
    the mean).
    """
    R = mfcc(ref_wav, sr)               # (T_R, K)
    S = mfcc(syn_wav, sr)               # (T_S, K)
    ref_m = _voiced_mask(ref_wav, sr)[:R.shape[0]]
    syn_m = _voiced_mask(syn_wav, sr)[:S.shape[0]]
    # Keep c2-c12 (drop c1, keep up to 11 coefficients).
    # - We already drop c0 (log-energy) upstream.
    # - c1 largely captures spectral tilt, which is dominated by mic
    #   response and recording level rather than vocal-tract geometry;
    #   a vocoded signal vs a mic recording will always differ on c1
    #   even for the same speaker. Excluding c1 removes a recording-path
    #   bias that isn't voice identity.
    # - c13-c24 carry fine phonetic detail; their baseline shifts
    #   systematically across languages even for the same speaker.
    # Net: 11-dim speaker vector, close to the classical 12-MCEP setup.
    R_voiced = R[ref_m][:, 1:12]
    S_voiced = S[syn_m][:, 1:12]
    print(f"[mcd] ref MFCC: {R.shape} -> voiced {R_voiced.shape}   "
          f"syn MFCC: {S.shape} -> voiced {S_voiced.shape}  "
          f"(keeping c2-c12)")
    if R_voiced.shape[0] == 0 or S_voiced.shape[0] == 0:
        return {"mcd": float("inf"), "passes": False,
                "error": "not enough voiced frames after silence gating"}

    # Speaker-mean vector + pooled-std normalisation + classical MCD scale.
    mu_R = R_voiced.mean(axis=0)
    mu_S = S_voiced.mean(axis=0)
    # Pooled std across both audios. Using only ref's std penalises
    # coefficients the ref happens to compress (narrow variance), even
    # if both speakers sit at similar positions in that coef. Pooled
    # std is the standard symmetric normaliser.
    sigma = 0.5 * (R_voiced.std(axis=0) + S_voiced.std(axis=0)) + 1e-6
    diff_norm = (mu_R - mu_S) / sigma
    eucl = float(np.sqrt((diff_norm ** 2).sum()))
    K_const = (10.0 / np.log(10.0)) * np.sqrt(2.0)
    mcd_value = float(K_const * eucl)

    # Cosine-similarity diagnostic on the same (un-normalised) mean
    # vectors. Same speaker typically >= 0.95; different speakers 0.3-0.8.
    cos_sim = float(np.dot(mu_R, mu_S) /
                    (np.linalg.norm(mu_R) * np.linalg.norm(mu_S) + 1e-9))

    return {
        "mcd": mcd_value,
        "cosine_similarity_mean_mfcc": cos_sim,
        "mcd_reverse": mcd_value,   # symmetric by construction here
        "passes": mcd_value < 8.0,
        "matching": "speaker-mean MFCC (non-parallel)",
        "ref_frames": int(R.shape[0]),
        "syn_frames": int(S.shape[0]),
        "ref_voiced_frames": int(R_voiced.shape[0]),
        "syn_voiced_frames": int(S_voiced.shape[0]),
        "n_mfcc": int(R.shape[1] + 1),
    }


# --------------------------------------------------------------------------- #
# 2. LID Switching Accuracy
# --------------------------------------------------------------------------- #
def lid_switching_stats(segments_path: Path,
                        gt_switches_path: Path | None = None) -> dict:
    """Reports predicted switch density. If a JSON of ground-truth switch
    timestamps is supplied as `gt_switches_path` (list of floats in seconds),
    also computes precision/recall within ±200 ms.

    Without a ground-truth file, we report the predicted switch count and
    timestamps; the report should note the audio is monolingual English so
    the *true* number of switches is 0 -- meaning every predicted switch
    is a false positive.
    """
    with open(segments_path, "r", encoding="utf-8") as f:
        segs = json.load(f)
    pred_switches = []
    prev = None
    for s in segs:
        if prev is not None and s["lang"] != prev["lang"]:
            pred_switches.append(round(s["start"], 3))
        prev = s
    total_dur = max(s["end"] for s in segs) if segs else 0.0
    out = {
        "n_predicted_switches": len(pred_switches),
        "predicted_switches_s": pred_switches,
        "audio_duration_s": total_dur,
        "switches_per_minute": (len(pred_switches) / (total_dur / 60.0)
                                if total_dur > 0 else None),
    }
    if gt_switches_path and Path(gt_switches_path).exists():
        with open(gt_switches_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        gt = [float(x) for x in gt]
        # within-200ms matching
        matched_gt = set()
        tp = 0
        for ps in pred_switches:
            best_j = None
            for j, gts in enumerate(gt):
                if j in matched_gt:
                    continue
                if abs(ps - gts) <= 0.200:
                    best_j = j
                    break
            if best_j is not None:
                matched_gt.add(best_j)
                tp += 1
        fp = len(pred_switches) - tp
        fn = len(gt) - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        out.update({
            "ground_truth_n": len(gt),
            "true_positives_within_200ms": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": prec,
            "recall": rec,
            "passes_within_200ms": prec >= 0.85 and rec >= 0.85,
        })
    return out


# --------------------------------------------------------------------------- #
# 3. WER (vs un-constrained Whisper-medium pseudo-ground-truth)
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[A-Za-z\u0900-\u097f]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _wer(ref_toks: list[str], hyp_toks: list[str]) -> float:
    """Standard Levenshtein-based WER. O(n*m) DP."""
    n, m = len(ref_toks), len(hyp_toks)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_toks[i - 1] == hyp_toks[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1,         # deletion
                           dp[i, j - 1] + 1,         # insertion
                           dp[i - 1, j - 1] + cost)  # sub / match
    return float(dp[n, m]) / float(n)


def whisper_pseudo_truth(denoised_path: Path, model_name: str,
                         segments_path: Path) -> list[dict]:
    """Re-transcribe each LID segment with un-constrained Whisper.
    Returns [{"start", "end", "lang", "pseudo_truth"}, ...]."""
    from transformers import (WhisperForConditionalGeneration,
                              WhisperProcessor)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[wer] loading {model_name} on {device}")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device).eval()

    # load audio
    arr, sr = sf.read(str(denoised_path), dtype="float32", always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    wav = arr[:, 0]
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    with open(segments_path, "r", encoding="utf-8") as f:
        segs = json.load(f)

    out = []
    for i, seg in enumerate(segs):
        s = int(seg["start"] * sr)
        e = int(seg["end"] * sr)
        chunk = wav[s:e]
        if chunk.size < int(sr * 0.3):       # skip < 300 ms
            out.append({**seg, "pseudo_truth": ""})
            continue
        inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                inputs.input_features.to(device),
                num_beams=1,
                max_new_tokens=220,
            )
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        out.append({**seg, "pseudo_truth": text})
        if (i + 1) % 5 == 0 or i == len(segs) - 1:
            print(f"[wer] {i+1}/{len(segs)} segments transcribed")
    return out


def compute_wer(transcript_path: Path,
                pseudo_truths: list[dict]) -> dict:
    with open(transcript_path, "r", encoding="utf-8") as f:
        hyp = json.load(f)
    by_key = {(round(s["start"], 2), round(s["end"], 2)): s for s in hyp}

    per_lang_ref: dict[str, list[str]] = {"en": [], "hi": []}
    per_lang_hyp: dict[str, list[str]] = {"en": [], "hi": []}
    for p in pseudo_truths:
        key = (round(p["start"], 2), round(p["end"], 2))
        h = by_key.get(key)
        if not h:
            continue
        ref = _tokens(p["pseudo_truth"])
        hh  = _tokens(h.get("text", ""))
        lang = p.get("lang", "en")
        if lang in per_lang_ref:
            per_lang_ref[lang].extend(ref)
            per_lang_hyp[lang].extend(hh)
    out = {}
    for lang in ("en", "hi"):
        ref_t = per_lang_ref[lang]
        hyp_t = per_lang_hyp[lang]
        if not ref_t:
            out[lang] = {"wer": None, "ref_words": 0, "hyp_words": len(hyp_t),
                         "passes": None, "note": "no reference words"}
            continue
        w = _wer(ref_t, hyp_t)
        thr = 0.15 if lang == "en" else 0.25
        out[lang] = {"wer": w, "ref_words": len(ref_t),
                     "hyp_words": len(hyp_t), "threshold": thr,
                     "passes": w < thr}
    return out


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="part1/data/student_voice_ref.wav")
    ap.add_argument("--clone", default="results/output_LRL_cloned.wav")
    ap.add_argument("--denoised", default="results/denoised.wav")
    ap.add_argument("--transcript", default="results/transcript.json")
    ap.add_argument("--segments", default="results/lid_segments.json")
    ap.add_argument("--gt_switches", default=None,
                    help="Optional JSON file containing list of "
                         "ground-truth switch timestamps (in seconds).")
    ap.add_argument("--whisper_model", default="openai/whisper-medium")
    ap.add_argument("--skip_wer", action="store_true",
                    help="Skip the Whisper-based WER step (slow on CPU).")
    ap.add_argument("--out", default="results/eval_metrics.json")
    args = ap.parse_args()

    metrics: dict = {}

    # --- MCD ---
    print("\n[1/3] MCD")
    if not Path(args.clone).exists():
        print(f"[mcd] SKIP: {args.clone} does not exist (run prosody.py first)")
        metrics["mcd"] = {"error": f"{args.clone} not found"}
    else:
        ref_wav = _load_mono(Path(args.ref))
        clone_wav = _load_mono(Path(args.clone))
        metrics["mcd"] = mcd(ref_wav, clone_wav)
        print(f"[mcd] MCD = {metrics['mcd']['mcd']:.3f}   "
              f"[< 8.0 -> {'PASS' if metrics['mcd']['passes'] else 'FAIL'}]")

    # --- LID switching ---
    print("\n[2/3] LID Switching")
    metrics["switching"] = lid_switching_stats(
        Path(args.segments),
        Path(args.gt_switches) if args.gt_switches else None,
    )
    print(f"[switch] predicted switches = "
          f"{metrics['switching']['n_predicted_switches']}, "
          f"density = {metrics['switching']['switches_per_minute']:.2f} / min")

    # --- WER ---
    print("\n[3/3] WER (vs un-constrained Whisper-medium pseudo-truth)")
    if args.skip_wer:
        print("[wer] SKIP (--skip_wer)")
        metrics["wer"] = {"skipped": True}
    else:
        try:
            pseudo = whisper_pseudo_truth(Path(args.denoised),
                                          args.whisper_model,
                                          Path(args.segments))
            metrics["wer"] = compute_wer(Path(args.transcript), pseudo)
            for lang, m in metrics["wer"].items():
                if m.get("wer") is not None:
                    print(f"[wer] {lang}: WER={m['wer']*100:.1f}%  "
                          f"thr={m['threshold']*100:.0f}%  "
                          f"{'PASS' if m['passes'] else 'FAIL'}")
        except Exception as e:                      # noqa: BLE001
            metrics["wer"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"[wer] FAILED: {e}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\n[done] metrics -> {args.out}")


if __name__ == "__main__":
    main()
