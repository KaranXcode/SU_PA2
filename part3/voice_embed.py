"""
Task 3.1 -- Speaker embedding extraction (x-vector or d-vector).

Backend: SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`),
which produces a 192-d x-vector. This is a frozen speaker encoder; we are
NOT calling its training loop, only its embedding extractor -- the assignment
explicitly allows pre-trained extractors for embedding generation.

If SpeechBrain is unavailable, falls back to a torchaudio Wav2Vec2-based
mean-pooled embedding (1024-d). Less discriminative for speaker ID but
keeps the pipeline runnable.

Usage:
    python part3/voice_embed.py --ref data/student_voice_ref.wav \\
        --out results/student_xvector.npy
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def load_mono_16k(path: Path) -> torch.Tensor:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    wav = torch.from_numpy(arr[:, 0]).unsqueeze(0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    return wav


def _try_speechbrain(wav: torch.Tensor) -> np.ndarray | None:
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError:
        return None
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="checkpoints/spkr_ecapa",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    emb = enc.encode_batch(wav).squeeze().detach().cpu().numpy()
    print(f"[voice_embed] ECAPA-TDNN x-vector: dim={emb.shape}")
    return emb


def _wav2vec2_fallback(wav: torch.Tensor) -> np.ndarray:
    print("[voice_embed] SpeechBrain not found -> using Wav2Vec2 mean-pool fallback")
    bundle = torchaudio.pipelines.WAV2VEC2_LARGE
    model = bundle.get_model().eval()
    with torch.no_grad():
        feats, _ = model.extract_features(wav)
    # take last layer, mean-pool over time
    emb = feats[-1].mean(dim=1).squeeze().cpu().numpy()
    print(f"[voice_embed] Wav2Vec2 mean-pool d-vector: dim={emb.shape}")
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/student_voice_ref.wav",
                    help="Path to your 60s reference recording (mono).")
    ap.add_argument("--out", default="results/student_xvector.npy")
    args = ap.parse_args()

    ref = Path(args.ref)
    if not ref.exists():
        raise SystemExit(
            f"Reference not found: {ref}\n"
            f"Record exactly 60 s of your voice to that path "
            f"(any sample rate; we'll resample to 16 kHz)."
        )

    wav = load_mono_16k(ref)
    dur = wav.size(-1) / 16000.0
    print(f"[voice_embed] {ref}: {dur:.1f}s")
    if dur < 30 or dur > 90:
        print(f"[voice_embed] WARNING: assignment expects ~60s; got {dur:.1f}s")

    emb = _try_speechbrain(wav)
    if emb is None:
        emb = _wav2vec2_fallback(wav)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, emb)
    print(f"[voice_embed] saved -> {args.out}")


if __name__ == "__main__":
    main()
