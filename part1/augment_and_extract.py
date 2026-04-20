"""
Build a *harder* synthetic feature dataset for the LID model.

Motivation
----------
This script augments every clip with a randomised lecture-like acoustic
transform *before* running the backbone, so the cached features reflect
noisy/reverb/codec-corrupted audio. Aug is fixed per clip (overwritten in
place, no extra disk) -- training still stirs in feature-level noise on top.

Per-clip randomised pipeline:
    1. Gain jitter            (+/-6 dB)
    2. Speed perturbation     (0.9x / 1.0x / 1.1x)   (simple time-stretch)
    3. Synthetic reverb       (RT60 ~ 0.15-0.55 s)   (no external RIR file)
    4. Additive noise         (pink + brown at 5-25 dB SNR)
    5. Low-pass               (4-8 kHz cutoff, mimics YouTube bandwidth)
    6. Soft clipping          (tanh, mimics gain-stage compression)

Everything implemented with numpy/torch -- no ffmpeg, no external noise files.

Usage
-----
    python part1/augment_and_extract.py --data_dir part1/data/cv \
        --out_dir part1/data/cv_feats
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Model

SAMPLE_RATE = 16000
MAX_PER_LANG = 2000


# --------------------------------------------------------------------------- #
# Audio augmentation helpers (all torch, no ffmpeg)
# --------------------------------------------------------------------------- #
def _gain_jitter(w: torch.Tensor, rng: random.Random) -> torch.Tensor:
    db = rng.uniform(-6.0, 6.0)
    return w * (10.0 ** (db / 20.0))


def _speed_perturb(w: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """Pick one of {0.9, 1.0, 1.1}. 1.0 is a no-op (50% prob)."""
    choice = rng.choices([0.9, 1.0, 1.1], weights=[0.25, 0.5, 0.25])[0]
    if choice == 1.0:
        return w
    new_len = max(1, int(round(w.numel() / choice)))
    # Linear interp is good enough for LID (not ASR); torchaudio's functional
    # resample would be principled but slower.
    t = torch.linspace(0, w.numel() - 1, steps=new_len)
    i0 = t.floor().long().clamp(0, w.numel() - 1)
    i1 = (i0 + 1).clamp(0, w.numel() - 1)
    frac = (t - i0.float())
    return w[i0] * (1.0 - frac) + w[i1] * frac


def _synth_rir(rng: random.Random, sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Tiny synthetic room impulse response: direct pulse + a few sparse early
    reflections + exponentially-decaying noise tail (RT60 0.15-0.55 s)."""
    rt60 = rng.uniform(0.15, 0.55)
    length = int(rt60 * sr)
    ir = torch.zeros(length)
    ir[0] = 1.0
    # 3-6 early reflections in the first 30 ms
    for _ in range(rng.randint(3, 6)):
        idx = rng.randint(1, int(0.03 * sr))
        ir[idx] += rng.uniform(0.2, 0.6) * (1 if rng.random() < 0.5 else -1)
    # Exponential-decay white-noise tail
    tail_len = length - 1
    decay = torch.exp(-6.0 * torch.arange(tail_len) / length)  # -60 dB at RT60
    tail = torch.randn(tail_len) * decay * rng.uniform(0.02, 0.08)
    ir[1:] += tail
    # Normalise to unit energy so we don't blow the level.
    ir = ir / ir.norm().clamp_min(1e-8)
    return ir


def _reverb(w: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.3:
        return w
    ir = _synth_rir(rng)
    # FFT conv via torch
    n = w.numel() + ir.numel() - 1
    W = torch.fft.rfft(w, n=n)
    IR = torch.fft.rfft(ir, n=n)
    y = torch.fft.irfft(W * IR, n=n)[:w.numel()]
    # Balance dry/wet -- pure convolution is too wet for short clips.
    mix = rng.uniform(0.4, 0.8)
    out = (1.0 - mix) * w + mix * y
    # Re-normalise peak to avoid clipping later.
    peak = out.abs().max().clamp_min(1e-8)
    return out / peak * w.abs().max().clamp_min(1e-8)


def _pink_noise(n: int) -> torch.Tensor:
    """Voss-McCartney approximation: sum of 16 white-noise layers updated at
    halving rates. Cheap and close enough to 1/f for our purposes."""
    layers = 16
    noise = torch.zeros(n)
    for k in range(layers):
        stride = 2 ** k
        vals = torch.randn((n + stride - 1) // stride)
        noise += vals.repeat_interleave(stride)[:n]
    return noise / layers


def _brown_noise(n: int) -> torch.Tensor:
    """Random walk; heavy low-frequency content (rumble / HVAC)."""
    w = torch.randn(n).cumsum(0)
    w = w - w.mean()
    return w / w.abs().max().clamp_min(1e-8)


def _add_noise(w: torch.Tensor, rng: random.Random) -> torch.Tensor:
    noise_kind = rng.choice(["pink", "brown", "white", "pink+brown"])
    n = w.numel()
    if noise_kind == "pink":
        nz = _pink_noise(n)
    elif noise_kind == "brown":
        nz = _brown_noise(n)
    elif noise_kind == "white":
        nz = torch.randn(n)
    else:
        nz = 0.5 * _pink_noise(n) + 0.5 * _brown_noise(n)
    snr_db = rng.uniform(5.0, 25.0)
    sig_rms = w.pow(2).mean().sqrt().clamp_min(1e-8)
    nz_rms = nz.pow(2).mean().sqrt().clamp_min(1e-8)
    target_nz_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    nz = nz * (target_nz_rms / nz_rms)
    return w + nz


def _low_pass(w: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.4:
        return w
    cutoff = rng.uniform(4000.0, 8000.0)
    # Use torchaudio's biquad low-pass (single pole is too soft).
    return torchaudio.functional.lowpass_biquad(w, SAMPLE_RATE, cutoff_freq=cutoff)


def _soft_clip(w: torch.Tensor, rng: random.Random) -> torch.Tensor:
    if rng.random() < 0.5:
        return w
    drive = rng.uniform(1.2, 2.5)
    return torch.tanh(w * drive) / drive


def augment(w: torch.Tensor, rng: random.Random) -> torch.Tensor:
    w = _gain_jitter(w, rng)
    w = _speed_perturb(w, rng)
    w = _reverb(w, rng)
    w = _add_noise(w, rng)
    w = _low_pass(w, rng)
    w = _soft_clip(w, rng)
    # Final soft cap to avoid hard-clipping during the backbone forward.
    return w.clamp_(-0.99, 0.99)


# --------------------------------------------------------------------------- #
# IO
# --------------------------------------------------------------------------- #
def _collect(d: Path):
    if not d.exists():
        return []
    exts = ("wav", "flac", "mp3", "ogg", "m4a")
    files = []
    for ext in exts:
        files.extend(d.rglob(f"*.{ext}"))
        files.extend(d.rglob(f"*.{ext.upper()}"))
    return sorted(set(files))[:MAX_PER_LANG]


def _load_clip(path: Path) -> torch.Tensor:
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    w = torch.from_numpy(arr[:, 0])
    if sr != SAMPLE_RATE:
        w = torchaudio.functional.resample(w.unsqueeze(0), sr, SAMPLE_RATE)[0]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/cv")
    ap.add_argument("--out_dir", default="data/cv_feats")
    ap.add_argument("--backbone", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-extract even if the target .npy already exists.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[aug] loading backbone on {device} ...")
    model = Wav2Vec2Model.from_pretrained(args.backbone).to(device).eval()

    root = Path(args.data_dir)
    out_root = Path(args.out_dir)

    for lang_idx, lang in enumerate(("en", "hi")):
        clips = _collect(root / lang)
        if not clips:
            print(f"[aug] WARNING: no clips found in {root / lang}")
            continue
        out_dir = out_root / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(args.seed + lang_idx * 9973)

        print(f"[aug] {lang}: {len(clips)} clips -> {out_dir} (overwrite={args.overwrite})")
        for i, path in enumerate(tqdm(clips, desc=lang)):
            out_path = out_dir / f"{i:05d}.npy"
            if out_path.exists() and not args.overwrite:
                continue
            try:
                wav = _load_clip(path)
                wav = augment(wav, rng)
            except Exception as e:
                tqdm.write(f"  skip {path.name}: {e}")
                continue
            with torch.no_grad():
                feats = model(wav.unsqueeze(0).to(device)).last_hidden_state
                # feats: (1, T_frames, 1024)
            np.save(out_path, feats[0].cpu().numpy().astype(np.float16))

    print(f"\n[aug] done. Augmented features saved to {out_root}/")
    print("  Next: retrain with existing train_lid.py "
          "(feats_dir points at this same folder).")


if __name__ == "__main__":
    main()
