"""
Task 1.3  --  Denoising & Normalization.

Primary:   DeepFilterNet (if installed) -- state of the art DNN denoiser.
Fallback:  Classical Spectral Subtraction (Boll, 1979) implemented in PyTorch.

Also:      RMS loudness normalization + optional high-pass @ 80 Hz to kill room rumble.

Usage:
    python denoise.py --in data/original_segment.wav --out data/denoised.wav
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as AF


# --------------------------------------------------------------------------- #
# Spectral subtraction (Boll 1979) -- PyTorch implementation.
# --------------------------------------------------------------------------- #
def spectral_subtraction(
    wav: torch.Tensor,
    sr: int,
    noise_duration: float = 0.5,
    n_fft: int = 1024,
    hop_length: int = 256,
    alpha: float = 2.0,       # over-subtraction factor
    beta: float = 0.02,       # spectral floor
) -> torch.Tensor:
    """
    wav: (1, T) mono. Uses the first `noise_duration` seconds as a noise profile.
    Y_clean(f) = max( |Y(f)| - alpha * |N(f)|,  beta * |Y(f)| ) * e^{j phi_Y(f)}
    """
    assert wav.ndim == 2 and wav.size(0) == 1
    window = torch.hann_window(n_fft, device=wav.device)

    spec = torch.stft(
        wav[0], n_fft=n_fft, hop_length=hop_length, window=window,
        return_complex=True, center=True,
    )                                   # (F, T)
    mag = spec.abs()
    phase = torch.angle(spec)

    n_noise_frames = max(1, int(noise_duration * sr / hop_length))
    noise_mag = mag[:, :n_noise_frames].mean(dim=1, keepdim=True)  # (F, 1)

    clean_mag = torch.maximum(mag - alpha * noise_mag, beta * mag)
    clean_spec = clean_mag * torch.exp(1j * phase)

    clean = torch.istft(
        clean_spec, n_fft=n_fft, hop_length=hop_length, window=window,
        length=wav.size(1), center=True,
    )
    return clean.unsqueeze(0)


# --------------------------------------------------------------------------- #
# DeepFilterNet wrapper.
# --------------------------------------------------------------------------- #
def try_deepfilternet(wav: torch.Tensor, sr: int):
    try:
        from df.enhance import enhance, init_df
    except ImportError:
        return None
    model, df_state, _ = init_df()
    # DeepFilterNet expects 48 kHz
    if sr != df_state.sr():
        wav48 = AF.resample(wav, sr, df_state.sr())
    else:
        wav48 = wav
    enhanced = enhance(model, df_state, wav48)
    if sr != df_state.sr():
        enhanced = AF.resample(enhanced, df_state.sr(), sr)
    return enhanced


# --------------------------------------------------------------------------- #
# Post-processing: high-pass + loudness normalization.
# --------------------------------------------------------------------------- #
def normalize(wav: torch.Tensor, sr: int, target_rms_db: float = -23.0) -> torch.Tensor:
    wav = AF.highpass_biquad(wav, sr, cutoff_freq=80.0)
    rms = wav.pow(2).mean().sqrt().clamp_min(1e-9)
    target_rms = 10 ** (target_rms_db / 20)
    return (wav * (target_rms / rms)).clamp(-1.0, 1.0)


def denoise_file(in_path: Path, out_path: Path, method: str = "auto"):
    # soundfile-based I/O (avoids torchaudio's new torchcodec dependency).
    arr, sr = sf.read(str(in_path), dtype="float32", always_2d=True)  # (T, C)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    wav = torch.from_numpy(arr[:, 0]).unsqueeze(0)                    # (1, T)

    enhanced = None
    if method in ("auto", "dfn"):
        enhanced = try_deepfilternet(wav, sr)
        if enhanced is not None:
            print("[denoise] using DeepFilterNet")

    if enhanced is None:
        print("[denoise] falling back to spectral subtraction")
        enhanced = spectral_subtraction(wav, sr)

    enhanced = normalize(enhanced, sr)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), enhanced[0].cpu().numpy(), sr)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=["auto", "dfn", "specsub"], default="auto")
    args = ap.parse_args()
    denoise_file(Path(args.inp), Path(args.out), args.method)
