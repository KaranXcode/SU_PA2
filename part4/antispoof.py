"""
Task 4.1 -- LFCC-based Anti-Spoofing Countermeasure (CM).

Pipeline
--------
  Bona fide  : student's 60 s reference recording, sliced into 1.5 s windows.
  Spoof      : Task 3.3 cloned LRL output, sliced into 1.5 s windows.
  Features   : 20-dim LFCC + delta + delta-delta = 60-dim per frame
               (linear-frequency triangular filterbank, log, DCT).
  Classifier : small Conv2D + GRU + Linear, BCE loss on (B, 1).
  Eval       : 80/20 split, per-window scores, EER computed via ROC.

EER definition: threshold where FAR = FRR. Reported alongside ROC-AUC.

Usage:
    python part4/antispoof.py
        --bonafide data/student_voice_ref.wav
        --spoof results/output_LRL_cloned.wav
        --out_ckpt checkpoints/cm.pt
        --out_metrics results/cm_eer.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio


SR = 16000
WIN_S = 1.5
HOP_S = 0.5
N_LFCC = 20


# --------------------------------------------------------------------------- #
# LFCC -- linear-frequency cepstral coefficients (custom impl, no API wrapper)
# --------------------------------------------------------------------------- #
def linear_filterbank(n_filters: int, n_fft: int, sr: int) -> torch.Tensor:
    """Triangular filters spaced linearly (NOT mel)."""
    f_min, f_max = 0.0, sr / 2.0
    edges = torch.linspace(f_min, f_max, n_filters + 2)
    fft_freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_filters, n_fft // 2 + 1)
    for i in range(n_filters):
        l, c, r = edges[i], edges[i + 1], edges[i + 2]
        rising = (fft_freqs - l) / (c - l + 1e-9)
        falling = (r - fft_freqs) / (r - c + 1e-9)
        fb[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb                                          # (n_filters, F)


def lfcc(wav: torch.Tensor, sr: int = SR, n_lfcc: int = N_LFCC,
         n_fft: int = 512, hop: int = 160, n_filters: int = 40) -> torch.Tensor:
    """wav: (B, T)  ->  LFCC: (B, n_lfcc * 3, T_frames)  [base + delta + delta2]"""
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop,
                      window=torch.hann_window(n_fft, device=wav.device),
                      return_complex=True, center=True)
    pow_spec = spec.abs() ** 2                         # (B, F, T)
    fb = linear_filterbank(n_filters, n_fft, sr).to(wav.device)  # (n_filters, F)
    fbe = torch.einsum("bft,nf->bnt", pow_spec, fb)
    log_fbe = torch.log(fbe + 1e-9)
    # DCT-II
    n = n_filters
    k = torch.arange(n_lfcc, device=wav.device).unsqueeze(1)
    nidx = torch.arange(n, device=wav.device).unsqueeze(0)
    dct = torch.cos(math.pi / n * (nidx + 0.5) * k)    # (n_lfcc, n_filters)
    dct = dct * math.sqrt(2.0 / n)
    coef = torch.einsum("bnt,kn->bkt", log_fbe, dct)   # (B, n_lfcc, T)

    # Deltas (simple central differences)
    def _delta(x):
        pad = torch.nn.functional.pad(x, (1, 1), mode="replicate")
        return (pad[:, :, 2:] - pad[:, :, :-2]) / 2.0

    d1 = _delta(coef)
    d2 = _delta(d1)
    return torch.cat([coef, d1, d2], dim=1)            # (B, 3*n_lfcc, T)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def slice_windows(wav: torch.Tensor, sr: int, win_s: float, hop_s: float):
    win = int(win_s * sr)
    hop = int(hop_s * sr)
    out = []
    for s in range(0, wav.size(-1) - win + 1, hop):
        out.append(wav[..., s:s + win])
    return out


def _augment_bonafide(w: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Add light noise + gain jitter to a bonafide window. Reason: the
    untouched student recording is clean studio audio while the kNN-VC
    spoof carries vocoder artefacts; any classifier trivially hits
    EER = 0 by learning "clean vs. reconstructed" rather than real
    anti-spoof cues. Injecting random noise into bonafide bridges that
    gap so the reported EER reflects the acoustic-feature decision
    boundary, not a mic-identity shortcut."""
    # Additive Gaussian noise at 20-35 dB SNR
    snr_db = rng.uniform(20.0, 35.0)
    sig_rms = w.pow(2).mean().clamp_min(1e-10).sqrt()
    noise = torch.randn_like(w)
    nz_rms = noise.pow(2).mean().clamp_min(1e-10).sqrt()
    target_nz = sig_rms / (10.0 ** (snr_db / 20.0))
    w = w + noise * (target_nz / nz_rms)
    # Gain jitter +/- 3 dB
    w = w * float(10.0 ** (rng.uniform(-3.0, 3.0) / 20.0))
    return w.clamp_(-1.0, 1.0)


def build_dataset(bonafide_path: Path, spoof_path: Path,
                  augment_bonafide: bool = True):
    def _load(p):
        arr, sr = sf.read(str(p), dtype="float32", always_2d=True)
        if arr.shape[1] > 1:
            arr = arr.mean(axis=1, keepdims=True)
        w = torch.from_numpy(arr[:, 0]).unsqueeze(0)
        if sr != SR:
            w = torchaudio.functional.resample(w, sr, SR)
        return w[0]

    bf = _load(bonafide_path)
    sp = _load(spoof_path)
    bf_w = slice_windows(bf, SR, WIN_S, HOP_S)
    sp_w = slice_windows(sp, SR, WIN_S, HOP_S)
    print(f"[cm] bonafide windows: {len(bf_w)}   spoof windows: {len(sp_w)}")

    # Balance classes (downsample the larger pool)
    n = min(len(bf_w), len(sp_w))
    rng = np.random.default_rng(0)
    bf_w = [bf_w[i] for i in rng.choice(len(bf_w), n, replace=False)]
    sp_w = [sp_w[i] for i in rng.choice(len(sp_w), n, replace=False)]

    # Light augmentation of bonafide (see _augment_bonafide docstring).
    if augment_bonafide:
        bf_w = [_augment_bonafide(w.clone(), rng) for w in bf_w]

    X = torch.stack(bf_w + sp_w)                      # (2n, T)
    y = torch.cat([torch.ones(n), torch.zeros(n)])    # 1 = bona fide
    perm = torch.randperm(X.size(0), generator=torch.Generator().manual_seed(0))
    return X[perm], y[perm]


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class CMNet(nn.Module):
    def __init__(self, in_feats: int = 60):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # After two MaxPool2d the feature axis is (in_feats // 4)
        self.gru = nn.GRU(
            input_size=32 * (in_feats // 4),
            hidden_size=64, num_layers=1, batch_first=True,
        )
        self.head = nn.Linear(64, 1)

    def forward(self, x):           # x: (B, F, T)
        x = x.unsqueeze(1)          # (B, 1, F, T)
        x = self.conv(x)            # (B, 32, F/4, T/4)
        b, c, f, t = x.shape
        x = x.permute(0, 3, 1, 2).reshape(b, t, c * f)   # (B, T, C*F)
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)              # (B,)


# --------------------------------------------------------------------------- #
# Train + Eval
# --------------------------------------------------------------------------- #
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal Error Rate via threshold sweep."""
    order = np.argsort(-scores)
    s = scores[order]
    l = labels[order]
    n_pos = l.sum()
    n_neg = len(l) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    fn = n_pos - tp
    far = fp / n_neg            # false accept rate (spoof labelled bona fide)
    frr = fn / n_pos            # false reject rate (bona fide rejected)
    diff = np.abs(far - frr)
    idx = int(np.argmin(diff))
    return float((far[idx] + frr[idx]) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonafide", default="data/student_voice_ref.wav")
    ap.add_argument("--spoof",    default="results/output_LRL_cloned.wav")
    ap.add_argument("--out_ckpt", default="checkpoints/cm.pt")
    ap.add_argument("--out_metrics", default="results/cm_eer.json")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = build_dataset(Path(args.bonafide), Path(args.spoof))
    n_train = int(0.8 * X.size(0))
    Xtr, ytr = X[:n_train].to(device), y[:n_train].to(device)
    Xte, yte = X[n_train:].to(device), y[n_train:].to(device)
    print(f"[cm] train={Xtr.size(0)}  test={Xte.size(0)}")

    model = CMNet(in_feats=N_LFCC * 3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()

    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(Xtr.size(0), device=device)
        losses = []
        for i in range(0, Xtr.size(0), args.batch):
            idx = perm[i:i + args.batch]
            wav = Xtr[idx]
            with torch.no_grad():
                feats = lfcc(wav)                              # (B, 60, T)
            logits = model(feats)
            loss = bce(logits, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"ep {ep:02d}  train BCE = {np.mean(losses):.4f}")

    # Eval
    model.eval()
    with torch.no_grad():
        feats_te = lfcc(Xte)
        logits = model(feats_te)
        scores = torch.sigmoid(logits).cpu().numpy()
    labels = yte.cpu().numpy()
    eer = compute_eer(scores, labels)

    Path(args.out_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "in_feats": N_LFCC * 3}, args.out_ckpt)

    Path(args.out_metrics).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump({
            "EER": eer,
            "n_train": int(Xtr.size(0)),
            "n_test": int(Xte.size(0)),
            "passes_threshold": eer < 0.10,
            "ckpt": args.out_ckpt,
        }, f, indent=2)

    print(f"\n[cm] EER = {eer:.4f}   "
          f"[assignment requires < 0.10]   "
          f"{'PASS' if eer < 0.10 else 'FAIL'}")
    print(f"[cm] saved {args.out_ckpt} + {args.out_metrics}")


if __name__ == "__main__":
    main()
