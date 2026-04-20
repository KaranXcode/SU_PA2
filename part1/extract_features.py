"""
Pre-extract Wav2Vec2-XLS-R features for all clips in data/cv/{en,hi}.

This makes LID training ~100x faster per step because the frozen 300M-param
backbone runs only ONCE per clip, not every training step.

Usage:
    python extract_features.py --data_dir data/cv --out_dir data/cv_feats
    python train_lid.py --data_dir data/cv --feats_dir data/cv_feats --epochs 10 ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import Wav2Vec2Model

SAMPLE_RATE = 16000
MAX_PER_LANG = 2000


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
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[extract] loading backbone on {device} ...")
    model = Wav2Vec2Model.from_pretrained(args.backbone).to(device).eval()

    root = Path(args.data_dir)
    out_root = Path(args.out_dir)

    for lang in ("en", "hi"):
        clips = _collect(root / lang)
        if not clips:
            print(f"[extract] WARNING: no clips found in {root / lang}")
            continue
        out_dir = out_root / lang
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[extract] {lang}: {len(clips)} clips -> {out_dir}")
        for i, path in enumerate(tqdm(clips, desc=lang)):
            out_path = out_dir / f"{i:05d}.npy"
            if out_path.exists():
                continue
            try:
                wav = _load_clip(path)
            except Exception as e:
                tqdm.write(f"  skip {path.name}: {e}")
                continue
            with torch.no_grad():
                feats = model(wav.unsqueeze(0).to(device)).last_hidden_state
                # feats: (1, T_frames, 1024)
            np.save(out_path, feats[0].cpu().numpy().astype(np.float16))

    print(f"\n[extract] done. Features saved to {out_root}/")
    print(f"  Now train with:\n"
          f"    python train_lid.py --data_dir {root} --feats_dir {out_root} "
          f"--epochs 10 --n_items 2000 --batch 8 --out ../checkpoints/lid.pt")


if __name__ == "__main__":
    main()
