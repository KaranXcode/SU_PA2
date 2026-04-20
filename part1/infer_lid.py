"""
Run the trained FrameLID model on a long audio file and emit a JSON
of segments: [{"start": s, "end": s, "lang": "en"|"hi"|"sil"}, ...]

Long audio is processed in 30-second chunks with 2 s overlap; overlapping
frame logits are averaged before decoding (avoids boundary artefacts).

Usage:
    python infer_lid.py --wav data/denoised.wav --ckpt checkpoints/lid.pt \
        --out data/lid_segments.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from lid_model import FrameLID, decode_segments, FRAME_HZ

SAMPLE_RATE = 16000


@torch.no_grad()
def infer(wav_path, ckpt_path, chunk_s=30.0, overlap_s=2.0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameLID().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    # strict=False: the checkpoint may be a head-only file (backbone stripped
    # for GitHub's 100 MB limit; see part1/strip_ckpt.py). The fresh Wav2Vec2
    # backbone is already loaded from HuggingFace at FrameLID.__init__, so
    # missing "backbone.*" keys are expected and safe.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    arr, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    wav = torch.from_numpy(arr[:, 0]).unsqueeze(0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    wav = wav[0]

    chunk = int(chunk_s * SAMPLE_RATE)
    hop   = int((chunk_s - overlap_s) * SAMPLE_RATE)
    total_frames = int(np.ceil(wav.size(0) / SAMPLE_RATE * FRAME_HZ))

    logits_sum = torch.zeros(total_frames, 3)
    switch_sum = torch.zeros(total_frames)
    counts     = torch.zeros(total_frames)

    for start in range(0, wav.size(0), hop):
        end = min(start + chunk, wav.size(0))
        seg = wav[start:end].unsqueeze(0).to(device)
        logits, sw = model(seg)
        logits = logits[0].cpu()
        sw     = sw[0].cpu()
        f_start = start * FRAME_HZ // SAMPLE_RATE
        f_end   = f_start + logits.size(0)
        f_end   = min(f_end, total_frames)
        n       = f_end - f_start
        logits_sum[f_start:f_end] += logits[:n]
        switch_sum[f_start:f_end] += sw[:n]
        counts[f_start:f_end]     += 1
        if end == wav.size(0):
            break

    counts = counts.clamp_min(1)
    avg_logits = logits_sum / counts.unsqueeze(-1)
    avg_switch = switch_sum / counts

    segments = decode_segments(avg_logits, avg_switch)
    return segments


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="checkpoints/lid.pt")
    ap.add_argument("--out", default="results/lid_segments.json")
    args = ap.parse_args()

    segs = infer(args.wav, args.ckpt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out = [{"start": round(s, 3), "end": round(e, 3), "lang": l}
           for (s, e, l) in segs]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} segments -> {args.out}")
