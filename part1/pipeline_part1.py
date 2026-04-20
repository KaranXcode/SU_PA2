"""
End-to-end Part I orchestrator:
    (1.3) Denoise  ->  (1.1) Frame-level LID  ->  (1.2) Constrained Whisper ASR

Usage:
    python pipeline_part1.py \
        --wav data/original_segment.wav \
        --ckpt checkpoints/lid.pt \
        --lm   checkpoints/lm.pkl

Assumes you have already:
    * downloaded audio        (download_audio.py)
    * trained the LID model   (train_lid.py)
    * built the N-gram LM     (ngram_lm.py build)
"""
import argparse
import json
import subprocess
from pathlib import Path


def run(cmd):
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", default="checkpoints/lid_head.pt",
                    help="LID weights. Default is the head-only 11.6 MB "
                         "file checked into git; passes checkpoints/lid.pt "
                         "(full 1.2 GB with backbone) work too.")
    ap.add_argument("--lm", default="checkpoints/lm.pkl")
    ap.add_argument("--corpus", default="syllabus.txt")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    denoised  = out / "denoised.wav"
    segments  = out / "lid_segments.json"
    transcript = out / "transcript.json"

    # 1) Denoise
    run(["python", "denoise.py", "--in", args.wav, "--out", str(denoised)])

    # 2) Frame-level LID
    run(["python", "infer_lid.py",
         "--wav", str(denoised),
         "--ckpt", args.ckpt,
         "--out", str(segments)])

    # 3) Constrained decode
    run(["python", "constrained_decode.py",
         "--wav", str(denoised),
         "--segments", str(segments),
         "--lm", args.lm,
         "--corpus", args.corpus,
         "--out", str(transcript)])

    print("\nDone. Part I outputs:")
    print("  denoised   ->", denoised)
    print("  LID JSON   ->", segments)
    print("  transcript ->", transcript)


if __name__ == "__main__":
    main()
