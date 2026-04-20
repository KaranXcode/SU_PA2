"""
Stop-gap transcript producer so Part II / III / IV can be developed before
Part I's full LID + constrained-decoding pipeline is ready.

Runs Whisper-v3 directly on the 10-min Modi clip with `language="hi"` (Whisper
auto-handles English code-switches when the primary language is set to Hindi).
Then crudely tags each chunk's `lang` field by Devanagari content:
    any Devanagari char in the output -> 'hi', else 'en'.

This is NOT the final Part I output -- the constrained Whisper + custom LID
pipeline ([infer_lid.py] + [constrained_decode.py]) supersedes this. But it
produces a transcript JSON in the same shape so Part II/III/IV can run today.

Usage:
    python part2/quick_transcript.py
        [--wav data/original_segment.wav] [--out results/transcript.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def is_devanagari(s: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default="data/original_segment.wav")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--language", default="hi",
                    help="Primary language hint to Whisper (hi handles "
                         "Hinglish well).")
    ap.add_argument("--out", default="results/transcript.json")
    args = ap.parse_args()

    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    print(f"[whisper] loading {args.model} on "
          f"{'cuda:0' if device >= 0 else 'cpu'}")
    asr = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        chunk_length_s=30,
        stride_length_s=5,
        return_timestamps=True,
        device=device,
    )

    print(f"[whisper] transcribing {args.wav} ...")
    out = asr(
        args.wav,
        generate_kwargs={"language": args.language, "task": "transcribe"},
    )

    chunks = out.get("chunks", [])
    segments = []
    for ch in chunks:
        ts = ch.get("timestamp", (None, None))
        text = (ch.get("text") or "").strip()
        if not text or ts[0] is None:
            continue
        end = ts[1] if ts[1] is not None else (ts[0] + 1.0)
        lang = "hi" if is_devanagari(text) else "en"
        segments.append({
            "start": float(ts[0]),
            "end": float(end),
            "lang": lang,
            "text": text,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"[whisper] wrote {len(segments)} segments -> {args.out}")


if __name__ == "__main__":
    main()
