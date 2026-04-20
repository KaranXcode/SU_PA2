"""
Prepare a small Common Voice subset for Task 1.1 LID training.

Streams Mozilla Common Voice 16.1 from Hugging Face Hub (no full corpus
download), filters / resamples on the fly, and writes per-language 16 kHz mono
WAVs to:

    data/cv/hi/00000.wav, 00001.wav, ...        (standard Hindi)
    data/cv/en/00000.wav, 00001.wav, ...        (Indian-accented English ONLY,
                                                  to match the Hinglish "L2"
                                                  domain of the Modi clip)

This is the *training fuel* for ``train_lid.py`` -- the synthetic code-switched
montages are built by concatenating random clips from these two folders.
The 10-min Modi clip itself is the inference target and is never used here.

One-time setup:
    pip install datasets soundfile librosa tqdm
    huggingface-cli login           # free HF account; CV requires accepting ToS

Usage:
    python part1/prep_cv.py                              # 500 per language, default
    python part1/prep_cv.py --n_per_lang 800 --split train
    python part1/prep_cv.py --hi_only                    # only refresh Hindi
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

try:
    from datasets import load_dataset
except Exception as e:                                   # noqa: BLE001
    import sys as _sys
    raise SystemExit(
        "Failed to import `datasets`.\n"
        f"  Python used    : {_sys.executable}\n"
        f"  Error type     : {type(e).__name__}\n"
        f"  Error message  : {e}\n\n"
        "Likely causes:\n"
        "  (a) package is installed in a DIFFERENT interpreter. Fix:\n"
        "          python -m pip install -U datasets\n"
        "  (b) broken install / version conflict. Fix:\n"
        "          python -m pip install --force-reinstall -U datasets fsspec huggingface_hub"
    ) from e


CV_REPO = "mozilla-foundation/common_voice_16_1"
TARGET_SR = 16000

# Each LID training example is 1-3 s, so we keep clips in this window. Anything
# shorter is too sparse; anything much longer wastes disk.
MIN_DURATION_S = 1.5
MAX_DURATION_S = 8.0


def is_indian_english(row) -> bool:
    """Common Voice 'accent' field is a free-form string. We accept anything
    that looks Indian-subcontinent."""
    raw = (row.get("accent") or row.get("accents") or "")
    raw = raw.lower()
    if not raw:
        return False
    return any(tag in raw for tag in (
        "india", "indian", "south asia", "pakistan", "sri lanka", "bangladesh",
    ))


def write_clip(audio_dict, out_path: Path) -> bool:
    arr = np.asarray(audio_dict["array"], dtype=np.float32)
    src_sr = int(audio_dict["sampling_rate"])
    if arr.size == 0:
        return False
    dur = arr.size / src_sr
    if dur < MIN_DURATION_S or dur > MAX_DURATION_S:
        return False
    if src_sr != TARGET_SR:
        arr = librosa.resample(arr, orig_sr=src_sr, target_sr=TARGET_SR)
    # Normalize peak to avoid mixed loudness in the synthetic montages.
    peak = float(np.max(np.abs(arr)))
    if peak < 1e-4:
        return False
    arr = (arr / peak * 0.95).astype(np.float32)
    sf.write(str(out_path), arr, TARGET_SR, subtype="PCM_16")
    return True


def dump_language(lang: str, out_dir: Path, n_target: int,
                  split: str, accent_filter: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.wav"))
    start_idx = len(existing)
    if start_idx >= n_target:
        print(f"[{lang}] already have {start_idx} clips in {out_dir} "
              f"(>= target {n_target}); skipping.")
        return

    print(f"[{lang}] streaming {CV_REPO} ({split}) -> {out_dir} "
          f"(need {n_target - start_idx} more clips)")

    ds = load_dataset(
        CV_REPO, lang, split=split,
        streaming=True, trust_remote_code=True,
    )

    written = start_idx
    skipped = 0
    pbar = tqdm(total=n_target, initial=start_idx, desc=f"{lang}")
    for row in ds:
        if written >= n_target:
            break
        if accent_filter and not is_indian_english(row):
            skipped += 1
            continue
        out_path = out_dir / f"{written:05d}.wav"
        if out_path.exists():
            written += 1
            pbar.update(1)
            continue
        try:
            ok = write_clip(row["audio"], out_path)
        except Exception as e:                                  # noqa: BLE001
            skipped += 1
            tqdm.write(f"[{lang}] skip ({type(e).__name__}: {e})")
            continue
        if ok:
            written += 1
            pbar.update(1)
        else:
            skipped += 1
    pbar.close()
    print(f"[{lang}] done: {written} clips written, {skipped} skipped.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="data/cv",
                    help="Parent dir; subfolders 'en' and 'hi' will be filled.")
    ap.add_argument("--n_per_lang", type=int, default=500,
                    help="How many clips to keep per language.")
    ap.add_argument("--split", default="train",
                    choices=["train", "validation", "test"],
                    help="Common Voice split to stream from.")
    ap.add_argument("--en_only", action="store_true")
    ap.add_argument("--hi_only", action="store_true")
    args = ap.parse_args()

    root = Path(args.out_root)

    if not args.en_only:
        dump_language(
            lang="hi", out_dir=root / "hi",
            n_target=args.n_per_lang, split=args.split,
            accent_filter=False,
        )

    if not args.hi_only:
        dump_language(
            lang="en", out_dir=root / "en",
            n_target=args.n_per_lang, split=args.split,
            accent_filter=True,            # <- Indian English only (L2)
        )

    print("\nReady. Train the LID model with:")
    print(f"    python part1/train_lid.py --data_dir {root} "
          "--epochs 6 --out checkpoints/lid.pt")


if __name__ == "__main__":
    main()
