"""
Post-download audio prep.

Takes a manually-placed 10-minute source WAV (any sample rate, mono or stereo)
and produces the two canonical inputs the downstream pipeline expects:

    <dir>/original_segment.wav        -- 16 kHz mono   (ASR / LID)
    <dir>/original_segment_22k.wav    -- 22.05 kHz mono (TTS)

No YouTube / ffmpeg / yt-dlp involved -- the user supplies the source clip.

Usage
-----
    python part1/prepare_audio.py --in part1/data/original_segment.wav
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[1] > 1:
        return arr.mean(axis=1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def _resample(wav: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return wav.astype(np.float32, copy=False)
    g = np.gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return resample_poly(wav, up=up, down=down).astype(np.float32)


def _write(path: Path, wav: np.ndarray, sr: int):
    wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
    sf.write(str(path), wav, sr, subtype="PCM_16")
    print(f"[wrote] {path}  ({wav.size / sr:.2f}s @ {sr} Hz, mono)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="part1/data/original_segment.wav",
                    help="Path to the manually-placed 10-minute source WAV.")
    args = ap.parse_args()

    src = Path(args.inp)
    if not src.exists():
        raise SystemExit(f"[error] source not found: {src}")

    arr, sr = sf.read(str(src), dtype="float32", always_2d=True)
    mono = _to_mono(arr)
    print(f"[read]  {src}  sr={sr}  channels={arr.shape[1]}  "
          f"dur={mono.size / sr:.2f}s")

    out_16k = src.with_name("original_segment.wav")
    _write(out_16k, _resample(mono, sr, 16000), 16000)

    out_22k = src.with_name("original_segment_22k.wav")
    _write(out_22k, _resample(mono, sr, 22050), 22050)


if __name__ == "__main__":
    main()
