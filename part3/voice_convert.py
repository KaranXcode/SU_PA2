"""
Task 3 addition -- kNN Voice Conversion (Baas et al., 2023).

Why this exists
---------------
The assignment requires the final audio to match the *student's* voice
(MCD < 8.0 vs student_voice_ref.wav). MMS-TTS is single-speaker per language
and does not accept speaker conditioning, so its raw output carries the MMS
default speaker's timbre. `prosody.py` transfers F0 / rhythm but NOT timbre
(timbre lives in the spectral envelope, which pitch/time warps leave mostly
intact). This script closes that gap.

How kNN-VC works
----------------
    1. Extract frame-level WavLM self-supervised features from:
         query     = synthesized LRL audio  (content / phonemes)
         ref pool  = student's 60 s recording (target timbre)
    2. For each query frame, find its top-k nearest neighbours (cosine) in
       the ref pool and average them.
    3. Feed the rematched feature stream into a HiFi-GAN vocoder pretrained
       on WavLM features. Output = student's voice speaking the MMS content.

Pipeline position
-----------------
    synthesize_lrl.py (MMS-TTS)
        -> synthesized_lrl_raw.wav          (MMS default speaker)
    voice_convert.py    (this file)
        -> synthesized_lrl_vc.wav           (student voice, no prosody warp)
    prosody.py
        -> output_LRL_flat.wav              (ablation: VC, no prosody)
        -> output_LRL_cloned.wav            (VC + Modi-style prosody)

Usage
-----
    python part3/voice_convert.py
        --content results/synthesized_lrl_raw.wav
        --ref part1/data/student_voice_ref.wav
        --out results/synthesized_lrl_vc.wav
        [--k 4]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

SR_WAVLM = 16000          # kNN-VC / WavLM native rate
SR_OUT = 22050            # Assignment requirement + prosody.py input rate


# --------------------------------------------------------------------------- #
# torchaudio.load monkey-patch
# --------------------------------------------------------------------------- #
# torchaudio 2.8 dispatches `load()` through `torchcodec`, which is not a
# default dependency on Windows and pulls in ffmpeg. The kNN-VC `matcher.py`
# deep inside the repo calls `torchaudio.load(path, normalize=True)`
# directly, so we replace that symbol with a thin soundfile wrapper before
# loading the model. The call signature matches: returns (waveform, sr)
# where waveform is a (channels, samples) float32 tensor in [-1, 1].
def _load_with_soundfile(path, normalize=True, **_kwargs):
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # soundfile returns (samples, channels); torchaudio expects (channels, samples).
    wav = torch.from_numpy(arr.T.copy())          # copy = contiguous
    return wav, sr


torchaudio.load = _load_with_soundfile


def _load_mono_16k(path: Path) -> torch.Tensor:
    """Load a WAV as 16 kHz mono float32 torch tensor (1D)."""
    arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if arr.shape[1] > 1:
        arr = arr.mean(axis=1, keepdims=True)
    w = torch.from_numpy(arr[:, 0])
    if sr != SR_WAVLM:
        import torchaudio
        w = torchaudio.functional.resample(w.unsqueeze(0), sr, SR_WAVLM)[0]
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True,
                    help="Audio whose phonetic/linguistic content to preserve "
                         "(normally the MMS-TTS output).")
    ap.add_argument("--ref", required=True,
                    help="Audio whose voice/timbre to match "
                         "(student's 60 s recording).")
    ap.add_argument("--out", required=True,
                    help="Output WAV at 22.05 kHz.")
    ap.add_argument("--k", type=int, default=4,
                    help="k for kNN matching (4 is the paper's default).")
    ap.add_argument("--chunk_s", type=float, default=20.0,
                    help="Split the content audio into this-many-second chunks "
                         "before feeding WavLM. WavLM's conv stack OOMs on "
                         "multi-minute inputs (~1.2 GB of activations); 20 s "
                         "stays well under 500 MB on CPU.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[vc] loading bshall/knn-vc via torch.hub on {device} ...")
    # prematched=True uses the HiFi-GAN checkpoint trained on already-matched
    # features (higher fidelity than the non-prematched variant).
    knn_vc = torch.hub.load(
        "bshall/knn-vc", "knn_vc",
        prematched=True, trust_repo=True, pretrained=True,
        device=device,
    )

    content_path = Path(args.content)
    ref_path = Path(args.ref)
    if not content_path.exists():
        raise SystemExit(f"[error] content not found: {content_path}")
    if not ref_path.exists():
        raise SystemExit(f"[error] reference not found: {ref_path}")

    # The matching set is built from the full student reference (60 s is
    # fine, WavLM's conv stack stays <500 MB on that length).
    print(f"[vc] building matching set from ref = {ref_path.name}")
    matching_set = knn_vc.get_matching_set([str(ref_path)])

    # WavLM OOMs on the full ~6-minute content (conv stack produces ~1.2 GB
    # activations). Chunk the query at 20 s, run VC per chunk, concatenate
    # the 16 kHz output with a short crossfade to avoid audible seams.
    CHUNK_S = args.chunk_s
    XFADE_S = 0.1
    content_16k = _load_mono_16k(content_path).numpy()
    n = content_16k.size
    chunk_len = int(CHUNK_S * SR_WAVLM)
    xfade_len = int(XFADE_S * SR_WAVLM)

    import tempfile
    out_chunks = []
    starts = list(range(0, n, chunk_len))
    print(f"[vc] content = {n / SR_WAVLM:.1f}s, {len(starts)} chunks "
          f"of up to {CHUNK_S}s each")
    for i, s in enumerate(starts):
        e = min(s + chunk_len, n)
        sub = content_16k[s:e]
        # Guard against very short tail chunks (WavLM needs enough frames).
        if sub.size < SR_WAVLM:   # < 1 s -> pad with the tail of previous
            sub = np.concatenate([content_16k[max(0, s - SR_WAVLM):s], sub])
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            sf.write(tmp_path, sub, SR_WAVLM, subtype="PCM_16")
            q = knn_vc.get_features(tmp_path)
            out = knn_vc.match(q, matching_set, topk=args.k)
            out_np_chunk = out.detach().cpu().numpy().astype(np.float32)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
        print(f"[vc]   chunk {i+1}/{len(starts)}: "
              f"{sub.size / SR_WAVLM:.1f}s -> {out_np_chunk.size / SR_WAVLM:.1f}s")
        out_chunks.append(out_np_chunk)
        # Free memory between chunks.
        del q, out, out_np_chunk
        import gc
        gc.collect()

    # Crossfade-stitch at chunk boundaries.
    stitched = out_chunks[0].copy()
    for nxt in out_chunks[1:]:
        if stitched.size >= xfade_len and nxt.size >= xfade_len:
            a = stitched[-xfade_len:]
            b = nxt[:xfade_len]
            alpha = np.linspace(0.0, 1.0, xfade_len, dtype=np.float32)
            mixed = (1.0 - alpha) * a + alpha * b
            stitched = np.concatenate([stitched[:-xfade_len], mixed, nxt[xfade_len:]])
        else:
            stitched = np.concatenate([stitched, nxt])

    # Resample 16 kHz -> 22.05 kHz so prosody.py (22.05 kHz internally)
    # doesn't have to up-sample again.
    out_t = torch.from_numpy(stitched).unsqueeze(0)
    out_t = torchaudio.functional.resample(out_t, SR_WAVLM, SR_OUT)
    out_np = out_t.squeeze(0).numpy().astype(np.float32)

    out_np = np.clip(out_np, -1.0, 1.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, out_np, SR_OUT, subtype="PCM_16")
    print(f"[vc] wrote {out_np.size / SR_OUT:.1f}s @ {SR_OUT} Hz -> {args.out}")


if __name__ == "__main__":
    main()
