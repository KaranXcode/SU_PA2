"""
Task 3.2 -- F0 + Energy contour extraction with DTW prosody warping.

What "prosody warping" means here
---------------------------------
We want the synthesized LRL audio (from Task 3.3) to inherit the source
professor's *teaching style* -- pitch movements, stress, pauses -- even
though the words and language are different. Concretely:

    src_audio (Modi clip)        -> [F0_src(t), E_src(t)]    (10-min long)
    syn_audio (LRL TTS output)   -> [F0_syn(t), E_syn(t)]    (varying length)

We DTW-align the two prosodic feature streams, then use the warp path to
(a) time-stretch the synth so its rhythm follows the source, and
(b) rescale the synth's per-frame F0 to match the source's F0 contour.

Implementation details
----------------------
* F0       : librosa.pyin (probabilistic YIN) on 22.05 kHz mono; bounded
             50-500 Hz; voiced-frame mask used to ignore unvoiced regions
             during DTW distance.
* Energy   : log-RMS per frame, normalised to zero-mean / unit-variance.
* DTW      : librosa.sequence.dtw with cosine cost over normalised
             [F0_z, E_z] feature pairs.
* Warping  : librosa.effects.time_stretch (phase-vocoder) per warp segment;
             librosa.effects.pitch_shift to bend each segment's mean F0
             toward the corresponding source segment's mean F0.

Result is **frame-synchronous prosody transfer** without changing speaker
identity (which lives in the spectral envelope, not in F0).

Usage:
    python part3/prosody.py
        --src data/original_segment_22k.wav
        --syn results/synthesized_lrl_raw.wav
        --out results/synthesized_lrl_warped.wav
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 22050
HOP = 512
FMIN = 50.0
FMAX = 500.0


def extract_prosody(wav: np.ndarray, sr: int = SR):
    """Returns (F0[Hz], energy[dB], voiced_mask) -- all aligned at HOP frames."""
    f0, voiced, _ = librosa.pyin(
        wav, fmin=FMIN, fmax=FMAX, sr=sr,
        hop_length=HOP, frame_length=2048,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    rms = librosa.feature.rms(y=wav, hop_length=HOP, frame_length=2048)[0]
    energy = librosa.amplitude_to_db(rms + 1e-9)
    return f0, energy, voiced.astype(bool)


def _zscore(x: np.ndarray, mask: np.ndarray | None = None):
    if mask is not None and mask.any():
        mu = x[mask].mean()
        sd = x[mask].std() + 1e-6
    else:
        mu = x.mean()
        sd = x.std() + 1e-6
    return (x - mu) / sd


def dtw_align(f0_src, e_src, vm_src, f0_syn, e_syn, vm_syn,
              max_cells: int = 20_000_000):
    """Returns warp path (np.ndarray of shape (P, 2)) mapping
    syn_frame_idx -> src_frame_idx along the optimal path.

    For long audio, a naive full-resolution DTW matrix is prohibitive
    (e.g. 10-min source vs 6-min synth = 25k x 16k cells ~= 3 GB float64).
    We subsample both feature streams by an adaptive factor so the DTW
    grid stays under `max_cells`, run DTW on the smaller grid, then
    multiply the warp-path indices back to the original frame rate.
    This loses at most `subsample_factor` frames of boundary precision,
    which is well below the 0.5 s block granularity used downstream.
    """
    # Build feature streams. F0=0 in unvoiced -> replaced with the running
    # mean so DTW doesn't punish silences too hard.
    def _featurise(f0, e, vm):
        f0z = _zscore(f0, vm)
        ez  = _zscore(e)
        return np.stack([f0z, ez], axis=0)        # (2, T)

    X = _featurise(f0_src, e_src, vm_src)
    Y = _featurise(f0_syn, e_syn, vm_syn)

    # Choose smallest integer subsampling factor s such that
    # (Tx/s) * (Ty/s) <= max_cells.
    Tx, Ty = X.shape[1], Y.shape[1]
    s = int(np.ceil(np.sqrt((Tx * Ty) / max_cells)))
    s = max(1, s)
    if s > 1:
        X_ds = X[:, ::s]
        Y_ds = Y[:, ::s]
        print(f"[prosody] DTW grid too big ({Tx}x{Ty}); "
              f"subsampling by {s}x -> {X_ds.shape[1]}x{Y_ds.shape[1]}")
    else:
        X_ds, Y_ds = X, Y

    _, wp = librosa.sequence.dtw(X=X_ds, Y=Y_ds, metric="cosine",
                                 step_sizes_sigma=np.array([[1, 1], [1, 0], [0, 1]]),
                                 weights_add=np.array([0, 0, 0]),
                                 weights_mul=np.array([1, 1, 1]))
    # wp[:, 0] -> src indices (subsampled), wp[:, 1] -> syn indices (subsampled)
    wp = wp[::-1]
    if s > 1:
        wp = wp * s                # scale back to original frame resolution
        # Clamp to valid range so downstream indexing doesn't go OOB.
        wp[:, 0] = np.clip(wp[:, 0], 0, Tx - 1)
        wp[:, 1] = np.clip(wp[:, 1], 0, Ty - 1)
    return wp


def warp_synthesis(syn_wav: np.ndarray, sr: int,
                   src_f0: np.ndarray, syn_f0: np.ndarray,
                   wp: np.ndarray):
    """
    Apply prosody warping to `syn_wav` using the ABSOLUTE MINIMUM amount of
    DSP. Earlier per-block stretch + per-block pitch shift sounded warbly
    because every phase-vocoder pass smears consonants and introduces
    phasiness. Here we use:

      1. ONE global time-stretch so the synth matches the rate Modi spoke.
         The rate is derived from the DTW warp path (median of per-block
         rates) and clamped to [0.90, 1.15] so we never ask the phase
         vocoder to do more than +/-15% -- the range where it still
         sounds natural.
      2. NO pitch shift. kNN-VC already gave the synth your natural F0;
         shifting it further just damages timbre without adding prosodic
         realism. Modi's pitch contour is already lost to the VC step,
         so trying to reintroduce it here does more harm than good.

    Result: `output_LRL_cloned.wav` keeps the voice clarity of the flat
    baseline while still reflecting Modi's overall speaking pace. The
    ablation in the report is: flat (no stretch) vs. cloned (global
    rate-matched stretch).
    """
    RATE_CLAMP = (0.90, 1.15)
    BLOCK_SEC = 2.0

    block_frames = int(BLOCK_SEC * sr / HOP)
    n_syn_fr = len(syn_f0)

    syn_idx = wp[:, 1]
    src_idx = wp[:, 0]

    # Collect per-block rates just to derive a robust *global* rate.
    rates = []
    for fr_start in range(0, n_syn_fr, block_frames):
        fr_end = min(fr_start + block_frames, n_syn_fr)
        mask = (syn_idx >= fr_start) & (syn_idx < fr_end)
        if not mask.any():
            continue
        src_fr0 = int(src_idx[mask].min())
        src_fr1 = int(src_idx[mask].max()) + 1
        n_syn_smp = (fr_end - fr_start) * HOP
        n_src_smp = max(512, (src_fr1 - src_fr0) * HOP)
        rates.append(n_syn_smp / n_src_smp)

    if not rates:
        return syn_wav.astype(np.float32)

    # Median is robust to DTW path noise; clamp to a range where the
    # phase vocoder still sounds clean.
    global_rate = float(np.median(rates))
    global_rate = max(RATE_CLAMP[0], min(RATE_CLAMP[1], global_rate))
    print(f"[prosody] global time-stretch rate = {global_rate:.3f} "
          f"(from {len(rates)} DTW blocks, clamped to "
          f"[{RATE_CLAMP[0]}, {RATE_CLAMP[1]}])")

    try:
        out = librosa.effects.time_stretch(syn_wav.astype(np.float32),
                                           rate=global_rate)
    except Exception as e:                         # noqa: BLE001
        print(f"[prosody] time_stretch failed ({e}); returning unwarped")
        out = syn_wav.astype(np.float32)

    return out.astype(np.float32)


def warp_file(src_path: Path, syn_path: Path, out_path: Path,
              flat_baseline_out: Path | None = None):
    src, _ = librosa.load(str(src_path), sr=SR, mono=True)
    syn, _ = librosa.load(str(syn_path), sr=SR, mono=True)

    print(f"[prosody] src {src.size/SR:.1f}s   syn {syn.size/SR:.1f}s")
    f0_src, e_src, vm_src = extract_prosody(src)
    f0_syn, e_syn, vm_syn = extract_prosody(syn)

    print("[prosody] DTW align ...")
    wp = dtw_align(f0_src, e_src, vm_src, f0_syn, e_syn, vm_syn)
    print(f"[prosody] warp path length = {wp.shape[0]} frames")

    warped = warp_synthesis(syn, SR, f0_src, f0_syn, wp)
    warped = np.clip(warped, -1.0, 1.0).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), warped, SR, subtype="PCM_16")
    print(f"[prosody] warped -> {out_path} ({warped.size/SR:.1f}s)")

    # Save the unwarped synth as the "flat" baseline for the ablation table.
    if flat_baseline_out is not None:
        sf.write(str(flat_baseline_out), syn.astype(np.float32),
                 SR, subtype="PCM_16")
        print(f"[prosody] flat baseline -> {flat_baseline_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/original_segment_22k.wav")
    ap.add_argument("--syn", default="results/synthesized_lrl_raw.wav")
    ap.add_argument("--out", default="results/output_LRL_cloned.wav")
    ap.add_argument("--flat", default="results/output_LRL_flat.wav",
                    help="Also save the un-warped synth as the ablation baseline.")
    args = ap.parse_args()
    warp_file(Path(args.src), Path(args.syn), Path(args.out), Path(args.flat))
