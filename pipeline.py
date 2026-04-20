"""
End-to-end pipeline for Speech Understanding Assignment 2.

Run order:
    Part 0 -- prepare_audio.py        (mono 16k + 22.05k from user-supplied WAV)
    Part I -- pipeline_part1.py       (denoise -> LID -> constrained Whisper)
    Part II  -- pipeline_part2.py     (G2P/IPA + LRL translation)
    Part III -- pipeline_part3.py     (voice embed -> MMS-TTS -> kNN-VC -> prosody)
    Part IV  -- pipeline_part4.py     (anti-spoofing CM + PGD adversarial on LID)
    eval_metrics.py                   (MCD + WER + LID switch density)

Each sub-step writes to results/ and is independently re-runnable. This script
only wires them together.

Pre-requisites (one-time, NOT created here):
    * part1/data/original_segment.wav    -- the 10-min lecture audio (you supply)
    * part1/data/student_voice_ref.wav   -- your 60-s voice reference
    * part1/data/cv/{en,hi}/*.{wav,flac} -- Common Voice training data
    * part1/data/cv_feats/{en,hi}/*.npy  -- pre-extracted features
                                            (run augment_and_extract.py once)
    * checkpoints/lid.pt                 -- trained LID weights
                                            (run train_lid.py once)
    * checkpoints/lm.pkl                 -- trigram LM
                                            (run ngram_lm.py build once)

Usage:
    python pipeline.py                   # run everything (~30 min on CPU)
    python pipeline.py --skip_part1      # skip Part I (assumes outputs exist)
    python pipeline.py --quick_transcript  # bypass constrained ASR (debug)

Run individual parts directly to recover from a mid-pipeline failure:
    python part3/pipeline_part3.py --ref part1/data/student_voice_ref.wav \\
                                   --src part1/data/original_segment_22k.wav
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# --- canonical paths (edit only here if you move the project around) -------
ROOT          = Path(__file__).resolve().parent
DATA_DIR      = ROOT / "part1" / "data"
RESULTS_DIR   = ROOT / "results"

SRC_WAV       = DATA_DIR / "original_segment.wav"
SRC_22K_WAV   = DATA_DIR / "original_segment_22k.wav"
STUDENT_REF   = DATA_DIR / "student_voice_ref.wav"
# Default to the head-only checkpoint (11.6 MB, fits GitHub's 100 MB file
# limit). Pass --ckpt with the full lid.pt if you have the 1.2 GB file.
LID_CKPT      = ROOT / "checkpoints" / "lid_head.pt"
if not LID_CKPT.exists():
    LID_CKPT = ROOT / "checkpoints" / "lid.pt"      # fallback for trainers
LM_PKL        = ROOT / "checkpoints" / "lm.pkl"


def run(cmd, env_extra=None):
    """Run a sub-step and abort the whole pipeline if it fails."""
    print(">>", " ".join(str(c) for c in cmd))
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    res = subprocess.run(cmd, env=env)
    if res.returncode != 0:
        sys.exit(f"[pipeline] step failed: {' '.join(map(str, cmd))} "
                 f"(exit {res.returncode})")


def precheck():
    """Bail out early with a clear error if any required file is missing."""
    missing = [p for p in [SRC_WAV, STUDENT_REF, LID_CKPT, LM_PKL] if not p.exists()]
    if missing:
        msg = "[pipeline] missing required files:\n  " + "\n  ".join(map(str, missing))
        msg += ("\n\nFix:\n"
                "  - Place your 10-min lecture WAV at part1/data/original_segment.wav\n"
                "  - Record 60s and save as part1/data/student_voice_ref.wav\n"
                "  - Train the LID:  python part1/train_lid.py --feats_dir part1/data/cv_feats\n"
                "  - Build the LM:   python part1/ngram_lm.py build --corpus part1/syllabus.txt\n")
        sys.exit(msg)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip_prep",  action="store_true", help="Skip Part 0 audio prep.")
    ap.add_argument("--skip_part1", action="store_true")
    ap.add_argument("--skip_part2", action="store_true")
    ap.add_argument("--skip_part3", action="store_true")
    ap.add_argument("--skip_part4", action="store_true")
    ap.add_argument("--skip_eval",  action="store_true",
                    help="Skip the slow Whisper-pseudo-truth WER evaluation.")
    ap.add_argument("--quick_transcript", action="store_true",
                    help="Use vanilla Whisper instead of the Part I constrained chain.")
    args = ap.parse_args()

    precheck()

    # Windows + miniconda often loads two OpenMP runtimes; this env var is
    # the standard workaround. Set it for every sub-process for safety.
    env_extra = {"PYTHONIOENCODING": "utf-8", "KMP_DUPLICATE_LIB_OK": "TRUE"}

    # ---- Part 0: audio prep (mono 16k + 22.05k from the supplied WAV) ----
    if not args.skip_prep:
        run([sys.executable, "part1/prepare_audio.py",
             "--in", str(SRC_WAV)], env_extra=env_extra)

    # ---- Part I: denoise -> LID -> constrained ASR ----
    if not args.skip_part1:
        if args.quick_transcript:
            run([sys.executable, "part2/quick_transcript.py"], env_extra=env_extra)
        else:
            run([sys.executable, "part1/pipeline_part1.py",
                 "--wav",   str(SRC_WAV),
                 "--ckpt",  str(LID_CKPT),
                 "--lm",    str(LM_PKL),
                 "--outdir", str(RESULTS_DIR)], env_extra=env_extra)

    # ---- Part II: G2P/IPA + LRL translation ----
    if not args.skip_part2:
        run([sys.executable, "part2/pipeline_part2.py"], env_extra=env_extra)

    # ---- Part III: voice embed -> MMS-TTS -> kNN-VC -> prosody warp ----
    if not args.skip_part3:
        run([sys.executable, "part3/pipeline_part3.py",
             "--ref", str(STUDENT_REF),
             "--src", str(SRC_22K_WAV)], env_extra=env_extra)

    # ---- Part IV: anti-spoof CM + PGD adversarial on LID ----
    if not args.skip_part4:
        run([sys.executable, "part4/pipeline_part4.py",
             "--bonafide", str(STUDENT_REF)], env_extra=env_extra)

    # ---- Final cross-cutting metrics ----
    if not args.skip_eval:
        run([sys.executable, "eval_metrics.py",
             "--ref", str(STUDENT_REF)], env_extra=env_extra)

    # ---- Summary -------------------------------------------------------
    print("\n=== Full pipeline complete ===\n")
    print("Audio manifest:")
    print(f"  {SRC_WAV}                 -- source 10-min lecture")
    print(f"  {STUDENT_REF}             -- your 60-s reference")
    print(f"  {RESULTS_DIR/'output_LRL_cloned.wav'}  -- final cloned-voice LRL synth")
    print(f"  {RESULTS_DIR/'output_LRL_flat.wav'}    -- VC-only ablation baseline")
    print("\nMetric outputs:")
    print(f"  {RESULTS_DIR/'lid_eval.json'}        -- Task 1.1 macro-F1 (EN vs HI)")
    print(f"  {RESULTS_DIR/'transcript.json'}      -- Task 1.2 constrained ASR")
    print(f"  {RESULTS_DIR/'transcript_ipa.json'}  -- Task 2.1 unified IPA")
    print(f"  {RESULTS_DIR/'transcript_lrl.json'}  -- Task 2.2 Maithili translation")
    print(f"  {RESULTS_DIR/'cm_eer.json'}          -- Task 4.1 EER")
    print(f"  {RESULTS_DIR/'adversarial.json'}     -- Task 4.2 minimum epsilon")
    print(f"  {RESULTS_DIR/'eval_metrics.json'}    -- MCD + WER + switch density")


if __name__ == "__main__":
    main()
