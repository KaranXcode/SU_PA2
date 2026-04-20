"""
End-to-end Part III orchestrator:
    (3.1) voice embedding  ->  (3.3) MMS-TTS synthesis
    ->   kNN voice conversion (inject student's voice)
    ->   (3.2) prosody warp (transfer professor's teaching style)

Inputs (must already exist):
    data/student_voice_ref.wav     : your 60s recording
    data/original_segment_22k.wav  : the source professor audio (Part 0)
    results/transcript_lrl.json    : Part II output

Outputs:
    results/student_xvector.npy        : Task 3.1 speaker embedding
    results/synthesized_lrl_raw.wav    : MMS-TTS raw output (default speaker)
    results/synthesized_lrl_vc.wav     : after kNN-VC (student's voice)
    results/output_LRL_flat.wav        : ablation: VC + no prosody warp
    results/output_LRL_cloned.wav      : Task 3.2 final (VC + prosody warp)
"""
import argparse
import subprocess
from pathlib import Path


def run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/student_voice_ref.wav")
    ap.add_argument("--src", default="data/original_segment_22k.wav")
    ap.add_argument("--transcript", default="results/transcript_lrl.json")
    ap.add_argument("--xvector", default="results/student_xvector.npy")
    ap.add_argument("--syn_raw", default="results/synthesized_lrl_raw.wav")
    ap.add_argument("--syn_vc", default="results/synthesized_lrl_vc.wav")
    ap.add_argument("--syn_flat", default="results/output_LRL_flat.wav")
    ap.add_argument("--syn_warped", default="results/output_LRL_cloned.wav")
    ap.add_argument("--vc_k", type=int, default=4)
    args = ap.parse_args()

    Path("results").mkdir(parents=True, exist_ok=True)

    # 3.1 speaker embedding
    run(["python", "part3/voice_embed.py",
         "--ref", args.ref, "--out", args.xvector])

    # 3.3 MMS-TTS synthesis (default MMS speaker)
    run(["python", "part3/synthesize_lrl.py",
         "--transcript", args.transcript, "--out", args.syn_raw])

    # kNN Voice Conversion: re-speak the MMS output in the student's voice.
    # This is the step that actually makes the final audio pass MCD < 8.0
    # against student_voice_ref.wav.
    run(["python", "part3/voice_convert.py",
         "--content", args.syn_raw,
         "--ref", args.ref,
         "--out", args.syn_vc,
         "--k", str(args.vc_k)])

    # 3.2 prosody warp. Input is the VC'd audio so the warped output keeps
    # the student's timbre while picking up the professor's F0 / rhythm.
    # --flat emits the VC'd-but-unwarped audio as the ablation baseline
    # (the report's "prosody warping vs. flat synthesis" comparison).
    run(["python", "part3/prosody.py",
         "--src", args.src, "--syn", args.syn_vc,
         "--out", args.syn_warped, "--flat", args.syn_flat])

    print("\nDone. Part III outputs:")
    print("  x-vector ->", args.xvector)
    print("  raw TTS  ->", args.syn_raw,  "  (MMS default speaker)")
    print("  VC'd     ->", args.syn_vc,   "  (student's voice, no prosody)")
    print("  flat     ->", args.syn_flat, "  (ablation: VC only)")
    print("  cloned   ->", args.syn_warped, "(VC + prosody = final)")


if __name__ == "__main__":
    main()
