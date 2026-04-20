"""
End-to-end Part IV orchestrator:
    (4.1) anti-spoofing CM training + EER
    (4.2) FGSM adversarial attack on the LID model

Inputs (must already exist):
    data/student_voice_ref.wav         (60 s reference)
    results/output_LRL_cloned.wav      (Part III final synth)
    results/denoised.wav               (Part I.3 output)
    results/lid_segments.json          (Part I.1 output)
    checkpoints/lid.pt                 (Part I.1 weights)

Outputs:
    checkpoints/cm.pt
    results/cm_eer.json
    results/adversarial.json
    results/adversarial_chunk.wav
"""
import argparse
import subprocess


def run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bonafide", default="data/student_voice_ref.wav")
    ap.add_argument("--spoof",    default="results/output_LRL_cloned.wav")
    ap.add_argument("--denoised", default="results/denoised.wav")
    ap.add_argument("--segments", default="results/lid_segments.json")
    ap.add_argument("--lid_ckpt", default="checkpoints/lid.pt")
    args = ap.parse_args()

    run(["python", "part4/antispoof.py",
         "--bonafide", args.bonafide,
         "--spoof",    args.spoof])

    run(["python", "part4/adversarial.py",
         "--wav",      args.denoised,
         "--segments", args.segments,
         "--ckpt",     args.lid_ckpt])

    print("\nDone. Part IV outputs in results/cm_eer.json + results/adversarial.json")


if __name__ == "__main__":
    main()
