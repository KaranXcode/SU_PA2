"""Quick MCD probe at every Part 3 stage. Run:
    python mcd_stages.py
"""
from pathlib import Path
from eval_metrics import _load_mono, mcd

REF = Path("part1/data/student_voice_ref.wav")
STAGES = [
    "part1/data/student_voice_ref.wav",   # sanity: ref vs ref -> 0.00
    "results/synthesized_lrl_raw.wav",    # MMS default speaker
    "results/synthesized_lrl_vc.wav",     # after kNN-VC
    "results/output_LRL_flat.wav",        # VC, no prosody warp
    "results/output_LRL_cloned.wav",      # VC + prosody warp (final)
]


def main():
    ref = _load_mono(REF)
    print(f"{'file':55s}  MCD   rev")
    for p in STAGES:
        try:
            syn = _load_mono(Path(p))
            r = mcd(ref, syn)
            print(f"{p:55s}  {r['mcd']:5.2f}  {r['mcd_reverse']:5.2f}")
        except Exception as e:
            print(f"{p:55s}  ERROR: {e}")


if __name__ == "__main__":
    main()
