"""
End-to-end Part II orchestrator:
    transcript.json -> (2.1) IPA -> (2.2) LRL (Maithili)

Usage:
    python part2/pipeline_part2.py
        [--transcript results/transcript.json]
        [--ipa_out results/transcript_ipa.json]
        [--lrl_out results/transcript_lrl.json]
        [--dict part2/lrl_dict.tsv]

Assumes a transcript JSON of the shape produced by either
[part1/constrained_decode.py] or [part2/quick_transcript.py].
"""
import argparse
from pathlib import Path

from g2p_hinglish import transcript_to_ipa_json
from translate_lrl import transcript_to_lrl_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="results/transcript.json")
    ap.add_argument("--ipa_out", default="results/transcript_ipa.json")
    ap.add_argument("--lrl_out", default="results/transcript_lrl.json")
    ap.add_argument("--dict", default="part2/lrl_dict.tsv")
    args = ap.parse_args()

    transcript_to_ipa_json(Path(args.transcript), Path(args.ipa_out))
    transcript_to_lrl_json(Path(args.ipa_out),    Path(args.lrl_out),
                           Path(args.dict))

    print("\nDone. Part II outputs:")
    print("  IPA  ->", args.ipa_out)
    print("  LRL  ->", args.lrl_out)


if __name__ == "__main__":
    main()
