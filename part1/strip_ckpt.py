"""
Strip the frozen Wav2Vec2 backbone from a FrameLID checkpoint.

Why: saving the full `model.state_dict()` in train_lid.py includes the
300M-param Wav2Vec2-XLS-R backbone that is FROZEN during training.
Those weights are identical to what `Wav2Vec2Model.from_pretrained(
"facebook/wav2vec2-xls-r-300m")` already serves from HuggingFace, so
they add 1.2 GB of dead weight to every checkpoint on disk. GitHub's
100 MB hard per-file limit rejects such files.

This script loads a full checkpoint and saves a "head-only" variant
containing only the learned parameters: proj, feat_dropout, lstm,
head_main, head_switch. Typical size: 5-15 MB (pushable).

`infer_lid.py` and `adversarial.py` already rebuild the backbone from
scratch at model init; they then call `model.load_state_dict(..., strict=False)`
so both full and head-only checkpoints load correctly.

Usage:
    python part1/strip_ckpt.py \
        --in  checkpoints/lid.pt \
        --out checkpoints/lid_head.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


HEAD_PREFIXES = ("proj.", "feat_dropout.", "lstm.", "head_main.", "head_switch.")


def strip(in_path: Path, out_path: Path) -> None:
    ckpt = torch.load(str(in_path), map_location="cpu")
    # Training code saves  {"model": state_dict, "val_f1": ..., "epoch": ...}
    sd_full = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    sd_head = {k: v for k, v in sd_full.items() if k.startswith(HEAD_PREFIXES)}

    kept = len(sd_head)
    dropped = len(sd_full) - kept
    size_before = sum(v.numel() * v.element_size() for v in sd_full.values()) / 1e6
    size_after = sum(v.numel() * v.element_size() for v in sd_head.values()) / 1e6

    out_obj = {"model": sd_head}
    # Preserve training-time metadata if present.
    if isinstance(ckpt, dict):
        for k in ("val_f1", "epoch", "args"):
            if k in ckpt:
                out_obj[k] = ckpt[k]
    out_obj["backbone_name"] = "facebook/wav2vec2-xls-r-300m"
    out_obj["head_only"] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, str(out_path))

    print(f"[strip] kept {kept} head tensors, dropped {dropped} backbone tensors")
    print(f"[strip] size: {size_before:.1f} MB -> {size_after:.1f} MB "
          f"({100*size_after/size_before:.1f}%)")
    print(f"[strip] wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  default="checkpoints/lid.pt",
                    help="Full checkpoint (with backbone).")
    ap.add_argument("--out", dest="outp", default="checkpoints/lid_head.pt",
                    help="Output head-only checkpoint.")
    args = ap.parse_args()
    strip(Path(args.inp), Path(args.outp))
