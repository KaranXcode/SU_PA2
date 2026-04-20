"""
Task 1.1 -- Multi-Head Frame-Level Language Identification (English vs Hindi).

Architecture:
    Wav2Vec2-XLS-R-300m (frozen)  ->  [B, T, 1024] @ 20 ms / frame
        -> Projection (1024 -> 256)
        -> 2-layer BiLSTM (256, bidir -> 512)
        -> Multi-head (2 parallel linear heads):
              head_main:   {EN, HI, SIL}    (3-way, softmax)
              head_switch: P(switch-boundary within 40 ms)  -- auxiliary
    The multi-head design lets the switch-boundary head sharpen the
    timestamp of a language change to <= 200 ms (assignment requirement).

Output:
    per-frame probabilities at 50 fps (20 ms stride). We threshold + median
    filter to produce (start, end, lang) segments for the pipeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

FRAME_HZ = 50          # Wav2Vec2 stride = 20 ms
LABELS = ["en", "hi", "sil"]


class ChannelDropout(nn.Module):
    """Drop entire feature channels (same mask across batch and time).
    Forces the model to use distributed cues instead of memorising a few
    speaker-correlated dimensions."""
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        keep = 1.0 - self.p
        mask = torch.bernoulli(
            torch.full((x.size(-1),), keep, device=x.device)
        ) / keep
        return x * mask


class FrameLID(nn.Module):
    def __init__(
        self,
        backbone_name: str = "facebook/wav2vec2-xls-r-300m",
        proj_dim: int = 256,
        num_classes: int = 3,
        freeze_backbone: bool = True,
        lstm_dropout: float = 0.3,
        channel_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = Wav2Vec2Model.from_pretrained(backbone_name)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, proj_dim)
        self.feat_dropout = ChannelDropout(channel_dropout)
        self.lstm = nn.LSTM(
            proj_dim, proj_dim, num_layers=2, bidirectional=True,
            batch_first=True, dropout=lstm_dropout,
        )
        self.head_main = nn.Linear(proj_dim * 2, num_classes)
        self.head_switch = nn.Linear(proj_dim * 2, 1)  # sigmoid -> switch prob

    def extract(self, wav: torch.Tensor) -> torch.Tensor:
        # wav: (B, T) at 16 kHz
        # During FGSM the input itself carries requires_grad=True; we MUST
        # let gradients flow back through the backbone to the input even
        # though the backbone params are frozen and we're in eval() mode.
        needs_grad = wav.requires_grad
        if self.backbone.training is False and not needs_grad:
            with torch.no_grad():
                out = self.backbone(wav).last_hidden_state      # (B, Tf, H)
        else:
            out = self.backbone(wav).last_hidden_state
        return out

    def forward(self, wav: torch.Tensor):
        feats = self.extract(wav)
        feats = self.proj(feats)
        feats = self.feat_dropout(feats)
        feats, _ = self.lstm(feats)
        logits = self.head_main(feats)                          # (B, Tf, C)
        switch = self.head_switch(feats).squeeze(-1)            # (B, Tf)
        return logits, switch


# --------------------------------------------------------------------------- #
# Post-processing: logits -> [(start_s, end_s, lang), ...]
# --------------------------------------------------------------------------- #
def decode_segments(
    logits: torch.Tensor,        # (Tf, C)
    switch_logits: torch.Tensor, # (Tf,)
    min_seg_frames: int = 25,    # 500 ms floor (old 200 ms jittered, 1 s ate Hindi)
    switch_boost: float = 0.3,   # was 1.5 -> aggressively over-split boundaries
    median_kernel: int = 11,     # 220 ms smoothing
    merge_gap_frames: int = 25,  # 500 ms gap for same-lang re-merge
):
    """Median-filter + collapse to segments. Uses the auxiliary switch head to
    snap boundaries (bias the per-frame argmax near high-switch-prob frames).
    A second pass merges adjacent same-language segments so that short silences
    or misclassified blips between them don't fragment the transcript.
    """
    import numpy as np
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    sw = torch.sigmoid(switch_logits).cpu().numpy()

    probs_sharp = probs.copy()
    probs_sharp = probs_sharp ** (1.0 + switch_boost * sw[:, None])
    probs_sharp /= probs_sharp.sum(axis=-1, keepdims=True)
    preds = probs_sharp.argmax(axis=-1)

    from scipy.signal import medfilt
    preds = medfilt(preds, kernel_size=median_kernel).astype(int)

    # Pass 1: collapse consecutive equal-prediction frames into raw spans.
    raw = []
    i = 0
    while i < len(preds):
        j = i
        while j < len(preds) and preds[j] == preds[i]:
            j += 1
        raw.append([i, j, int(preds[i])])
        i = j

    # Pass 2: drop spans < min_seg_frames BUT absorb them into neighbours so
    # their audio time isn't silently dropped from downstream ASR.
    cleaned = []
    for start, end, lab in raw:
        if (end - start) < min_seg_frames and cleaned:
            # Absorb this short blip into the previous span.
            cleaned[-1][1] = end
        else:
            cleaned.append([start, end, lab])

    # Pass 3: concatenate consecutive same-language spans (the absorb step
    # above can leave chains like [en][en] after it ate a short blip).
    merged: list[list[int]] = []
    for span in cleaned:
        if merged and span[2] == merged[-1][2]:
            merged[-1][1] = span[1]
        else:
            merged.append(span)

    # Pass 4: flatten "ABA" patterns where B is short and the two A's share a
    # language -- the classic "en, hi, en" stutter from noisy frames.
    changed = True
    while changed:
        changed = False
        i = 0
        while i + 2 < len(merged):
            a, b, c = merged[i], merged[i + 1], merged[i + 2]
            if (a[2] == c[2] and b[2] != a[2]
                    and (b[1] - b[0]) < merge_gap_frames):
                merged = merged[:i] + [[a[0], c[1], a[2]]] + merged[i + 3:]
                changed = True
            else:
                i += 1

    # Final: convert back to (start_s, end_s, lang).
    return [(s / FRAME_HZ, e / FRAME_HZ, LABELS[l]) for s, e, l in merged]


if __name__ == "__main__":
    m = FrameLID()
    wav = torch.randn(1, 16000 * 3)
    logits, sw = m(wav)
    print("logits:", logits.shape, "switch:", sw.shape)
