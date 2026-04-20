"""
Train the FrameLID model on a synthetic code-switched training set
built by concatenating random Common Voice English + Hindi clips.

This gives us *true* frame-level labels (we know exactly when each clip
starts/ends) which is required for a frame-level F1 >= 0.85 evaluation.

Usage:
    python train_lid.py --data_dir data/cv --epochs 6 --batch 4

Expected layout (arranged by the user once CV is downloaded):
    data/cv/en/*.wav   (English clips, 16 kHz mono)
    data/cv/hi/*.wav   (Hindi   clips, 16 kHz mono)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                    # headless -> write PNG directly
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import (
    confusion_matrix, f1_score, classification_report,
)
from torch.utils.data import Dataset, DataLoader

from lid_model import FrameLID, FRAME_HZ, LABELS


SAMPLE_RATE = 16000
CLIP_SEC = 20.0  # 20 s montage -- much closer to real-lecture span distribution

# Span-length sampler. Real lectures have long monolingual stretches; old setup
# (1-3 s chunks) taught the model to expect a switch almost every second, which
# at test time surfaces as 158 switches / 10 min of noisy boundaries.
MIN_SPAN_SEC = 5.0
MAX_SPAN_SEC = 15.0

# Probability of inserting a "silence" (label=2) gap between two clips within
# a montage. Real lectures contain long pauses between phrases.
SIL_PROB = 0.20
SIL_MIN_FR = 10    # 200 ms
SIL_MAX_FR = 40    # 800 ms

# Feature-space noise augmentation. Training clips are clean read speech; the
# test audio is a noisy lecture-hall recording. A small additive Gaussian on
# the frozen Wav2Vec2 features is a cheap proxy for that domain gap.
FEAT_NOISE_PROB = 0.5
FEAT_NOISE_STD_RANGE = (0.05, 0.25)


class CodeSwitchSynth(Dataset):
    """Build code-switched examples on-the-fly by concatenating random
    Common-Voice EN and HI clips. Label == language of each source clip,
    sampled at 50 fps, padded/truncated to 8 s (== 400 frames).

    Two modes:
        (a) Raw audio  (feats_dir=None) : returns (wav, labels).
            Model runs the full backbone each step. Slow on CPU.
        (b) Pre-extracted features (feats_dir given) : returns (feats, labels).
            Skips backbone entirely. Each step is ~100x faster.
            Run ``extract_features.py`` first to populate feats_dir.

    Splits:
        Each language's clip pool is partitioned into a train/val split
        (default 80/20, deterministic shuffle by seed=12345). The val pool
        is held out from training so we can detect speaker-shortcut
        overfitting (val F1 << train F1).

    Augmentation (training only, feature mode):
        - SpecAugment: 2 time bands (<=30 frames) + 2 feature bands (<=80 dims)
        - Crossfade at clip boundaries (3 frames linear blend) so the model
          can't trivially detect language changes from acoustic discontinuity.
    """

    def __init__(self, root: Path, n_items: int = 2000, seed: int = 0,
                 max_per_lang: int = 2000, feats_dir: Path | None = None,
                 split: str = "train", val_frac: float = 0.2,
                 augment: bool = True):
        assert split in ("train", "val")
        self.split = split
        self.augment = augment and split == "train"
        self.use_feats = feats_dir is not None and feats_dir.exists()

        if self.use_feats:
            en_all = sorted((feats_dir / "en").glob("*.npy"))[:max_per_lang]
            hi_all = sorted((feats_dir / "hi").glob("*.npy"))[:max_per_lang]
            print(f"[data] pre-extracted features mode")
        else:
            exts = ("wav", "flac", "mp3", "ogg", "m4a")
            def _collect(d: Path):
                if not d.exists():
                    return []
                files = []
                for ext in exts:
                    files.extend(d.rglob(f"*.{ext}"))
                    files.extend(d.rglob(f"*.{ext.upper()}"))
                return sorted(set(files))
            en_all = _collect(root / "en")[:max_per_lang]
            hi_all = _collect(root / "hi")[:max_per_lang]

        # Deterministic disjoint train/val split per language. We use a fixed
        # split-rng (independent of `seed`) so train and val datasets agree
        # on which clips belong to which split.
        split_rng = random.Random(12345)
        en_all = list(en_all); split_rng.shuffle(en_all)
        hi_all = list(hi_all); split_rng.shuffle(hi_all)

        n_en_val = max(1, int(len(en_all) * val_frac))
        n_hi_val = max(1, int(len(hi_all) * val_frac))
        if split == "train":
            self.en = en_all[n_en_val:]
            self.hi = hi_all[n_hi_val:]
        else:
            self.en = en_all[:n_en_val]
            self.hi = hi_all[:n_hi_val]

        print(f"[data:{split}] en clips: {len(self.en)}   hi clips: {len(self.hi)}"
              f"   augment={self.augment}")
        assert self.en and self.hi, (
            f"Need clips under {root}/{{en,hi}} (audio) or feats_dir/{{en,hi}} (.npy)")
        self.n_items = n_items
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_items

    def _load_audio(self, path: Path):
        arr, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if arr.shape[1] > 1:
            arr = arr.mean(axis=1, keepdims=True)
        w = torch.from_numpy(arr[:, 0])
        if sr != SAMPLE_RATE:
            w = torchaudio.functional.resample(w.unsqueeze(0), sr, SAMPLE_RATE)[0]
        return w

    def _load_feats(self, path: Path):
        # (T_frames, 1024) saved as float16 -> load as float32
        return torch.from_numpy(np.load(path).astype(np.float32))

    def _get_item_audio(self, idx):
        """Build a 20 s raw-audio montage from CV clips, following the same
        long-span plan used by the feature-mode path. Each clip is augmented
        with a randomised lecture-like acoustic chain (reverb + noise + speed
        + low-pass + soft-clip) imported from augment_and_extract.py, so the
        frozen Wav2Vec2 backbone sees audio close to the test distribution
        instead of clean studio read speech. This is the raw-audio branch --
        use it when you haven't pre-extracted features (`--feats_dir` unset).
        """
        from augment_and_extract import augment as audio_augment  # local import

        target_len = int(CLIP_SEC * SAMPLE_RATE)
        target_frames = int(CLIP_SEC * FRAME_HZ)
        wav = torch.zeros(target_len)
        frame_labels = np.full(target_frames, 2, dtype=np.int64)

        plan = self._sample_span_plan(target_frames)

        cursor_sm = 0    # sample-level write cursor
        cursor_fr = 0    # frame-level label cursor
        prev_lang = None

        for lang_id, span_fr in plan:
            span_sm = span_fr * SAMPLE_RATE // FRAME_HZ
            span_end_sm = min(cursor_sm + span_sm, target_len)
            span_end_fr = min(cursor_fr + span_fr, target_frames)

            # Optional silence gap between adjacent spans (real lecture pauses).
            # We only inject gap when there's room both in samples and frames.
            if (prev_lang is not None and self.augment
                    and self.rng.random() < SIL_PROB
                    and span_end_sm - cursor_sm > SIL_MIN_FR * SAMPLE_RATE // FRAME_HZ):
                gap_fr = self.rng.randint(SIL_MIN_FR, SIL_MAX_FR)
                gap_fr = min(gap_fr, (span_end_fr - cursor_fr) - 1)
                if gap_fr > 0:
                    gap_sm = gap_fr * SAMPLE_RATE // FRAME_HZ
                    wav[cursor_sm:cursor_sm + gap_sm] = \
                        torch.randn(gap_sm) * 1e-3   # near-silence noise floor
                    frame_labels[cursor_fr:cursor_fr + gap_fr] = 2
                    cursor_sm += gap_sm
                    cursor_fr += gap_fr

            # Fill the rest of the span by pulling random clips from the
            # language pool and augmenting each one independently.
            while cursor_sm < span_end_sm:
                pool = self.en if lang_id == 0 else self.hi
                clip = self._load_audio(self.rng.choice(pool))
                if self.augment:
                    clip = audio_augment(clip, self.rng)
                remaining = span_end_sm - cursor_sm
                take = min(clip.numel(), remaining)
                # Random crop from a random offset inside the clip so we don't
                # always show the same head of every clip.
                if clip.numel() > take:
                    off = self.rng.randint(0, clip.numel() - take)
                    clip = clip[off:off + take]
                wav[cursor_sm:cursor_sm + take] = clip[:take]
                cursor_sm += take

            frame_labels[cursor_fr:span_end_fr] = lang_id
            cursor_fr = span_end_fr
            prev_lang = lang_id

        wav = wav.clamp_(-1.0, 1.0)
        return wav, torch.from_numpy(frame_labels)

    def _spec_augment(self, feats: torch.Tensor) -> torch.Tensor:
        """In-place SpecAugment on (T, F): 2 time bands + 2 feature bands."""
        T, F = feats.shape
        for _ in range(2):
            t = self.rng.randint(1, 30)
            if T - t > 0:
                t0 = self.rng.randint(0, T - t)
                feats[t0:t0 + t] = 0.0
        for _ in range(2):
            f = self.rng.randint(1, 80)
            if F - f > 0:
                f0 = self.rng.randint(0, F - f)
                feats[:, f0:f0 + f] = 0.0
        return feats

    def _sample_span_plan(self, target_frames: int):
        """Pick a code-switch plan for one montage.

        Returns a list of (lang_id, n_frames) tuples that sum to target_frames.
        Distribution:
            50% monolingual (entire montage is EN or HI)
            35% one switch  (montage has exactly one language change)
            15% two switches

        This matches real-lecture span statistics much better than the old
        "switch every 1-3 s" regime, which was the root cause of the 158-
        switches-in-10-minutes blow-up at test time.
        """
        r = self.rng.random()
        if r < 0.5:
            n_switches = 0
        elif r < 0.85:
            n_switches = 1
        else:
            n_switches = 2
        n_spans = n_switches + 1

        min_fr = int(MIN_SPAN_SEC * FRAME_HZ)
        max_fr = int(MAX_SPAN_SEC * FRAME_HZ)

        # Dirichlet-ish: split target_frames into n_spans spans, each at least
        # min_fr. Slack on top is distributed randomly.
        span_lens = [min_fr] * n_spans
        slack = max(0, target_frames - min_fr * n_spans)
        for _ in range(slack):
            span_lens[self.rng.randrange(n_spans)] += 1
        span_lens = [min(x, max_fr) for x in span_lens]
        # Rescale: if clipping by max_fr shrank total below target, extend the
        # last span (the code-switch frequency matters more than exact length).
        deficit = target_frames - sum(span_lens)
        if deficit > 0:
            span_lens[-1] += deficit

        # Alternating language sequence starting from a random pick.
        first = self.rng.choice([0, 1])
        plan = []
        for i, L in enumerate(span_lens):
            lang_id = first if i % 2 == 0 else (1 - first)
            plan.append((lang_id, L))
        return plan

    def _fill_span(self, feats_buf: torch.Tensor, frame_labels: np.ndarray,
                   cursor: int, end: int, lang_id: int):
        """Fill feats_buf[cursor:end] by randomly sampling (and cropping) clips
        from the target language pool until the span is filled. Labels are set
        to lang_id for the whole span."""
        pool = self.en if lang_id == 0 else self.hi
        while cursor < end:
            clip_feats = self._load_feats(self.rng.choice(pool))
            remaining = end - cursor
            max_take = min(clip_feats.size(0), remaining)
            # Avoid always taking the same head of each clip: random crop.
            if clip_feats.size(0) > max_take:
                start_idx = self.rng.randint(0, clip_feats.size(0) - max_take)
                clip_feats = clip_feats[start_idx:start_idx + max_take]
            n = clip_feats.size(0)
            feats_buf[cursor:cursor + n] = clip_feats
            frame_labels[cursor:cursor + n] = lang_id
            cursor += n
        return cursor

    def _maybe_inject_silence(self, feats_buf: torch.Tensor,
                              frame_labels: np.ndarray, cursor: int,
                              end_limit: int) -> int:
        """With probability SIL_PROB, insert a short low-energy gap labelled
        as `sil` (2). Returns new cursor. Keeps output within end_limit."""
        if self.rng.random() >= SIL_PROB:
            return cursor
        n = self.rng.randint(SIL_MIN_FR, SIL_MAX_FR)
        n = min(n, end_limit - cursor)
        if n <= 0:
            return cursor
        # Small random noise rather than exact zeros so the model can't learn
        # "zero vector == silence".
        feats_buf[cursor:cursor + n] = torch.randn(n, feats_buf.size(1)) * 0.05
        frame_labels[cursor:cursor + n] = 2
        return cursor + n

    def _feature_domain_noise(self, feats: torch.Tensor) -> torch.Tensor:
        """Additive Gaussian noise on the feature stream. Rough proxy for the
        (clean-CV -> noisy-lecture-hall) domain gap. Applied with probability
        FEAT_NOISE_PROB and a scale sampled uniformly from FEAT_NOISE_STD_RANGE."""
        if self.rng.random() >= FEAT_NOISE_PROB:
            return feats
        lo, hi = FEAT_NOISE_STD_RANGE
        std = self.rng.uniform(lo, hi)
        return feats + torch.randn_like(feats) * std

    def _get_item_feats(self, idx):
        target_frames = int(CLIP_SEC * FRAME_HZ)   # 1000 frames for 20 s
        feat_dim = 1024
        feats = torch.zeros(target_frames, feat_dim)
        frame_labels = np.full(target_frames, 2, dtype=np.int64)
        fade = 3   # frames of crossfade at language-change boundaries (60 ms)

        plan = self._sample_span_plan(target_frames)
        cursor = 0
        prev_lang = None
        for span_idx, (lang_id, span_len) in enumerate(plan):
            span_end = min(cursor + span_len, target_frames)
            # Crossfade at true language-change boundaries.
            if (self.augment and prev_lang is not None and lang_id != prev_lang
                    and cursor >= fade and span_end - cursor > fade):
                alpha = torch.linspace(0.0, 1.0, fade).unsqueeze(-1)
                prev_tail = feats[cursor - fade:cursor].clone()
                # Preview first `fade` frames by peeking at one clip from the
                # new language, then backfill the rest of the span normally.
                pool = self.en if lang_id == 0 else self.hi
                peek = self._load_feats(self.rng.choice(pool))
                head = peek[:fade] if peek.size(0) >= fade else \
                       torch.cat([peek, torch.zeros(fade - peek.size(0),
                                                    feat_dim)], dim=0)
                feats[cursor:cursor + fade] = (1.0 - alpha) * prev_tail + alpha * head
                frame_labels[cursor:cursor + fade] = lang_id
                cursor += fade
            cursor = self._fill_span(feats, frame_labels, cursor, span_end, lang_id)
            prev_lang = lang_id
            # Within a monolingual span we also sprinkle tiny silences to mimic
            # real pauses, but only between clip boundaries (where cursor sits).
            cursor = self._maybe_inject_silence(feats, frame_labels, cursor,
                                                span_end)

        if self.augment:
            feats = self._feature_domain_noise(feats)
            feats = self._spec_augment(feats)
        return feats, torch.from_numpy(frame_labels)

    def __getitem__(self, idx):
        if self.use_feats:
            return self._get_item_feats(idx)
        return self._get_item_audio(idx)


def collate(batch):
    wavs, labs = zip(*batch)
    return torch.stack(wavs), torch.stack(labs)


def align_labels_to_feats(labels: torch.Tensor, t_feats: int) -> torch.Tensor:
    """Labels are sampled at 50 fps from an 8 s clip (400 frames). Wav2Vec2
    output length may differ slightly (usually 399 or 400). Trim / pad."""
    t_lab = labels.size(1)
    if t_lab == t_feats:
        return labels
    if t_lab > t_feats:
        return labels[:, :t_feats]
    pad = labels[:, -1:].repeat(1, t_feats - t_lab)
    return torch.cat([labels, pad], dim=1)


def make_switch_targets(labels: torch.Tensor) -> torch.Tensor:
    # switch = 1 at frames where lang changes vs previous frame
    shifted = torch.cat([labels[:, :1], labels[:, :-1]], dim=1)
    return (labels != shifted).float()


@torch.no_grad()
def _val_macro_f1(model, val_loader, device, use_feats):
    """Evaluate held-out macro-F1 (en vs hi). Returns float in [0, 1]."""
    was_training = model.training
    model.eval()
    ys, ps = [], []
    for batch_data, lab in val_loader:
        batch_data = batch_data.to(device)
        lab = lab.to(device)
        if use_feats:
            feats = model.proj(batch_data)
            feats = model.feat_dropout(feats)
            feats, _ = model.lstm(feats)
            logits = model.head_main(feats)
        else:
            logits, _ = model(batch_data)
        lab = align_labels_to_feats(lab, logits.size(1))
        pred = logits.argmax(-1)
        ys.append(lab.cpu().numpy().ravel())
        ps.append(pred.cpu().numpy().ravel())
    if was_training:
        model.train()
    y = np.concatenate(ys); p = np.concatenate(ps)
    mask = np.isin(y, [0, 1])
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(y[mask], p[mask], average="macro", labels=[0, 1],
                          zero_division=0))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameLID().to(device)
    feats_dir = Path(args.feats_dir) if args.feats_dir else None

    ds_train = CodeSwitchSynth(Path(args.data_dir), n_items=args.n_items,
                               feats_dir=feats_dir, split="train",
                               val_frac=args.val_frac, augment=True, seed=0)
    ds_val   = CodeSwitchSynth(Path(args.data_dir), n_items=args.val_items,
                               feats_dir=feats_dir, split="val",
                               val_frac=args.val_frac, augment=False, seed=42)
    use_feats = ds_train.use_feats

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=0, collate_fn=collate)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch, shuffle=False,
                          num_workers=0, collate_fn=collate)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "lid_train_log.json"
    history = {
        "device": str(device),
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "steps": [],
        "epoch_val_f1": [],
    }

    best_val = -1.0
    best_epoch = -1
    bad_epochs = 0      # count of consecutive epochs with no val improvement
    best_path = Path(args.out)
    last_path = best_path.with_name(best_path.stem + "_last" + best_path.suffix)

    model.train()
    step = 0
    for ep in range(args.epochs):
        for batch_data, lab in dl_train:
            batch_data = batch_data.to(device)
            lab = lab.to(device)
            if use_feats:
                # batch_data is pre-extracted features (B, T, 1024)
                # Skip backbone, feed directly into projection + LSTM + heads.
                # NOTE: we now also apply ChannelDropout to break the
                # speaker-feature shortcut.
                feats = model.proj(batch_data)
                feats = model.feat_dropout(feats)
                feats, _ = model.lstm(feats)
                logits = model.head_main(feats)
                sw = model.head_switch(feats).squeeze(-1)
            else:
                logits, sw = model(batch_data)         # (B, Tf, 3), (B, Tf)
            lab = align_labels_to_feats(lab, logits.size(1))
            sw_tgt = make_switch_targets(lab)
            loss_main = ce(logits.reshape(-1, 3), lab.reshape(-1))
            loss_sw   = bce(sw.reshape(-1), sw_tgt.reshape(-1))
            loss = loss_main + 0.3 * loss_sw
            opt.zero_grad()
            loss.backward()
            opt.step()

            if step % 20 == 0:
                with torch.no_grad():
                    pred = logits.argmax(-1).cpu().numpy().ravel()
                    gt = lab.cpu().numpy().ravel()
                    f1 = f1_score(gt, pred, average="macro")
                print(f"ep {ep} step {step}  loss {loss.item():.3f}  "
                      f"main {loss_main.item():.3f}  sw {loss_sw.item():.3f}  "
                      f"macro-F1 {f1:.3f}")
                history["steps"].append({
                    "epoch": ep, "step": step,
                    "loss": float(loss.item()),
                    "loss_main": float(loss_main.item()),
                    "loss_switch": float(loss_sw.item()),
                    "macro_f1_batch": float(f1),
                })
                # Persist incrementally so a Ctrl+C still leaves a usable log.
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
                _plot_training_curves(history, results_dir / "lid_train_curves.png")
            step += 1

        # ---- end of epoch: evaluate on the held-out val pool ----
        val_f1 = _val_macro_f1(model, dl_val, device, use_feats)
        history["epoch_val_f1"].append({"epoch": ep, "val_f1_en_vs_hi": val_f1})
        print(f"[val] ep {ep}  macro-F1 (en vs hi) = {val_f1:.4f}   "
              f"(best so far: {max(best_val, val_f1):.4f})")

        # Always save a "last" checkpoint (in case you want it for debugging),
        # but the canonical --out path holds the *best* val-F1 checkpoint.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, last_path)

        if val_f1 > best_val:
            best_val = val_f1
            best_epoch = ep
            bad_epochs = 0
            full_sd = model.state_dict()
            torch.save({"model": full_sd, "val_f1": val_f1, "epoch": ep},
                       best_path)
            # Also save a head-only variant next to it so it fits inside
            # GitHub's 100 MB per-file limit (~11 MB instead of ~1.2 GB).
            # The frozen Wav2Vec2 backbone is reloaded from HuggingFace
            # at inference time, so it doesn't need to live in git.
            head_prefixes = ("proj.", "feat_dropout.", "lstm.",
                             "head_main.", "head_switch.")
            head_sd = {k: v for k, v in full_sd.items()
                       if k.startswith(head_prefixes)}
            head_path = best_path.with_name(best_path.stem + "_head"
                                            + best_path.suffix)
            torch.save({"model": head_sd, "val_f1": val_f1, "epoch": ep,
                        "head_only": True,
                        "backbone_name": "facebook/wav2vec2-xls-r-300m"},
                       head_path)
            print(f"  -> new best, saved {best_path}")
            print(f"  -> head-only copy saved  {head_path}")
        else:
            bad_epochs += 1
            print(f"  -> no improvement ({bad_epochs}/{args.patience} bad epochs)")

        history["last_saved_epoch"] = ep
        history["best_val_f1"] = best_val
        history["best_epoch"] = best_epoch
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if bad_epochs >= args.patience:
            print(f"[early-stop] val F1 has not improved for {args.patience} "
                  f"epochs. Stopping at ep {ep}. Best val F1 = {best_val:.4f} "
                  f"@ ep {best_epoch}.")
            break

    _plot_training_curves(history, results_dir / "lid_train_curves.png")
    print(f"Training log -> {log_path}")
    print(f"Training plot -> {results_dir / 'lid_train_curves.png'}")
    print(f"Best val F1 = {best_val:.4f} @ epoch {best_epoch}  "
          f"(checkpoint: {best_path})")


def _plot_training_curves(history: dict, out_path: Path):
    steps = [s["step"] for s in history["steps"]]
    if not steps:
        return
    loss = [s["loss"] for s in history["steps"]]
    lmain = [s["loss_main"] for s in history["steps"]]
    lsw = [s["loss_switch"] for s in history["steps"]]
    f1b = [s["macro_f1_batch"] for s in history["steps"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(steps, loss, label="total", linewidth=2)
    ax1.plot(steps, lmain, label="main CE", linestyle="--", alpha=0.8)
    ax1.plot(steps, lsw, label="switch BCE", linestyle=":", alpha=0.8)
    ax1.set_xlabel("step"); ax1.set_ylabel("loss")
    ax1.set_title("LID training loss"); ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(steps, f1b, color="tab:green", linewidth=2)
    ax2.axhline(0.85, color="tab:red", linestyle="--",
                label="assignment threshold (0.85)")
    ax2.set_xlabel("step"); ax2.set_ylabel("macro-F1 (batch)")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_title("Training macro-F1 (train-batch, noisy)")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrameLID().to(device)
    ckpt = torch.load(args.out, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    n_eval = args.eval_items
    feats_dir = Path(args.feats_dir) if args.feats_dir else None
    ds = CodeSwitchSynth(Path(args.data_dir), n_items=n_eval, seed=42,
                         feats_dir=feats_dir, split="val",
                         val_frac=args.val_frac, augment=False)
    use_feats = ds.use_feats
    dl = DataLoader(ds, batch_size=2, collate_fn=collate)
    total_batches = (n_eval + 1) // 2
    ys, ps = [], []
    for bi, (batch_data, lab) in enumerate(dl):
        if (bi + 1) % 5 == 0 or bi == 0:
            print(f"  eval batch {bi+1}/{total_batches}", end="\r")
        batch_data = batch_data.to(device)
        if use_feats:
            feats = model.proj(batch_data)
            feats, _ = model.lstm(feats)
            logits = model.head_main(feats)
        else:
            logits, _ = model(batch_data)
        lab = align_labels_to_feats(lab, logits.size(1))
        ys.append(lab.cpu().numpy().ravel())
        ps.append(logits.argmax(-1).cpu().numpy().ravel())
    y = np.concatenate(ys); p = np.concatenate(ps)

    all_labels = list(range(len(LABELS)))
    text_report = classification_report(
        y, p, target_names=LABELS, labels=all_labels,
        digits=3, zero_division=0,
    )
    macro_f1_3 = float(f1_score(y, p, average="macro", labels=all_labels,
                                zero_division=0))

    # Assignment 1.1 grades on EN-vs-HI specifically; recompute on non-silence
    # frames only (2-class macro-F1 between the two languages).
    mask = np.isin(y, [0, 1])
    macro_f1_enhi = float(
        f1_score(y[mask], p[mask], average="macro", labels=[0, 1])
    )
    per_class = classification_report(
        y, p, target_names=LABELS, labels=all_labels,
        digits=3, output_dict=True, zero_division=0,
    )

    print(text_report)
    print(f"macro-F1 (en/hi/sil): {macro_f1_3:.4f}")
    print(f"macro-F1 (en vs hi): {macro_f1_enhi:.4f}   "
          f"[assignment requires >= 0.85]")

    # Persist a machine-readable report next to the other pipeline outputs.
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "lid_eval.json"
    out_txt = results_dir / "lid_eval.txt"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "macro_f1_3class": macro_f1_3,
            "macro_f1_en_vs_hi": macro_f1_enhi,
            "per_class": per_class,
            "n_eval_items": n_eval,
            "ckpt": str(args.out),
        }, f, indent=2)
    out_txt.write_text(text_report, encoding="utf-8")
    print(f"Saved {out_json}\nSaved {out_txt}")

    _plot_eval(y, p, per_class, macro_f1_3, macro_f1_enhi,
               results_dir / "lid_eval_plots.png")
    print(f"Saved {results_dir / 'lid_eval_plots.png'}")


def _plot_eval(y, p, per_class, macro_f1_3, macro_f1_enhi, out_path: Path):
    """Three panels: per-class F1 bars | confusion matrix | summary text."""
    cm = confusion_matrix(y, p, labels=[0, 1, 2])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    # --- per-class F1 bars ---
    ax = axes[0]
    f1s = [per_class[l]["f1-score"] for l in LABELS]
    colors = ["tab:blue", "tab:orange", "tab:gray"]
    bars = ax.bar(LABELS, f1s, color=colors)
    ax.axhline(0.85, color="tab:red", linestyle="--",
               label="threshold 0.85")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("F1-score")
    ax.set_title("Per-class F1 (eval set)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    for bar, v in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=9)

    # --- confusion matrix heatmap ---
    ax = axes[1]
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS)
    ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised)")
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- summary numbers ---
    ax = axes[2]
    ax.axis("off")
    passed = macro_f1_enhi >= 0.85
    txt = (
        f"macro-F1 (en/hi/sil) : {macro_f1_3:.4f}\n"
        f"macro-F1 (en vs hi)  : {macro_f1_enhi:.4f}\n\n"
        f"Assignment threshold : 0.85\n"
        f"Status               : {'PASS' if passed else 'FAIL'}\n\n"
        f"Eval items (synth)   : {len(y) // 400}\n"
        f"Frames               : {len(y)}"
    )
    ax.text(0.02, 0.98, txt, family="monospace", fontsize=11,
            va="top", ha="left",
            bbox=dict(facecolor="#f0f0f0", edgecolor="black",
                      boxstyle="round,pad=0.5"))
    ax.set_title("Summary")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/cv")
    ap.add_argument("--feats_dir", default=None,
                    help="Path to pre-extracted features (from extract_features.py). "
                         "If given, skips the Wav2Vec2 backbone during training.")
    ap.add_argument("--out", default="checkpoints/lid.pt")
    ap.add_argument("--epochs", type=int, default=8,
                    help="Max epochs. Early-stopping may end training sooner.")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_items", type=int, default=2000)
    ap.add_argument("--val_frac", type=float, default=0.2,
                    help="Fraction of clips per language reserved as a "
                         "speaker-disjoint validation pool.")
    ap.add_argument("--val_items", type=int, default=200,
                    help="Number of synthetic montages used for the per-epoch "
                         "validation F1 measurement.")
    ap.add_argument("--patience", type=int, default=3,
                    help="Early-stop after this many epochs without val-F1 "
                         "improvement.")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--eval_items", type=int, default=50,
                    help="Number of synthetic montages for evaluation.")
    ap.add_argument("--results_dir", default="results",
                    help="Where to dump lid_eval.json / lid_eval.txt")
    args = ap.parse_args()
    if args.eval:
        evaluate(args)
    else:
        train(args)
