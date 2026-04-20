"""
Task 3.3 -- Synthesize the Maithili (LRL) lecture using Meta MMS-TTS.

Model: facebook/mms-tts-mai (Maithili). MMS-TTS is single-speaker per language;
true zero-shot voice cloning conditioning lives in Task 3.2's prosody warp +
optional spectral-envelope morph. The pipeline order is:

    transcript_lrl.json -> [synthesize_lrl.py] -> synthesized_lrl_raw.wav
                        -> [prosody.py]        -> output_LRL_cloned.wav

We synthesize each transcript segment separately so we can re-time them
later if we need to keep them aligned with the source segment boundaries.

Usage:
    python part3/synthesize_lrl.py
        --transcript results/transcript_lrl.json
        --out results/synthesized_lrl_raw.wav

Notes
-----
* Output sample rate is 16 kHz from MMS; we resample to 22.05 kHz so the
  assignment's "must be 22.05 kHz or higher" requirement is met *before*
  prosody warping (which itself runs at 22.05 kHz).
* Empty / English-only segments are still synthesized (the LRL translation
  step has already converted them to Devanagari script).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

OUT_SR = 22050
SILENCE_BETWEEN_S = 0.20            # short pad between segments


def synth_segment(model, tokenizer, device, text: str) -> np.ndarray:
    if not text or not text.strip():
        return np.zeros(int(0.1 * model.config.sampling_rate), dtype=np.float32)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    audio = out.waveform.squeeze(0).cpu().numpy().astype(np.float32)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", default="results/transcript_lrl.json")
    ap.add_argument("--model", default="facebook/mms-tts-mai",
                    help="Meta MMS-TTS model. Replace with mms-tts-<iso639-3> "
                         "for a different LRL.")
    ap.add_argument("--out", default="results/synthesized_lrl_raw.wav")
    args = ap.parse_args()

    from transformers import VitsModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[synth] loading {args.model} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = VitsModel.from_pretrained(args.model).to(device).eval()
    src_sr = model.config.sampling_rate
    print(f"[synth] model native sr = {src_sr}")

    with open(args.transcript, "r", encoding="utf-8") as f:
        segments = json.load(f)

    pad = np.zeros(int(SILENCE_BETWEEN_S * src_sr), dtype=np.float32)
    chunks = []
    for i, seg in enumerate(segments):
        text = seg.get("text_lrl") or seg.get("text", "")
        try:
            audio = synth_segment(model, tokenizer, device, text)
        except Exception as e:                                  # noqa: BLE001
            print(f"[synth] seg {i} failed ({type(e).__name__}: {e}); "
                  f"replacing with silence")
            audio = np.zeros(int(0.5 * src_sr), dtype=np.float32)
        chunks.append(audio)
        chunks.append(pad)
        if (i + 1) % 10 == 0 or i == len(segments) - 1:
            print(f"[synth] {i+1}/{len(segments)} segments")

    full = np.concatenate(chunks).astype(np.float32)

    # Resample MMS native sr -> 22.05 kHz
    if src_sr != OUT_SR:
        import torchaudio
        t = torch.from_numpy(full).unsqueeze(0)
        t = torchaudio.functional.resample(t, src_sr, OUT_SR)
        full = t.squeeze(0).numpy().astype(np.float32)

    full = np.clip(full, -1.0, 1.0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, full, OUT_SR, subtype="PCM_16")
    print(f"[synth] wrote {full.size/OUT_SR:.1f}s @ {OUT_SR}Hz -> {args.out}")


if __name__ == "__main__":
    main()
