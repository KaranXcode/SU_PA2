# Speech Understanding -- Programming Assignment 2

End-to-end pipeline that ingests a 10-minute code-switched (English/Hindi)
academic lecture, transcribes it with a frame-level LID + constrained
Whisper decoder, translates to **Maithili** (low-resource language), and
re-synthesises the result in **the student's own voice** with the
professor's teaching-style prosody overlaid.

> **Final audio**: `results/output_LRL_cloned.wav`
> **Report**: `report.pdf` (compiled from `report.tex`)
> **1-page implementation note**: `implementation_notes.pdf`

---

## Pipeline at a glance

```
original_segment.wav (10 min)              student_voice_ref.wav (60 s)
        |                                            |
prepare_audio.py                                     |
   |--> 16 kHz mono ---------+                       |
   |--> 22.05 kHz mono ------|--+                    |
                             |  |                    |
                  denoise.py |  |                    |
                             v  |                    |
                  denoised.wav  |                    |
                       |        |                    |
              infer_lid.py      |                    |
                       v        |                    |
              lid_segments.json |                    |
                       |        |                    |
          constrained_decode.py |                    |
                       v        |                    |
              transcript.json   |                    |
                       |        |                    |
            g2p_hinglish.py     |                    |
                       v        |                    |
            transcript_ipa.json |                    |
                       |        |                    |
            translate_lrl.py    |                    |
                       v        |                    |
            transcript_lrl.json |                    |
                       |        |                    |
           synthesize_lrl.py    |                    |
                       v        |                    |
       synthesized_lrl_raw.wav  |                    |
                       |        |                    |
           voice_convert.py     |     <-- kNN-VC ---+
                       v        |
       synthesized_lrl_vc.wav   |
                       |        |
                  prosody.py <--+
                       v
       output_LRL_cloned.wav (FINAL)
                       |
             eval_metrics.py
                       v
              MCD / WER / switch density
```

(See `report.tex` Fig. 1 for a properly-rendered TikZ version of this.)

---

## Datasets

Three classes of audio go into this pipeline; only the first is recorded by
the user, the other two are public.

### 1. User-supplied (not shipped)

| File | Purpose | Expected format |
|---|---|---|
| `part1/data/original_segment.wav` | 10-minute code-switched lecture (test clip) | any SR, any channels; `prepare_audio.py` resamples to 16 kHz mono + 22.05 kHz mono |
| `part1/data/student_voice_ref.wav` | 60-s reference for voice cloning | any SR, any channels; resampled to 16 kHz inside kNN-VC |

Place both before running the pipeline. The 10-min lecture is the Task
1.1/1.2 evaluation input; the 60-s reference is the Task 3.1 input and
the `matching_set` for Task 3.3 kNN-VC.

> **IMPORTANT: training data is NOT in this repo.**
> GitHub's 100 MB per-file limit and ~1 GB repo recommendation forced us
> to exclude the LID training corpora (~660 MB raw audio) and the derived
> feature cache (~2.5 GB). You must download + place them yourself before
> running `train_lid.py`. Instructions below. If you only want to *run*
> the pipeline end-to-end on a new test clip (not retrain the LID), the
> shipped `checkpoints/lid.pt` is sufficient and you can skip steps 2-4.

### 2. LID training corpus -- **LibriSpeech dev-clean (English)**

| Field | Value |
|---|---|
| Source | OpenSLR 12 / LibriSpeech corpus (V. Panayotov et al., 2015) |
| Download URL | [https://www.openslr.org/resources/12/dev-clean.tar.gz](https://www.openslr.org/resources/12/dev-clean.tar.gz) |
| Clip count used | **2,703** `.flac` files (from 40 speakers) |
| Sample rate | 16 kHz mono |
| License | CC-BY 4.0 |
| Archive size | 337 MB compressed / ~365 MB extracted |
| Target placement | `part1/data/cv/en/dev-clean/<speaker>/<chapter>/*.flac` |

**How to fetch and place:**

```bash
# From the repo root
mkdir -p part1/data/cv/en
curl -L https://www.openslr.org/resources/12/dev-clean.tar.gz \
     -o /tmp/dev-clean.tar.gz
tar -xzf /tmp/dev-clean.tar.gz -C part1/data/cv/en/
# You should now see part1/data/cv/en/dev-clean/<speaker-ids>/...
rm /tmp/dev-clean.tar.gz  # optional
```

Or, on Windows PowerShell:

```powershell
mkdir part1\data\cv\en -Force
Invoke-WebRequest -Uri "https://www.openslr.org/resources/12/dev-clean.tar.gz" `
                  -OutFile "$env:TEMP\dev-clean.tar.gz"
tar -xzf "$env:TEMP\dev-clean.tar.gz" -C part1/data/cv/en/
```

`dev-clean` was picked over `train-clean-100` because (a) it is small
enough to download in minutes, and (b) our held-out val split is already
drawn from a different portion of the same distribution, so cross-pool
generalisation is not at issue for an LID task.

**Verify placement:**

```bash
find part1/data/cv/en -name "*.flac" | wc -l    # should print 2703
```

### 3. LID training corpus -- **Hindi speech**

| Field | Value |
|---|---|
| Source | Hindi ASR evaluation set (anonymous speakers, public) |
| Clip count used | **3,843** `.wav` files |
| Sample rate | 16 kHz mono |
| Target placement | `part1/data/cv/hi/*.wav` (flat directory, filenames like `0116_003.wav`) |

This is the `hi` pool of the synthetic montage generator. We initially
planned to use Common Voice 16.1 Hindi, but its streaming download is
painful on Windows and the license requires per-user attribution which
we couldn't easily propagate through the pipeline. Any other balanced
Hindi speech corpus works -- the key requirements are:

- **16 kHz mono WAV or FLAC** (the loader resamples anyway, but
  native-rate files save extraction time).
- **Clips between 2-10 s each** (short enough to compose montages,
  long enough to carry prosody).
- **At least ~2000 clips** per language for the 80/20 train/val split
  to produce a meaningful held-out set.

Recommended alternatives:

- **AI4Bharat IndicSUPERB** (public, permissive): [https://github.com/AI4Bharat/IndicSUPERB](https://github.com/AI4Bharat/IndicSUPERB). Use the Hindi subset of "Kathbath".
- **Common Voice 16.1 Hindi** via Hugging Face: see `datasets.load_dataset("mozilla-foundation/common_voice_16_1", "hi")`. Filter to clips 2--10 s long after download.
- **IIT-M Hindi speech corpus** if you have institutional access.

**Place the clips at `part1/data/cv/hi/` as a flat directory** (no
speaker subfolders needed -- our loader globs `**/*.wav` recursively).

**Verify placement:**

```bash
find part1/data/cv/hi -name "*.wav" | wc -l    # should print >= 2000
```

### 4. Derived artefact: augmented Wav2Vec2 feature cache

| Path | Content | Size |
|---|---|---|
| `part1/data/cv_feats/en/*.npy` | per-clip Wav2Vec2-XLS-R-300M features, post-augmentation | ~950 MB |
| `part1/data/cv_feats/hi/*.npy` | same, Hindi pool | ~850 MB |

**NOT in the repo.** You must regenerate this cache once after placing
the raw audio (steps 2 and 3 above). One-time ~30--60 min pass on CPU:

```bash
cd part1
python augment_and_extract.py --data_dir data/cv --out_dir data/cv_feats
# Resumable -- safe to Ctrl+C and restart.
# Wrote feature shape: (T_frames, 1024) float16 per clip
```

Each `.npy` is the frozen Wav2Vec2 activation stream for one augmented
clip. **This cache is the thing the LID trainer actually reads** -- the
raw `cv/` directories are touched only by the extractor. Once the cache
is built, you can delete `cv/` to save disk if you want (but you'll
need to re-download to re-augment).

**Verify cache:**

```bash
ls part1/data/cv_feats/en | wc -l    # should be ~2000
ls part1/data/cv_feats/hi | wc -l    # should be ~2000
```

### 4a. Full first-time data-setup sequence (copy-paste)

For a clean clone, the full dataset + feature setup is:

```bash
# 1. English: download LibriSpeech dev-clean + extract
mkdir -p part1/data/cv/en
curl -L https://www.openslr.org/resources/12/dev-clean.tar.gz \
     -o /tmp/dev-clean.tar.gz
tar -xzf /tmp/dev-clean.tar.gz -C part1/data/cv/en/

# 2. Hindi: download your preferred source (see above) and place clips
#    at part1/data/cv/hi/*.wav

# 3. Build augmented feature cache (~30-60 min on CPU, resumable)
cd part1
python augment_and_extract.py --data_dir data/cv --out_dir data/cv_feats
cd ..

# 4. Train LID (~15 min on CPU, resumes from cached features)
cd part1
python train_lid.py \
    --data_dir data/cv --feats_dir data/cv_feats \
    --epochs 8 --batch 16 --n_items 3000 --val_items 300 \
    --patience 3 --out ../checkpoints/lid.pt --results_dir ../results
cd ..

# 5. Run the full pipeline on your 10-min lecture
python pipeline.py
```

If you only want to evaluate an already-trained LID (no retraining),
the shipped `checkpoints/lid.pt` handles it; you can skip steps 1--4
and go straight to `python pipeline.py` after placing your 10-min
`part1/data/original_segment.wav`.

### 5. External pretrained models (downloaded on first run)

Cached under `~/.cache/huggingface/hub/` or `~/.cache/torch/hub/`.

| Model | Used by | Size |
|---|---|---|
| `facebook/wav2vec2-xls-r-300m` | LID, speaker embedding | 2.4 GB |
| `openai/whisper-medium` | constrained ASR | 2.9 GB |
| `facebook/mms-tts-mai` | Part 3 synthesis | 140 MB |
| `microsoft/wavlm-large` (via torch.hub / bshall/knn-vc) | kNN-VC | 1.2 GB |
| HiFi-GAN pre-matched (bshall/knn-vc release) | kNN-VC vocoder | 63 MB |

### 6. Training / evaluation splits

| Split | Source | Purpose |
|---|---|---|
| LID train pool | 80% of each language's clips (seed 12345) | backbone feature extraction + LID head training |
| LID val pool | 20% of each language's clips | per-epoch val macro-F1, early-stopping |
| LID eval | 300 freshly-sampled montages from val pool | `results/lid_eval.json` confusion matrix |
| ASR test | user's 10-min lecture | `transcript.json` |
| CM train / test | 80/20 split of sliced windows from (bonafide, spoof) | `results/cm_eer.json` |
| Adversarial target | longest HI segment in LID output | `results/adversarial.json` |

### Dataset summary table

| Class | Language | Clips | Duration | Location |
|---|---|---|---|---|
| LID train pool (EN) | English | 1,600 | ~4.5 hrs | `part1/data/cv/en/dev-clean/` |
| LID val pool (EN) | English | 400 | ~1.1 hrs | same, 20% split |
| LID train pool (HI) | Hindi | 1,600 | ~2.5 hrs | `part1/data/cv/hi/` |
| LID val pool (HI) | Hindi | 400 | ~38 min | same, 20% split |
| Test audio | EN + HI (CS) | 1 | 10 min | `part1/data/original_segment.wav` |
| Voice reference | student | 1 | 68 s | `part1/data/student_voice_ref.wav` |
| Syllabus corpus (for LM) | English | -- | ~3 KB text | `part1/syllabus.txt` |

---

## Trained model weights

The repo ships three trained checkpoints under `checkpoints/`:

| File | Size | Content |
|---|---|---|
| `lid_head.pt`   | **11.6 MB** | **The final LID weights used for evaluation.** Head-only (proj + BiLSTM + 2 classifier heads); backbone is restored from HuggingFace at load time. |
| `cm.pt`         | 432 KB     | Task 4.1 anti-spoof CM (LFCC + Conv2D + GRU) |
| `lm.pkl`        | 22 KB      | Task 1.2 trigram N-gram LM on `syllabus.txt` |

### Why a "head-only" LID checkpoint?

`torch.save(model.state_dict())` inside `train_lid.py` originally produced
a 1.2 GB file because the state dict included the full frozen
Wav2Vec2-XLS-R-300M backbone (300M parameters). GitHub rejects any single
file > 100 MB, so committing `lid.pt` is impossible.

However, the backbone is **never updated during training** -- it's
reloaded fresh from `facebook/wav2vec2-xls-r-300m` every time
`FrameLID()` is instantiated. So all we actually need to save are the
learned head weights: the linear projection (`proj`), channel dropout
(`feat_dropout`), 2-layer BiLSTM (`lstm`), and the two classifier heads
(`head_main`, `head_switch`). That's 22 tensors totalling 11.6 MB.

At inference time, `infer_lid.py` and `adversarial.py` both load with
`strict=False`, so missing `backbone.*` keys are simply ignored (they're
already populated by the HuggingFace download).

### Regenerating `lid_head.pt` from a full checkpoint

If you (re)train the LID and end up with a full `checkpoints/lid.pt`, run:

```bash
python part1/strip_ckpt.py --in checkpoints/lid.pt --out checkpoints/lid_head.pt
# [strip] kept 22 head tensors, dropped 422 backbone tensors
# [strip] size: 1273.3 MB -> 11.6 MB (0.9%)
```

`train_lid.py` now also writes the head-only variant automatically
alongside every full checkpoint it saves (since this iteration), so if
you retrain with the current code you'll see both `lid.pt` (1.2 GB,
git-ignored) and `lid_head.pt` (11.6 MB, committed) appear in
`checkpoints/`.

### Loading the checkpoint

Already done for you in the scripts that need it:

```python
from lid_model import FrameLID
model = FrameLID()                                   # backbone from HF
ckpt  = torch.load("checkpoints/lid_head.pt", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)   # strict=False is key
model.eval()
```

`pipeline_part1.py` and the top-level `pipeline.py` both point at
`checkpoints/lid_head.pt` by default; no CLI flag needed.

### What's git-ignored (and why)

The following are excluded via `.gitignore` (see the file for the full list):

| Path | Reason | Regenerate by |
|---|---|---|
| `checkpoints/lid.pt` (1.2 GB) | > 100 MB file limit | `train_lid.py` or use `lid_head.pt` |
| `checkpoints/lid_last.pt` | debug-only last-epoch weights | `train_lid.py` |
| `part1/data/cv/` (663 MB) | raw audio; re-downloadable | dataset instructions above |
| `part1/data/cv_feats/` (2.5 GB) | feature cache; regeneratable | `augment_and_extract.py` |
| `Hindi_test.tar.gz`, `dev-clean.tar.gz` | each > 100 MB | download from source |
| `__pycache__/`, `*.aux`, `*.log` | standard build artifacts | automatic |

---

## Repository layout

```
Assin_2/
|-- pipeline.py                  # Top-level orchestrator (Part 0 -> IV + eval)
|-- eval_metrics.py              # MCD + WER + LID switch density
|-- mcd_stages.py                # Per-stage MCD probe (debug helper)
|-- requirements.txt
|-- report.tex                   # 10-page IEEE-format report
|-- implementation_notes.tex     # 1-page non-obvious-design-choices doc
|-- README.md                    # this file
|
|-- Audio Manifest/              # ALL audio files in one place (submission bundle)
|   |-- original_segment.wav           # source 10-min lecture (mono 16 kHz)
|   |-- original_segment_22k.wav       # same, resampled to 22.05 kHz
|   |-- student_voice_ref.wav          # 60-s reference recording
|   |-- denoised.wav                   # Task 1.3 denoised lecture
|   |-- synthesized_lrl_raw.wav        # Task 3.3 MMS default speaker
|   |-- synthesized_lrl_vc.wav         # after kNN-VC (your voice)
|   |-- output_LRL_flat.wav            # ablation: VC without prosody warp
|   |-- output_LRL_cloned.wav          # FINAL: VC + prosody warp
|   
|
|-- part1/                       # Robust Code-Switched Transcription
|   |-- prepare_audio.py         # Part 0: mono 16k + 22.05k from supplied WAV
|   |-- denoise.py               # 1.3: spectral subtraction (DFN fallback)
|   |-- lid_model.py             # 1.1: FrameLID (XLS-R + BiLSTM + 2 heads)
|   |-- train_lid.py             # 1.1: synthetic CS training + early stop
|   |-- augment_and_extract.py   # 1.1: audio aug + feature cache builder
|   |-- extract_features.py      # 1.1: clean feature cache (no aug)
|   |-- infer_lid.py             # 1.1: chunked inference -> segments JSON
|   |-- ngram_lm.py              # 1.2: trigram LM (KN-smoothed) on syllabus
|   |-- constrained_decode.py    # 1.2: Whisper + logit bias + lang routing
|   |-- syllabus.txt             # 1.2: technical-term corpus
|   |-- pipeline_part1.py        # Part I orchestrator
|   |-- prep_cv.py               # (one-time) Common Voice + LibriSpeech prep
|   `-- data/
|       |-- original_segment.wav         # YOU SUPPLY: 10-min source
|       |-- original_segment_22k.wav     # auto-generated by prepare_audio.py
|       |-- student_voice_ref.wav        # YOU SUPPLY: 60-s reference
|       |-- cv/{en,hi}/                  # YOU SUPPLY: training clips
|       `-- cv_feats/{en,hi}/            # auto-generated by augment_and_extract.py
|
|-- part2/                       # Phonetic Mapping & LRL Translation
|   |-- g2p_hinglish.py          # 2.1: EN+HI -> unified IPA
|   |-- lrl_dict.tsv             # 2.2: 500+ technical-term parallel corpus
|   |-- translate_lrl.py         # 2.2: Hindi -> Maithili (dict + transliterate)
|   |-- quick_transcript.py      # debug: vanilla Whisper baseline
|   `-- pipeline_part2.py        # Part II orchestrator
|
|-- part3/                       # Zero-Shot Cross-Lingual Voice Cloning
|   |-- voice_embed.py           # 3.1: speaker embedding (Wav2Vec2 mean-pool)
|   |-- synthesize_lrl.py        # 3.3: MMS-TTS-mai (VITS) per-segment synth
|   |-- voice_convert.py         # NEW: kNN-VC (WavLM + HiFi-GAN), k=1
|   |-- prosody.py               # 3.2: DTW(F0+E) + global time-stretch
|   `-- pipeline_part3.py        # Part III orchestrator
|
|-- part4/                       # Adversarial Robustness & Spoofing
|   |-- antispoof.py             # 4.1: LFCC + Conv2D + GRU + EER
|   |-- adversarial.py           # 4.2: PGD attack on FrameLID
|   `-- pipeline_part4.py        # Part IV orchestrator
|
|-- checkpoints/
|   |-- lid.pt                   # trained Part 1.1 weights
|   |-- lid_last.pt              # last-epoch (debug)
|   |-- cm.pt                    # trained Part 4.1 weights
|   `-- lm.pkl                   # trained Part 1.2 N-gram LM
|
`-- results/                     # ALL evaluation outputs land here
    |-- denoised.wav             # 1.3 output
    |-- lid_segments.json        # 1.1 output
    |-- lid_eval.json            # 1.1 macro-F1 + per-class metrics
    |-- lid_eval_plots.png       # 1.1 confusion matrix + bars
    |-- lid_train_curves.png     # 1.1 training plots
    |-- transcript.json          # 1.2 ASR
    |-- transcript_ipa.json      # 2.1 IPA
    |-- transcript_lrl.json      # 2.2 Maithili text
    |-- synthesized_lrl_raw.wav  # 3.3 MMS default-speaker
    |-- synthesized_lrl_vc.wav   # 3   after kNN-VC (your voice)
    |-- output_LRL_flat.wav      # ablation baseline (no prosody warp)
    |-- output_LRL_cloned.wav    # FINAL (your voice + prosody warp)
    |-- student_xvector.npy      # 3.1 deliverable
    |-- cm_eer.json              # 4.1 EER
    |-- adversarial.json         # 4.2 minimum epsilon
    |-- adversarial_chunk.wav    # 4.2 attack waveform
    `-- eval_metrics.json        # MCD + WER + switch density (final)
```

---

## Install

The PyTorch stack must be installed in a single command so versions match.
On CPU-only Windows / Linux:

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you have CUDA, replace `/cpu` with the matching CUDA channel
(e.g. `/cu121`) -- but everything in this assignment runs fine on CPU.

### Disk requirements

The Hugging Face cache fills up fast. You need **at least 6 GB free**:

| Model | Size | Triggered by |
|---|---|---|
| `facebook/wav2vec2-xls-r-300m` | 2.4 GB | LID, voice embedding |
| `openai/whisper-medium` | 2.9 GB | constrained ASR |
| `facebook/mms-tts-mai` | 140 MB | Part 3 synthesis |
| WavLM-Large + HiFi-GAN (torch.hub) | 1.4 GB | kNN-VC |

### One-time first-run downloads

The first time you run the pipeline, expect ~5-10 minutes of downloads
into `~/.cache/huggingface/hub/` and `~/.cache/torch/hub/`. Subsequent
runs are offline-friendly.

---

## How to run

### Quickest path (full pipeline)

```bash
python pipeline.py
```

This assumes you've already:
1. Placed `part1/data/original_segment.wav` (your 10-min lecture).
2. Placed `part1/data/student_voice_ref.wav` (your 60-s recording).
3. Trained the LID + built the LM (one-time -- see "First-time setup" below).

The full pipeline takes ~25-40 minutes on CPU.

### Re-running individual parts

Each part is independently re-runnable. Example: re-do only Part 3
because you tweaked kNN-VC k:

```bash
python part3/pipeline_part3.py \
    --ref part1/data/student_voice_ref.wav \
    --src part1/data/original_segment_22k.wav \
    --vc_k 1
```

Skip flags on the orchestrator:

```bash
python pipeline.py --skip_part1 --skip_part2     # only Part 3 + 4 + eval
python pipeline.py --skip_eval                   # everything except slow WER
```

---

## First-time setup (do these once)

### 1. Place audio files

```
part1/data/original_segment.wav    # 10-min lecture (any sample rate)
part1/data/student_voice_ref.wav   # 60-s of you reading
```

The `prepare_audio.py` script will derive the 22.05 kHz copy automatically.

### 2. Get Common Voice + LibriSpeech (LID training data)

```bash
# Hindi: Common Voice 16 (Hindi subset)
# English: LibriSpeech dev-clean
python part1/prep_cv.py
# -> populates part1/data/cv/{en,hi}/
```

### 3. Build the augmented feature cache (one-time, ~30-60 min on CPU)

```bash
cd part1
python augment_and_extract.py --data_dir data/cv --out_dir data/cv_feats
```

This applies reverb + noise + speed perturbation to each clip and runs
WavLM-XLS-R once over the augmented audio, caching features as `.npy`.
Subsequent LID training reads only the cache and is ~100x faster than
rerunning the backbone.

### 4. Train the LID model

```bash
cd part1
python train_lid.py \
    --data_dir data/cv \
    --feats_dir data/cv_feats \
    --epochs 8 --batch 16 --n_items 3000 --val_items 300 --patience 3 \
    --out ../checkpoints/lid.pt --results_dir ../results
```

Expected: best val macro-F1 ~0.97-0.99 after 5-7 epochs.

### 5. Build the N-gram LM

```bash
cd part1
python ngram_lm.py build --corpus syllabus.txt --out ../checkpoints/lm.pkl
```

### 6. (Optional) Train a baseline LID without augmentation

```bash
python part1/extract_features.py --data_dir data/cv --out_dir data/cv_feats_clean
python part1/train_lid.py --data_dir data/cv --feats_dir data/cv_feats_clean ...
```

Useful for the report's "augmentation impact" discussion.

---

## Evaluation thresholds

The assignment's "strict passing criteria":

| Metric | Threshold | Where measured |
|---|---|---|
| LID macro-F1 (EN vs HI) | >= 0.85 | `results/lid_eval.json` |
| WER -- English segments | < 15% | `results/eval_metrics.json` |
| WER -- Hindi segments | < 25% | `results/eval_metrics.json` |
| LID switch precision | <= 200 ms | qualitative on real audio |
| MCD -- cloned vs ref | < 8.0 | `results/eval_metrics.json` |
| Spoof EER | < 10% | `results/cm_eer.json` |
| Adversarial min epsilon | report value | `results/adversarial.json` |

See `report.tex` Section VII for our measured numbers and discussion.

---

## Common pitfalls

### `OSError [WinError 127] / operator torchvision::nms does not exist`
Your torch / torchaudio / torchvision versions disagree. Reinstall all
three together with the command in the **Install** section above.

### `OMP: Error #15 ... libiomp5md.dll already initialized`
Windows + conda OpenMP collision. Set the env var:
```powershell
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
```
The `pipeline.py` orchestrator does this automatically; only matters if
you run a sub-script directly.

### `UnicodeEncodeError 'charmap' codec ...` when printing Devanagari
Set `$env:PYTHONIOENCODING = "utf-8"` in PowerShell. Again, `pipeline.py`
sets this for sub-processes.

### `RuntimeError: not enough memory ... 1.2 GB`
WavLM OOMs on multi-minute audio. `voice_convert.py` chunks at 20 s by
default; lower `--chunk_s 10` if you still hit it.

### kNN-VC `torchcodec` import error
torchaudio 2.8 dispatches `torchaudio.load` through `torchcodec` which
isn't a default Windows dep. `voice_convert.py` monkey-patches
`torchaudio.load` to use `soundfile` before kNN-VC loads -- no extra
install needed.

### MMS-TTS `narrow(): length must be non-negative`
Tokeniser edge case on very short / empty translation strings. Affected
segments are silence-replaced; no action needed.

### Whisper repetition loops on Hindi (`के लिएटर के लिएटर ...`)
Fixed by `no_repeat_ngram_size=3` + `repetition_penalty=1.15` +
duration-scaled `max_new_tokens` in `constrained_decode.py`.

### `eval_metrics.py` MCD shows weird high numbers (40+, 200+)
Classical MCD assumes parallel utterances. Our metric is the speaker-mean
variant defined in `eval_metrics.py:mcd()` -- see report Sec. VII.D for
the formulation. Numbers in the 8-15 range are typical for our setup.

---

## Deviations from the assignment spec (declared up front)

1. **Whisper-medium instead of Whisper-v3** -- v3 is 3.1 GB; our eval
   machine had 2.9 GB free disk. The `LogitsProcessor` integration is
   identical; only encoder depth differs.
2. **DeepFilterNet -> spectral subtraction** -- DFN's ONNX runtime
   refused to load on our Windows install. `denoise.py` falls back to a
   PyTorch implementation of Boll 1979 spectral subtraction.
3. **Wav2Vec2 mean-pool x-vector instead of ECAPA-TDNN** -- SpeechBrain's
   pip install fails on our Windows/miniconda combination. The Wav2Vec2
   fallback path produces a 1024-dim d-vector, used as the Task 3.1
   deliverable.
4. **kNN-VC inserted between MMS-TTS and prosody warp** -- MMS-TTS is
   single-speaker per language and cannot be directly conditioned on the
   student's voice. We add kNN-VC (Baas et al. 2023) so the final audio
   genuinely speaks in the student's voice.
5. **Non-parallel MCD formulation** -- classical parallel MCD is
   inapplicable when ref and synth differ in language and content; see
   report Sec. VII.D for our speaker-mean MFCC formulation.
6. **PGD instead of single-step FGSM** -- single-step FGSM was too weak
   to flip predictions at SNR > 40 dB. We use 20-step PGD which finds a
   working perturbation at SNR ~80 dB.

---

## Citations

See `report.tex` bibliography for full citations. Key works:

- Whisper -- Radford et al. (OpenAI 2022)
- XLS-R -- Babu et al. (Interspeech 2022)
- MMS -- Pratap et al. (Meta AI 2023)
- kNN-VC -- Baas, van Niekerk, Kamper (Interspeech 2023)
- WavLM -- Chen et al. (IEEE JSTSP 2022)
- HiFi-GAN -- Kong et al. (NeurIPS 2020)
- PGD -- Madry et al. (ICLR 2018)
- Spectral subtraction -- Boll (IEEE TASSP 1979)
- pYIN -- Mauch & Dixon (ICASSP 2014)
