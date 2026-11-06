# PolyGuard — handwritten prescription VLM evaluation

Component 1 only: image → Gemini → validated medicine information → research evaluation.
No frontend, interaction checker, database, training, fine-tuning, or deployment.

**Delivery status:** code and offline validation are complete. Live handwriting evaluation is **not run**: the dataset download returned HTTP 403 in the build environment, and no Gemini key was configured. No real prescription images, model predictions, or accuracy figures are bundled. Start with `reports/experiment_report.md` for the exact evidence.

## 1. Windows / VS Code setup

Use Python **3.11 or 3.12**; validated on 3.12. Extract the ZIP and open the `polyguard-vlm` folder in VS Code. In its PowerShell terminal:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

The explicit interpreter path avoids PowerShell activation-policy problems. In VS Code, select `.venv\Scripts\python.exe` with **Python: Select Interpreter**. On macOS/Linux use `python3 -m venv .venv` and `.venv/bin/python` instead.

Create a key in [Google AI Studio](https://aistudio.google.com/apikey), then put it in `.env`:

```dotenv
GEMINI_API_KEY=your_actual_key
GEMINI_MODEL=gemini-3.8-flash
```

The model ID is configured in one place, not embedded in extraction code. Its image-input and structured-output capabilities were checked against [Google's model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash) on 17 September 2026. Availability in your account was not tested. If rejected, list available models, select a vision-capable structured-output model, and update `.env`:

```powershell
.\.venv\Scripts\python.exe -m src.main models
```

The official [google-genai SDK](https://github.com/googleapis/python-genai) is used. `google-generativeai` is not used. `requirements-tested.txt` pins the direct package versions validated here; `requirements.txt` allows compatible updates. `.env`, output records, raw images and Kaggle credentials are ignored by Git. No actual key is bundled.

## 2. Obtain the selected dataset

Selected: **A Curated Bangladesh-Based Dataset of Handwritten and Printed Prescription Images, version 2**, by Dipto Saha, Sourav Debnath and Anirban Roy, [DOI 10.17632/k62rfd23kz.2](https://data.mendeley.com/datasets/k62rfd23kz/2). The publisher reports 200 de-identified prescription pages, mixed handwritten/printed and Bangla/English content, CC BY 4.0. Its full-page context and stated provenance suit this task better than isolated drug-word crops. Actual difficulty distribution remains unverified.

Try automatic download:

```powershell
.\.venv\Scripts\python.exe -m src.main download
```

The build environment returned HTTP 403, recorded in `reports/download_status.json`. If automatic download is blocked on your computer too:

1. Open the dataset page above, select **Download All**, and save the ZIP.
2. Import it without modifying the original archive:

```powershell
.\.venv\Scripts\python.exe -m src.main download --archive "$HOME\Downloads\YOUR_DOWNLOADED_FILE.zip"
```

The importer preserves nested folders under `data/raw/selected/`; an `images/` folder can be nested further. Discovery is recursive, so there is no dependency on the ZIP's exact top-level folder name. Nonempty destinations are refused to protect originals; use `--destination data/raw/another-dataset` to import another archive. No dataset images are redistributed in this project.

### Optional Kaggle integration

The selected Mendeley dataset does not require Kaggle credentials. For an alternative dataset, install the optional CLI:

```powershell
.\.venv\Scripts\python.exe -m pip install kaggle
```

In Kaggle account settings, create an API credential. Either configure `KAGGLE_API_TOKEN` in your environment, or save a legacy `kaggle.json` as `%USERPROFILE%\.kaggle\kaggle.json` (PowerShell: `$HOME\.kaggle\kaggle.json`). Never put it inside the project or send it in chat. The integration detects the token, username/key pair, or file without printing values. With the virtual environment activated so the `kaggle` executable is on PATH:

```powershell
.\.venv\Scripts\Activate.ps1
python -m src.main download --kaggle OWNER/DATASET-SLUG --destination data/raw/kaggle
```

Use an actual dataset slug whose source and license you have checked. Candidate comparisons and unresolved licensing questions are in `reports/dataset_report.md`.

## 3. Inspect before running

```powershell
.\.venv\Scripts\python.exe -m src.main inspect
```

Outputs:

- `data/metadata.csv`: formats, decoded dimensions, validation status, hashes, duplicates and label availability.
- `data/ground_truth.csv`: an empty annotation template populated with discovered image paths; no invented labels.
- `reports/inspection.json`: counts and candidate label files.
- `reports/contact_sheet.jpg`: up to 24 previews when images exist.

The original files stay intact. Inspection detects exact decoded-pixel duplicates, corrupted and oversized images. Batch selection removes byte-identical duplicate files. Visually near-identical scans or crops need human review; this is not a perceptual duplicate detector. Use metadata to assemble a unique evaluation folder when pixel-duplicate files differ in encoding.

## 4. Add ground truth without inventing it

The source's medicine bounding boxes and counts **are not transcribed medicine names or instructions**. They cannot establish name/dosage accuracy. Preserve its original `labels/` files. Initially, all images remain qualitative-only.

Annotate before viewing predictions; ideally use two independent qualified readers and resolve disagreements. Never resolve a disputed name using model output. In `ground_truth.csv`, use paths relative to the directory supplied with `--data`, with `/` separators. Edit using a CSV-aware editor.

| Column | Contents |
|---|---|
| `image_path` | e.g. `images/rx_001.jpg`; preserve actual relative path |
| `raw_ground_truth` | Original human transcription, unchanged |
| `medicine_names` | JSON array, e.g. `["DrugA", "DrugB"]` |
| `dosages`, `frequencies`, `durations`, `routes`, `instructions` | Aligned JSON arrays, e.g. `["5 mg", null]`; null is unavailable/absent |
| `handwriting_type` | `very_messy`, `moderate`, `handwritten_unspecified`, `printed`, or `unknown` |
| `annotation_status` | `complete`, `partial`, or `unavailable` |
| `notes` | Annotator, ambiguity, visibility issues, original label provenance |

The DrugA/DrugB text above is a format example, not a prescription or dataset label.
An empty `medicine_names` cell means **unknown**; `[]` with `complete` means a human verified there are no medicines. `complete` requires the full medicine-name list; unreadable names require `partial`. A complete name list can still have unavailable dosage fields. Do not guess a name to make an image quantitatively scorable.

Inspect never replaces existing annotations. Newly discovered paths go into `ground_truth_new_rows.csv` for deliberate merging. Image types are human-assigned, not guessed from filenames. Very messy means substantial joined/overlapping or illegible medicine writing; moderate means handwritten but mostly separable and readable. Record blur, shadows, folds and language separately in notes. Predefine these criteria before evaluating.

## 5. Run Gemini

Start with a small batch; calls can incur provider charges. The default is 10 images, ordered by human difficulty labels with printed baselines last. No GPU is needed.

```powershell
# One image (replace with an actual path)
.\.venv\Scripts\python.exe -m src.main run --image data/raw/selected/images/rx_001.jpg

# Ten selected images, then a larger sample
.\.venv\Scripts\python.exe -m src.main run --limit 10
.\.venv\Scripts\python.exe -m src.main run --limit 100

# All images, or only the human-tagged hardest category
.\.venv\Scripts\python.exe -m src.main run --all
.\.venv\Scripts\python.exe -m src.main run --all --category very_messy
```

Configure requests per minute, retries, timeout and image limits in `.env`. The runner paces every attempt, retries temporary network/429/selected 5xx errors, and honours numeric Retry-After headers. HTTP-date Retry-After values fall back to exponential backoff. Authentication and schema failures are not blindly retried. Malformed model outputs are recorded as failures, never repaired by inventing content. Fenced but otherwise exact JSON is accepted and flagged.

Every successful output is cached using image content, model, prompt, schema, software version and preprocessing configuration. Repeat the same command to resume: successes reuse cache and failed images are retried. Raw response text and usage metadata are retained locally with each run. Model version aliases can change server-side; use a new output folder to force a fresh experiment when that matters.

For a genuinely cached demo with **zero API calls**:

```powershell
.\.venv\Scripts\python.exe -m src.main run --limit 10 --replay
```

This reports cache misses, not fabricated responses. Keep the same model/preprocessing configuration. A hard process crash can leave `outputs/.run.lock`; only after confirming no runner is active, remove that file and rerun. Normal exceptions and Ctrl+C release the lock. Each completed image already has its own atomic record.

## 6. Preprocessing experiments

Every variant applies EXIF orientation, RGB conversion, alpha-to-white compositing, metadata removal and aspect-preserving size limits. `original` means this safe baseline, not unmodified uploaded bytes. Contrast and grayscale are separate optional experimental variants, never changes to raw originals.

```powershell
.\.venv\Scripts\python.exe -m src.main run --limit 10 --variant contrast
.\.venv\Scripts\python.exe -m src.main run --limit 10 --variant grayscale
.\.venv\Scripts\python.exe -m src.main run --image data/raw/selected/images/rx_001.jpg --rotation 90

# Paired comparison: same selected images; up to 3 × limit fresh requests
.\.venv\Scripts\python.exe -m src.compare --limit 10
```

Comparison outputs live in `outputs/comparison/`, including `comparison.csv`. It uses its own caches. Missing labels produce unavailable scores, never a claimed winner. Compare missing/extra names and failures as well as F1. Rotation is manual clockwise 0/90/180/270 plus EXIF handling. Never choose a rotation simply because it produces more medicines; that can reward hallucinations.

## 7. Read the outputs

`outputs/` contains the latest exports. `outputs/runs/<timestamp_id>/` retains every run, its exact manifest, per-image JSON, logs and exports. `outputs/cache/` allows recovery; `outputs/processed/` holds the actual PNG inputs used for API calls.

| File | Purpose |
|---|---|
| `predictions.jsonl` | One full result envelope per image, including failures and raw response |
| `predictions.csv` | One row per extracted medicine |
| `evaluation_report.csv` | Image-level counts and labelled field scores |
| `metrics.json` | Operational metrics and separate difficulty-group scores |
| `errors.csv` | Failed images with safe error codes and retry counters |
| `qualitative_review.csv` | Outputs, original-image paths and blank human-review fields |

A prediction includes raw text, names, dose, frequency, duration, route, instructions, subjective confidence, unclear fragments and notes. `image_id` is attached locally, so the model cannot change the image identity. An absent field is `null`; a visible but unreadable field is `UNCLEAR`.

All outputs need human review. Confidence is self-reported, not a calibrated probability of correctness. The pipeline does not guarantee that UNCLEAR detection catches every hallucination. It neither expands abbreviations nor maps brands to ingredients.

### Evaluation definitions

Only **complete medicine-name annotations** enter name precision/recall/F1. Partial and unlabelled images stay qualitative. Failed selected images with complete labels count as missing medicines. There are separate case-sensitive, case-insensitive (whitespace/case normalization), and fuzzy (SequenceMatcher ratio ≥ 0.85) one-to-one match scores. Fuzzy similarity is diagnostic, not medical equivalence. Duplicate predicted names cannot reuse a ground-truth match.

Detail accuracy is case-insensitive exact text agreement for annotated fields, conditioned on case-insensitive name alignment. A missed medicine's annotated details count as wrong. Null/unreadable labels are excluded. Repeated identical medicine names with differing instructions can make automatic alignment ambiguous; review those rows. No units or clinical abbreviations are converted.

Unmatched predictions are **extra names**, not automatically confirmed hallucinations: they can be spelling mistakes or annotation problems. `confident_unmatched_names` uses confidence ≥ 0.8 as a review flag. Confirmed hallucinations remain `null` until adjudicated manually in the review CSV; the program does not yet ingest those adjudications into aggregate metrics. Unclear names are abstentions, not invented drugs. Incomplete prescriptions count labelled images with missed names.

JSON validity is measured over obtained model responses (not input failures); schema validity is separate. Missing denominators are `null`. Operational success is not transcription correctness. Current-run calls/retries/errors exclude cached runs; source latency includes rate-limit waiting and retries. Token usage is saved, but cost is not guessed from stale prices. No patient-level uncertainty intervals or model confidence calibration are implemented.

After adding labels, rescore without calling Gemini:

```powershell
.\.venv\Scripts\python.exe -m src.main evaluate
```

## 8. Replace data / run tests

Put your authorized, de-identified images in another folder. Supported formats: JPG, JPEG, PNG, WEBP. Folder names and layouts can differ.

```powershell
.\.venv\Scripts\python.exe -m src.main inspect --data "D:\MyPrescriptions" --truth data/my_ground_truth.csv
.\.venv\Scripts\python.exe -m src.main run --data "D:\MyPrescriptions" --truth data/my_ground_truth.csv --limit 10
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Use the same root for inspection and batch processing so paths align. For a single image outside the specified root, its parent is used and its ID is its filename. Keep different studies in separate output folders via `--output`.

Tests use artificial pixel fixtures and a deterministic fake transport, explicitly not real handwritten prescriptions. They check image corruption, EXIF rotation, limits, JSON/schema rejection, matching, retry classification, cache reuse/invalidation, failure continuation, ZIP traversal rejection and partial-ground-truth handling.

## 9. Structure and limits

```text
polyguard-vlm/
  README.md, requirements.txt, requirements-tested.txt, .env.example
  prompts/extract.txt
  src/                 # config, schema, images, downloader, inspection,
                       # Gemini client, parser, runner, evaluator, CLI, comparison
  tests/test_pipeline.py
  data/raw/, data/processed/, data/samples/
  data/metadata.csv, data/ground_truth.csv
  outputs/             # exports; populated by real runs later
  reports/             # dataset selection, experiment status, test evidence
```

Research prototype only, **not a clinical decision-making tool**. Use only images you are authorized to process through a third-party API. De-identification must happen before a live call; the tool removes EXIF but does not redact visible patient information. Stored outputs may retain all visible text. The source's stated anonymization is not a guarantee for replacement datasets.

The attached PDF was used for context, not copied uncritically: model IDs were checked; no “choose the rotation with most drugs” rule, guessed clinical expansion, or fabricated live demo was adopted. Next steps are obtaining the data, stratifying handwriting difficulty, independently annotating a held-out set, running the paired experiment, and reporting errors and uncertainty alongside scores. Freeze the held-out set before changing the prompt to avoid tuning on the test set.
