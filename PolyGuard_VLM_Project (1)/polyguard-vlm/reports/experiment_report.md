# Initial implementation and validation report

Date: 17 September 2026. Scope: PolyGuard Component 1, prescription VLM extraction and evaluation only.

## Outcome

Implemented a modular Python CLI using the official Google Gen AI SDK, strict Pydantic schema, conservative handwritten-prescription prompt, non-destructive image handling, batch retries and rate limits, atomic per-image records, configuration-aware success caching, qualitative review exports and quantitative metrics. Windows instructions, dataset research, safe archive import and paired preprocessing comparison are included.

**Live extraction was not run. This package does not demonstrate handwriting recognition accuracy.**

| Item | Verified result |
|---|---|
| Offline test suite | **39 tests passed**; see `test_results.txt` |
| Python | 3.12; exact version in `environment.json` |
| Gemini SDK | google-genai 1.75.0 installed and schema/retry config instantiated |
| Model configured in example | `gemini-3.8-flash`; documentation verified; account access not tested |
| Gemini credential | Not configured; no live API calls |
| Kaggle credentials | No environment token, username/key pair or default credential file found |
| Selected dataset | Mendeley k62rfd23kz version 2; full-page prescriptions |
| Publisher-reported images | 200; this is not a local measurement |
| Automatic archive download | Attempted; HTTP 403 |
| Actual prescription images obtained | 0 |
| Usable actual prescription images here | 0 |
| Source handwritten-image count | Unknown; no visual access to archive |
| Local ground-truth transcriptions | None |
| Actual prescription accuracy / API latency | Unavailable |

The inspection command was executed against the empty acquisition folder to record the actual local state. Its zero counts do not mean the published source dataset is empty. No contact sheet can be produced without images. The supplied PDF and markdown were read as project specifications, not treated as prescription samples.

## Test coverage

The suite uses temporary artificial pixel fixtures and mocked model transport. It covers:

- JPG/JPEG/PNG/WEBP loading, corrupt/empty/missing/unsupported inputs, pixel limits, EXIF rotation, aspect ratio and raw-file preservation.
- Strict JSON parsing, fence handling, truncated/prose/missing-field rejection, invalid confidence, NaN, uncertainty consistency and legitimate empty medicine lists.
- ZIP traversal prevention and refusal to overwrite raw archives; duplicate-image inspection and invalid/alignment-sensitive ground truth.
- Exact/normalized/fuzzy one-to-one matching, failure-as-missing scoring, exclusion of partial annotations, unavailable metric denominators and confident-extra review flags.
- Success-cache reuse, preprocessing cache invalidation, continued processing after an invalid image, filename collisions in separate folders, keyless failure before API work, cache-only replay.
- Transient network retry and immediate nontransient authentication failure.

A resource warning in a test's image reader was fixed by closing its file handle; the final run is clean. Retry behaviour is mocked, not measured against Google's service. The structured-output request configuration was constructed with the installed SDK but not sent to a real model.

## Important evaluation decisions

No ground-truth labels were fabricated. The selected source's bounding boxes are not medicine transcriptions. Partial annotation sets are excluded from aggregate name precision/recall so missing labels cannot falsely create apparent hallucinations. Failed selected images with complete labels lower recall. The evaluator uses one-to-one matches; fuzzy similarity is never silently treated as exact correctness or drug equivalence.

Extra/unmatched names and high-confidence unmatched names are review flags, not automatically proven hallucinations. Confirmed-hallucination counts remain unavailable until human adjudication. Subjective confidence cannot certify safe extraction. Empty denominators are null, not artificial 0% or 100% success.

## Files and how to continue

Open `README.md` for exact commands. Install dependencies, copy `.env.example` to `.env`, set your key, and download/import the selected ZIP. Run `inspect`, independently annotate a stratified subset and run `run --limit 10`. Use `--all` for the full collection and `python -m src.compare --limit 10` for a paired original/contrast/grayscale experiment. Inspect CSV review rows alongside originals; do not approve uncertain names automatically.

The latest `outputs/` exports are intentionally empty templates with `execution_status: not_run`. They are not cached demo results. New successful runs will replace these latest exports while retaining historical runs in separate timestamped folders.

## Remaining limitations and next improvements

Dataset downloading and real API execution must still be verified on the user's machine. Difficulty proportions and source annotation layout were not inspected. Credentials, provider access, quota and model availability may vary. Tests validate code behaviour, not clinical validity. Ground truth still needs independent human annotation; source language and brand distributions may differ from Indian prescriptions. Confidence calibration, perceptual deduplication, adjudication ingestion, cost estimates and statistically justified uncertainty intervals are future work. Repeated identical drug names with different instructions require review of automatic alignment. The initial request timeout is per attempt, not a whole-batch deadline.

First research priority: acquire and audit the selected data, freeze a held-out challenge set, run the live experiment, and investigate confidently wrong names before changing the prompt. Keep development and test sets separate. This is not a clinical decision-making tool.
