# Dataset selection — 17 September 2026

This is a source-based selection, not a claim of completed image inspection. The selected archive could not be downloaded in this environment.

| Candidate and primary source | Content and published size | Labels / difficulty / image variation | Access / licensing evidence | Decision |
|---|---|---|---|---|
| [Curated Bangladesh prescriptions v2](https://data.mendeley.com/datasets/k62rfd23kz/2), DOI 10.17632/k62rfd23kz.2 | 200 full prescription images; handwritten and printed; Bangla/English/mixed | Medicine-region boxes and counts, not name transcription; messy proportion and exact quality distribution unknown | Public Mendeley record, CC BY 4.0, de-identification described; local download returned 403 | **Primary choice**, subject to visual audit and manual handwriting stratification |
| [Doctor's Handwritten Prescription BD](https://data.mendeley.com/datasets/zjjtptvn6f), also [Kaggle](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset) | Publisher describes 4,680 word images, 78 medicine classes; standardized 128×128 crops | Independently reviewed word labels; lacks page layout, frequency, duration and full clinical clutter | Public record describes education/research-only use; check actual terms before acquisition or redistribution | Useful secondary name-only benchmark; cannot replace page-level evaluation |
| [Handwritten Medical Prescriptions Collection](https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset) | Prescription-image candidate surfaced by Kaggle; exact count not verified | Difficulty, label completeness and quality distribution not verified from available page | Detail page did not expose usable metadata to retrieval; license/provenance unresolved | Do not select based on “illegible” in the title alone |
| [Medical-Prescription-OCR repository](https://github.com/JonSnow1807/Medical-Prescription-OCR) and [linked HF dataset](https://huggingface.co/datasets/chinmays18/medical-prescription-dataset) | Author reports 1,000 synthetic PNGs | JSON annotations reported; synthetic variation cannot establish real messy handwriting performance | Repository is public and MIT-licensed; dataset licensing must be checked separately | Secondary synthetic integration baseline only |

## Why select the full-page collection?

The target is medicine extraction amid genuine prescription layout and handwriting, not classification of pre-cropped drug words. The primary collection offers a clearer provenance/license and a modest page-level workload. This choice does **not** establish that it contains enough extremely messy handwriting. Inspect before finalizing the research test set. If the hardest stratum is too small, add an independently sourced, consented and de-identified challenge set.

Original bounding boxes must remain untouched. Do not convert coordinates or medicine counts into fabricated transcriptions. The CSV annotation template preserves original text and aligned medicine fields, and separates complete, partial and unavailable annotations.

## Acquisition evidence

- Selected version is v2, not the older v1 count of 182.
- Attempted automatic archive download: HTTP 403; see `download_status.json`.
- Real images downloaded: **0**. Locally usable prescription images: **0**.
- Handwritten proportion, dimensions, corrupt-file count and duplicates in the source archive: **unknown**, not measured.
- Medicine-name / dosage ground truth locally available: **none**.
- The supplied project brief and PDF are specification documents, not prescription test images.
- No alternate restricted host or credential workaround was used.

The README contains an automatic retry command, exact ZIP import commands, optional credential-detecting Kaggle integration, and custom-image instructions. No source dataset has been repackaged or redistributed.

## Suggested first research pass

Inspect all acquired images, then tag handwriting and quality before inference. Select hard handwritten examples first, with a separate printed baseline. Annotate medicine names independently of Gemini and explicitly mark unresolved handwriting. Maintain a locked held-out set; tune the prompt on a separate development set. Compare the same images across preprocessing variants and report case-sensitive, normalized and fuzzy scores separately. Report source limitations, language/domain mismatch and uncalibrated confidence. Small convenience samples do not establish clinical reliability.
