# Patient case extension

This folder holds the capstone additions for institutional worked examples and
patient-case evaluation alongside the existing RAG system.

## Folder layout

```text
patient_cases/
├── README.md
├── data/                 # Local workbook and inputs; contents ignored by Git
│   └── README.md
├── load_scenarios.py      # Workbook loading and validation
├── run_baseline.py        # Placeholder for development-case baseline runs
└── outputs/              # Local generated results; contents ignored by Git
```

## Local setup

Place the team-provided workbook at:

```text
patient_cases/data/RadOnc-Scenario-Bank-v2-Combined-Excel.xlsx
```

Each teammate supplies their own local copy from the team's approved source.
The workbook and generated results are not included in Git. Keep credentials
and original chart-rounds source files out of this folder and repository.

## Run the loader

The loader requires Python 3.9+ and openpyxl (already in requirements.txt).
For loader-only work, install just this dependency in your Python environment:

```bash
python -m pip install 'openpyxl>=3.1.0'
python patient_cases/load_scenarios.py
```

Run these commands from the repository root. Workbook and output paths are
resolved relative to the script, not the terminal's current directory.
You may also supply a workbook explicitly:

```bash
python patient_cases/load_scenarios.py --workbook /path/to/workbook.xlsx --baseline-size 10 --seed 42
```

The loader reads only Scenario Bank. It checks required headers and values,
unique Case IDs, valid split/eligibility/use labels, and no patient overlap
between Development and Locked test. It does not invent missing clinical
values, change split assignments, or export locked-test rows. It rejects
formula cells in scenario records. The expected 406/178 counts are reported,
not hardcoded as requirements, so a future approved workbook can be used.

On success it creates or replaces:

| Local file | Contents |
| --- | --- |
| data/processed/development_examples.jsonl | Eligible development records only, with vignette, question, search_text, attending_plan, study IDs, label, and remaining original columns under metadata |
| data/processed/baseline_inputs.jsonl | Seeded sample of 10 development records by default; only case_id, vignette, clinical_question |
| outputs/data_summary.json | Counts, validation status, sample seed and size, timestamp, source filename and SHA-256 hash; no narratives or reference plans |

JSONL contains one JSON object per line. search_text combines only vignette
and question; the attending plan is separate payload for future example
retrieval. These files are plain local exports, not an embedding index.
Baseline sampling is reproducible for identical eligible records, seed and
size, independent of workbook row order; it is not stratified by disease site.
If validation fails, no new exports are written. Previously generated files
are not deleted and must not be treated as current after a failed run.

run_baseline.py remains a placeholder. No embeddings, AWS requests, or LLM
calls are performed by the loader.

Keep locked-test cases out of development retrieval indexes and prompt tuning.
For baseline queries, pass only the vignette and clinical question; keep the
attending plan separate for review. Later, development attending plans may be
used as worked-example payloads in a separate index, following the workbook's
integration instructions. Institutional examples are not citable evidence.

When running a development query against an example index later, exclude that
case and other encounters belonging to the same patient. Patient ID is for
this grouping check, not a clinical similarity feature or model input. Other
metadata also needs review before model use: for example, Technique may
describe the delivered treatment rather than information known at decision time.
