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
├── run_baseline.py        # Development-case baseline runs; dry-run by default
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
| data/processed/evaluation_inputs.jsonl | Fixed evaluation set of 40 development records: the 10 baseline cases, then up to 3 per disease site (2 for Pediatrics and Skin), favouring cases that need cumulative-dose reasoning, any prior RT, then boosts, with one case per new patient; same three fields |
| data/processed/evaluation_set.json | Case IDs and workbook labels (site, setting, prior RT, encounter type) of the evaluation set, for reports; no narratives, plans or patient IDs |
| outputs/data_summary.json | Counts, validation status, sample seed and size, timestamp, source filename and SHA-256 hash; no narratives or reference plans |

JSONL contains one JSON object per line. search_text combines only vignette
and question; the attending plan is separate payload for future example
retrieval. These files are plain local exports, not an embedding index.
Baseline sampling is reproducible for identical eligible records, seed and
size, independent of workbook row order; it is not stratified by disease site.
If validation fails, no new exports are written. Previously generated files
are not deleted and must not be treated as current after a failed run.

No embeddings, AWS requests, or LLM calls are performed by the loader.

## Baseline runner

Use the project virtual environment. The baseline engine needs boto3, faiss-cpu,
sentence-transformers, python-dotenv and openpyxl. The loader-only installation
above does not install the complete baseline environment.

```bash
source .venv/bin/activate
python -m pip install boto3 faiss-cpu sentence-transformers python-dotenv openpyxl
python patient_cases/run_baseline.py
```

The default is a dry run: it validates all baseline inputs against eligible
development rows in the current workbook and selects one case, without
importing the engine, contacting AWS, or printing narratives. Extra fields
(including reference plans), non-development IDs, and modified input text
are rejected. Run the loader again after changing the source workbook.

After setting up your local ignored .env and AWS SSO login, explicitly request
one real case:

```bash
python patient_cases/run_baseline.py --execute --limit 1
```

This downloads the reranker model if needed, reads existing AWS evidence,
creates query embeddings, and makes paid LLM calls in the configured account.
It does not retrieve institutional examples or change the existing RAG code.
The separate LLM judge is off by default; self-critique remains enabled as
configured by the existing engine. Add --judge only if its extra model calls
and calibration requirements are intended. Temperature is set to zero; this
does not guarantee identical model responses on repeated runs. Every engine LLM
call uses Llama 3.1 70B (128k context) by default for evaluation runs; pass
--model to choose another. The web app keeps the config.py default.

Each run has its own ignored folder, outputs/<name>_answers/ when --name is
given (for example --name stage1), otherwise outputs/baseline_<timestamp>_<id>/, with
manifest.json, startup.log, and results.jsonl. Results are appended after
each case and include answer, citations, retrieved passages/task plan, logs,
timing and errors. These files contain clinical text and may contain expiring
source URLs; keep them local or in approved storage. A startup failure is
saved separately. Metadata records the code commit, dirty-tree status,
configuration and input hashes; the live evidence index is not frozen.

Queries exceeding the existing input limit are skipped, never truncated.
Blocked responses, missing evidence/citations and incomplete retrieval are
flagged. Inspect logs even when response generation succeeds: the existing
engine may fall back after reranker or retrieval errors. A generated response
is not a clinical accuracy assessment. No automatic retries or resume are
provided; rerunning creates a new folder and can incur charges again.

After reviewing the first case, use --execute --limit 10 for the full
development subset. Locked-test evaluation is outside this runner's scope.

## Build ladder

Every stage of the build ladder answers the same 40-case evaluation set and is
scored by the frozen judge:

```bash
python patient_cases/run_baseline.py --inputs patient_cases/data/processed/evaluation_inputs.jsonl --limit 40 --name stage1 --execute
python patient_cases/judge_baseline.py patient_cases/outputs/stage1_answers --execute
python patient_cases/ladder_report.py add 1 patient_cases/outputs/judge_<id>
```

`add` refuses a run that did not answer the current evaluation set in full, or
whose judge (model, prompts, code, workbook) or answer-writing model settings
differ from stage 0. Stage 0 takes two runs, which measure run-to-run noise;
later stages take one each, in order. It renames judge_<id> after its answers
folder (stage1_answers is scored in stage1_scores) and writes the ignored
outputs/ladder.json registry and outputs/ladder_report.html, a self-contained
page of scores and case IDs only. Use `ladder_report.py build` to regenerate
the page without registering a run.

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

## Past-case index

Development cases eligible as worked examples form a local index of earlier
cases, kept apart from the evidence index:

```bash
python patient_cases/build_past_case_index.py --execute
python patient_cases/check_past_cases.py
```

The build embeds the vignette and clinical question with Titan v2 and writes
the ignored data/processed/past_case_index/ folder: vectors, records carrying
the attending plan as payload, and a manifest. It embeds only new or reworded
cases; label corrections update records without new embeddings. Locked-test
cases are refused.

Selection (core/past_cases.py) keeps at most two cases from the selected
disease site: same setting group and prior-RT group first, then same setting
group, then site only. It never uses another site, the case itself or another
encounter of the same patient. Encounter type reorders the ten most similar
cases in each tier, and an optional similarity floor drops weak examples.

The check runs every indexed case as a query with its own workbook labels,
counts rule violations, and writes outputs/past_cases_check/review.html: a
seeded sample of cases outside the evaluation set beside their chosen
examples. The page holds clinical text; keep it local. --min-similarity writes
to its own folder.

### Selection quality

```bash
python patient_cases/check_past_case_quality.py
python patient_cases/check_past_case_quality.py --split evaluation
```

Selection never sees the attending plan, so the plan grades it: does a chosen
example share the query's technique, or one of the dose regimens (total dose
and number of fractions) parsed from its plan? Four ways of choosing are
compared: same site or all labels, each in random or similarity order, with
95% intervals that resample patients rather than encounters. The development
run searches only cases outside the evaluation set and also reports match rate
by similarity and a sweep of similarity floors; the floor in core/past_cases.py
(MIN_SIMILARITY) was set from it. The evaluation run reports on the 40
evaluation cases against the whole index. Outputs hold metrics only.

### Finding similar cases

```bash
python patient_cases/find_similar_cases.py --case-id BR-003
python patient_cases/find_similar_cases.py --input path/to/new_case.json
```

--case-id uses an indexed case's stored vector and labels and excludes that
case and its patient; it makes no model call. --input reads a JSON file with
"vignette", "clinical_question" and any of the four labels (missing means Not
specified) and embeds it with Titan v2; it refuses to run if the configured
embedding model differs from the one the index was built with. Both write a
local page to outputs/similar_cases/ with up to two examples, each labelled as
an institutional example and not evidence. Keep case files and pages local.
