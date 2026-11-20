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
The writer also gets up to two similar past cases from the local past-case
index as institutional examples, never cited; the case itself and its
patient's other encounters are excluded, and a missing index stops the run.
Pass --no-institutional-examples to write from evidence alone. The separate LLM judge is off by default; self-critique remains enabled as
configured by the existing engine. Add --judge only if its extra model calls
and calibration requirements are intended. Temperature is set to zero; this
does not guarantee identical model responses on repeated runs. Every engine LLM
call uses the web app's model from config.py (GPT-5.6 Terra, the model of the
final comparison) by default; pass --model to choose another. Llama, Claude and
Titan models keep their own request formats; any other Bedrock model, such as
the default, goes through the Converse API, without temperature and top_p if it rejects
them, with a token limit of 8192 (converse_max_tokens in config.py) instead of
Llama's 2048, since a reasoning model's thinking counts toward the limit. A
reply cut off at the token limit is noted in the case's engine log.

Each run has its own ignored folder, outputs/<name>_answers/ when --name is
given (for example --name p3-final), otherwise outputs/baseline_<timestamp>_<id>/, with
manifest.json, startup.log, and results.jsonl. Results are appended after
each case and include answer, citations, retrieved passages/task plan, logs,
timing and errors. These files contain clinical text and may contain expiring
source URLs; keep them local or in approved storage. A startup failure is
saved separately. Metadata records the code commit, dirty-tree status,
configuration and input hashes; the live evidence index is not frozen.

Queries exceeding the existing input limit are skipped, never truncated.
Blocked responses, missing evidence/citations and incomplete retrieval are
flagged; an answer written after incomplete retrieval is still judged, as one
left with no citations is. Inspect logs even when response generation succeeds: the existing
engine may fall back after reranker or retrieval errors. A generated response
is not a clinical accuracy assessment. No automatic retries or resume are
provided; rerunning creates a new folder and can incur charges again.

After reviewing the first case, use --execute --limit 10 for the full
development subset. Locked-test evaluation is outside this runner's scope.

### Citation check

After the answer is written, a citation check (core/citation_check.py) runs in
the web app and in baseline runs. One more LLM call per answer asks, for each sentence and each
passage it cites, whether the passage states what the sentence says, numbers
included. A rejected citation is removed and the sentence stays; sentences that
lost a citation are listed in the response metadata (citation_check), and the
web page underlines them and lists them under an "unsourced" badge. A note
written into the answer text instead lowered judged completeness and guideline
adherence. An answer left with no citations is still judged, with Citations
Verifiable set to N; skipping it would leave the least-grounded answers out of
the scores.
Brackets such as [4, 5] or [4], [5] are checked one number at a time; the
kept ones are rewritten as [4] [5], and a removed one takes the comma or "and"
before it along. A reference list after a "References:" line is not checked:
an entry stays while the answer still cites its number. An unparseable reply
is retried once with the error; if it fails again, the answer is left
unchanged and the error is recorded.
The answer as written is kept in the report (answer_before), so a change to
the check can be replayed on the same text instead of a new run.

### Replaying plans

```bash
python patient_cases/run_baseline.py --inputs patient_cases/data/processed/evaluation_inputs.jsonl \
    --limit 40 --execute --name heading-test --replay-plans gpt56terra-run3
```

The planner writes different search queries from run to run even at
temperature zero (33 of 40 cases between two Llama 3.3 runs), so about half
the passages differ and a small change is lost in that noise.
--replay-plans NAME skips the query analyzer and planner and searches with the
first-round plan saved in outputs/NAME_answers: its analysis and sub-task
queries, read from the saved task plan up to the size in the "PLAN: Created
N sub-tasks" log line, and the planner's reasoning from that same line (a
re-plan overwrites the saved one, and the next re-plan would read it). The
re-plan after self-critique still runs, since it answers the new draft. Any saved run can be the source, and a replayed run can
be replayed in turn. Each case's saved input text must match, or the runner
stops before any paid call; the dry run checks this too. The manifest records
the source run and the hash of its results. Compare runs that replay the same
source case by case, for example McNemar's test on hallucination.

### Self-critique and rewrite

The self-critique lists what the question asks for and the parts of a plan
(see Completeness template), and marks each addressed or not; saying that the
evidence does not cover an item does not count. One missing item, an unsafe
recommendation or low confidence sends the answer back once: the planner
re-plans from the feedback, the new passages are added to the first round's,
and the writer rewrites with the feedback and its draft. Every run keeps the
draft the critique judged (metadata.first_draft) next to the critique itself
(self_critique).

### Rerank query

Each topic's passages are ranked against the case's disease site and the topic
label, for example "Gastrointestinal boost dose", instead of the whole vignette
and the label (docs/rag_improvement_plan_v2.md, E2 option (b)). The
site comes from the case labels (the workbook here, the user's selection in the
web app); with no site selected the vignette is kept. A re-planned topic usually
has no label, so it ranks on the site alone. On run3's 1,014 sent passages the
on-site against off-site AUC is 0.75, against 0.54 with the vignette.

### Completeness template

The writer answers under the seven parts of the workbook's Completeness anchor (indication,
target volumes, dose and fractionation, technique, organ-at-risk constraints,
systemic therapy, follow-up). For a part no passage covers it gives the usual
practice, uncited and marked "(not in the provided evidence)"; for a case detail
the vignette leaves out it states its assumption and answers under it; for a
part that does not apply it writes "Not applicable" with a reason. A first
pilot that named missing parts instead lost completeness, as the judge scores a
named gap as missing. The critique checks those parts as well as what the
question asks, so a missing part sends the answer back for a re-plan and a
rewrite. Without the template, the
critique's checklists held follow-up in 0 of 80 and OAR constraints in 14 of
80, the two parts the judge most often found missing.

On the 40 cases, run twice against the same plans without it, completeness rose
from 3.48 to 4.30 (37 cases up, none down) with accuracy and adherence
unchanged, and verifiable citations from 81% to 90%. Statements the passages
do not hold rose from 3.0 to 5.4 per judging, most of them marked, and answers
grew about a fifth. One answer reached Harm 3 where the runs without it had
none: SKN-002 acted on a node-negative status the case does not give. The bar
set before the run allowed no such answer; the template was kept after that
result.

### Experiments not kept

The switches for experiments that were not kept were removed after the final
comparison. The tag pre-cleanup-2026-11-08 still has every one of them, and
docs/rag_improvement_plan_v2.md has their results: --site-filter (evidence
limited to the selected site), --repair-batch, --no-replan, --citation-flag-only and the
critique's named gap (Part 1), --critic-edits and --critic-model (E4), and
--no-citation-check, --no-trial-papers, --no-completeness-template and the
vignette as the rerank query. To rerun one of them, or the final, E4 and E1
arms, check out that tag in a worktree; the base arm ran at f315841.
Institutional examples were removed with them and later brought back, on by
default; the final, E4 and E1 arms ran without them.

Keep locked-test cases out of development retrieval indexes and prompt tuning.
For baseline queries, pass only the vignette and clinical question; keep the
case's own attending plan separate for review. Other development cases'
attending plans are worked-example payloads in a separate index, following the
workbook's integration instructions. Institutional examples are not citable evidence.

A development query run against the example index excludes that case and
other encounters belonging to the same patient. Patient ID is for
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
