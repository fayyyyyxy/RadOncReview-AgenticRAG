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
It uses evidence only by default. `--institutional-examples` additionally supplies
up to two historical institutional examples to the writer, as described below.
The separate LLM judge is off by default; self-critique remains enabled as
configured by the existing engine. Add --judge only if its extra model calls
and calibration requirements are intended. Temperature is set to zero; this
does not guarantee identical model responses on repeated runs. Every engine LLM
call uses the web app's model from config.py (Llama 3.3 70B) by default; pass
--model to choose another. Llama, Claude and Titan models keep their own
request formats; any other Bedrock model (for example us.openai.gpt-5.6-terra)
goes through the Converse API, without temperature and top_p if it rejects
them, with a token limit of 8192 (converse_max_tokens in config.py) instead of
Llama's 2048, since a reasoning model's thinking counts toward the limit. A
reply cut off at the token limit is noted in the case's engine log.

Each run has its own ignored folder, outputs/<name>_answers/ when --name is
given (for example --name site-filter), otherwise outputs/baseline_<timestamp>_<id>/, with
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

```bash
python patient_cases/run_baseline.py --execute --no-citation-check
```

After the answer is written, a citation check (core/citation_check.py) runs in
the web app and in baseline runs; --no-citation-check skips it, as runs made
before it did. One more LLM call per answer asks, for each sentence and each
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

## Institutional examples in generated answers

The local `codex/institutional-examples` implementation builds on
`grounded-citations` (base commit `1994a8c`). It preserves that branch's
citation checker, model settings, and self-critique/rewrite behavior. The new
feature is off by default and is available through the development baseline
runner and the Python engine; the web/API request schema is unchanged.

Use the existing index format described below. Generated `.npy`, `.jsonl`
and manifest files are ignored by Git; fetching a branch does not copy them.
An existing index can be supplied with `--past-case-index /path/to/past_case_index`.
Its parent folder must contain its original `development_examples.jsonl`.

```bash
# No AWS calls; validates the development inputs, not the index files.
python patient_cases/run_baseline.py --institutional-examples
python patient_cases/verify_institutional_examples.py

# Paid answer-generation calls, after AWS login and index preparation.
python patient_cases/run_baseline.py --execute --limit 1 --name examples-off
python patient_cases/run_baseline.py --execute --limit 1 --name examples-on --institutional-examples
```

Both runs use the same default input case. Use new names for deliberate
repeats. Keep writer model, evidence settings and evaluation cohort fixed for
comparisons; the live evidence sources are still not a frozen snapshot.

The adapter rejects an index with a different embedding model/dimension,
changed payloads or manifest, a stale development export, or a corpus that
differs from the current workbook's development cases. Only eligible
Development records are permitted. The exact query case and all encounters
of its patient are excluded. The existing matching tiers and 0.50 similarity
floor select at most two examples. A known development query reuses its
stored vector; an unindexed query needs an embedding.

Selected examples are labelled historical practice, are not numbered as
evidence, and cannot enter the citation-passage list. The writer is instructed
to support medical recommendations with numbered clinical evidence and to
request missing patient information rather than copy it from an example.
When enabled, an empty evidence retrieval cannot be replaced by examples.
These instructions are constraints on the prompt, not proof of clinical accuracy.

Each saved response now records `institutional_examples` (index provenance,
selected case IDs, similarities and payloads) and `synthesis_requests` (exact
writer prompts, including rewrites). Patient IDs and the query case's own
attending plan are not supplied to the writer. These local outputs contain
clinical text. Eason's citation checker still receives clinical passages only;
the judge's evidence context also remains separate from examples.

For direct Python use, set `config.features.institutional_examples=True` and
call `engine.query(context.query(), labels, case_context=context)`, where
`context` is `core.institutional_examples.CaseContext` with vignette,
clinical_question, case_id and patient_id. Provide an expected development
export SHA-256 through `config.features.past_case_source_sha256` when a
workbook corpus is known; the baseline runner does this automatically.

## Flexible versioned LLM planning (opt-in)

`--versioned-planning` separates **patient facts**, **information needs** and
**search language**. The current schema is `source_anchored_llm_v2`:

- A requirement has a normalized `need`, `kind` (`explicit` or `inferred`),
  and exact provenance quotes in `anchors`. Anchors may quote the question
  or vignette; follow-ups may also quote supplied critique feedback. Inferred
  research needs require an `inference_reason`; they are not patient facts.
- Each topic supplies `search_terms`, which may use clinical synonyms,
  abbreviations and standard terminology absent from the original wording.
  For example, a reference to previous radiation may motivate a research
  concept such as “re-irradiation.” No fixed topic vocabulary is imposed.
- Each topic selects `context_ids` from a catalog of original sentences/lines,
  with a `context_reason`. The application copies those sentences verbatim
  into that search, preserving their wording, including negations and temporal
  qualifiers. It does not append the complete case automatically. The prompt
  asks for current-condition context and relevant prior treatment while
  omitting unrelated history. General conceptual searches may select no context.

Both existing clinical backends receive each constructed query. The full
original case remains on `TaskPlan.original_query` and goes to the writer and
critic. All context units, including omitted sentences, remain in planning
metadata for review. A missing or incorrectly selected context sentence can
still harm retrieval; source copying alone does not establish correct routing.

Initial plans allow up to six topics by default, and follow-ups up to three
(`config.features.max_planning_topics` / `max_followup_topics`). The prompt asks
for separate searches for independently answerable needs. Inseparable needs or
needs grouped to fit the budget may share a search with a recorded
`grouping_reason`; the application no longer
forces one search per requirement or rejects provenance quotes based on comma
counts. Every declared need must map to exactly one topic. Oversized topic
lists, unknown context IDs, invented provenance quotes and duplicate searches
are rejected rather than silently truncated. Unsupported standalone numbers
in generated search terms are rejected against that topic's original context
and non-feedback anchors; this catches some invented doses but is not a
semantic clinical validator. Digits inside standard terms such as EQD2 are
not treated as a new patient dose.

Validation checks structure, provenance, verbatim context selection and
**declared** coverage. It does not establish that all needs were identified,
that a synonym or inferred need is medically appropriate, that selected context
is complete, or that condition routing is correct. The prompt forbids invented
patient facts and asks that citation requests apply across topics; those semantic
instructions are not mechanically guaranteed. Inspect the recorded plan and
the actual passages. The earlier relevance-audit problems remain open.

```bash
# Validate inputs without AWS/model calls.
python patient_cases/run_baseline.py --versioned-planning

# Paid LLM planning plus three fresh AWS retrieval passes; no answer writing.
python patient_cases/check_retrieval_consistency.py --case-id BR-003 --mode versioned --repeats 3 --execute

# Later answer generation, including the normal critique/rewrite flow.
python patient_cases/run_baseline.py --versioned-planning --execute --limit 1 --name versioned-pilot
```

The first valid plan is saved locally; repeated requests reuse it and create
fresh pending search tasks. They do not reuse passage results. By default,
plan records and locks live in ignored `patient_cases/data/processed/search_plans/`.
Both runners accept `--plan-cache path/to/folder` for an isolated experiment.
Plans contain clinical input and feedback: keep them local and ignored.

Cache keys include exact patient input, analysis, model/sampling settings,
topic budget, retrieval settings, prompt hash, planner implementation hash
and LLM-client implementation hash. Follow-up keys additionally include the
parent payload hash and exact feedback. Rewording changes the request key, but
verified paraphrase aliases can share one saved initial plan. Changed model,
prompt, policy or tracked implementation versions require a compatible new
plan. Cached identities and payloads are checked on read; corruption raises
an error. POSIX file locks and atomic writes prevent concurrent identical misses
from publishing different plans. New files have owner-only permissions.

### Verified paraphrase reuse

The versioned planner accepts verified paraphrases of initial requests. There
is one cache behavior, with no exact-input-only mode. Repeating a saved wording
needs no model call. For new wording, the planner considers up to three
compatible saved plans from the 128 most recently modified cache records.
Model/settings, planning budgets, prompts and relevant implementation/policy
hashes must match. Conservative checks reject changed numbers, quantities,
laterality, sex, explicit stage markers, negation or uncertainty before model
verification; these checks can reject some valid paraphrases too.

Lexical similarity only ranks candidate plans. A separate structured LLM call
must affirm that patient facts, uncertainty and information needs are unchanged
in both directions. It must map every original and current context unit and
re-anchor every saved requirement to exact current-source quotes. The application
retains the saved search concepts and topic groupings, copies only current
verbatim context into the rebuilt searches, and revalidates the resulting plan.
Uncertain, invalid or failed verification falls back to normal fresh planning.
An LLM equivalence decision is not proof of clinical equivalence; matching
facts or requested needs incorrectly remains a risk requiring evaluation.

The first accepted paraphrase has `cache_status=paraphrase_hit` and uses one
verification call instead of generating a new search plan. Its verified wording
is saved as an alias inside the original record, not another full plan file.
Repeating that wording gives `paraphrase_alias_hit` without any LLM call.
At most 16 aliases are retained per plan (`paraphrase_cache_aliases`); old
aliases are evicted in insertion order. New patient cases still create new
records; global expiration/disk-size eviction is not implemented here.
Request locks use 256 shared slots to avoid a new lock file per wording.
Alias updates are locked and atomic, and alias identities, verification data
and adapted-payload hashes are checked on reuse. Concurrent requests for the
same wording perform one verification.

`TaskPlan.planning` records `cache_source_key`, `paraphrase_check_calls`,
`paraphrase_verification` and a miss reason. `llm_calls` includes equivalence
checks as well as plan-generation attempts. Exact cache keys still include
the exact current input; aliases link verified variants to the saved plan.
Follow-up plans/feedback continue to use exact matching, preserving their
parent-specific behavior. The legacy planner does not use this mechanism.

This reuses the search structure, not passages or answers. Verbatim patient
context can differ after rewording, so identical retrieved passages across
paraphrases are not guaranteed. Full current input remains available to the
writer. Validate paraphrases and clinical near misses before production use.

### Cache provenance and fallbacks

The v2 schema/version and changed implementation invalidate earlier v1 plan
keys automatically. Old files can remain local but are not used by v2 runs.
Cache records preserve provenance, explicit/inferred needs, search terms,
selected context IDs and their reasons, topic mappings, rationale,
generation timestamp, model settings, system prompt and planning attempts.
`TaskPlan.planning` records cache status, hashes, call count and validation
scope; `AgentResponse.routing.planning_events` preserves both initial and
follow-up events. A failed initial generation gets one validation-repair
attempt, then an explicit **uncached full-input fallback**. Failed follow-up
planning schedules no additional searches and records its failure; existing
evidence and writer feedback remain available. A rewrite-only critique can
also explicitly request no additional search. Exact duplicate follow-up
queries are skipped. The consistency check does not count a planning fallback
as a healthy validated-plan run.

The original planner remains the default; versioned planning is opt-in.
Versioned planning uses rule-based initial
analysis so an uncached generative analysis step cannot change the plan key
on every request. Multiple initial topics use the existing per-topic
retrieval/reranking path, retaining its three final passages per topic.

Plan reuse establishes repeatability of the recorded plan; it does not show
that independent LLM generations are deterministic. Different feedback may
legitimately create different follow-up searches. Live source changes can
also change returned passages. The earlier local 14-case audit evaluated a
discarded vocabulary-based prototype that is not included in this branch.
It is not evidence of the versioned planner's relevance or clinical accuracy.

## Retrieval repeatability and relevance checks

Clinical evidence is retrieved by `core/retriever.py` in both the examples-off
and examples-on paths. Hospital examples use `core/past_cases.py`, connected to
the writer through `core/institutional_examples.py` after evidence retrieval.

The original complex-question planner uses a generative model to rewrite
search topics. Even with temperature zero, repeated requests can produce
different queries and retrieve different passages. The versioned planner
described above reuses validated search plans and can create new follow-up
plans for explicit critique gaps. The retrieval checker defaults to versioned
planning; use `--mode llm` to measure the original planner instead.

Passage identity now includes the complete text hash, preventing distinct
trial passages sharing the first 100 characters from being merged. Reranking
uses a canonical candidate order and a stable tie-breaker. These identity
and ordering fixes apply in both planning modes.

Check fresh retrieval without writing or judging an answer:

```bash
# Validate a Development input without AWS calls.
python patient_cases/check_retrieval_consistency.py --case-id BR-003

# Paid planning on a cache miss plus repeated live AWS retrieval; no generated answers.
python patient_cases/check_retrieval_consistency.py --case-id BR-003 --mode versioned --repeats 3 --execute

# The original generative planner can be measured separately.
python patient_cases/check_retrieval_consistency.py --case-id BR-003 --mode llm --execute

# Interleave repeats across a supplied, workbook-validated Development cohort.
python patient_cases/check_retrieval_consistency.py --inputs path/to/inputs.jsonl --all-inputs --repeats 3 --execute

# Separately evaluate the existing disease-site restriction on the same cohort.
python patient_cases/check_retrieval_consistency.py --inputs path/to/inputs.jsonl --all-inputs --repeats 2 --site-filter --execute
```

The check records semantic query plans, raw candidate pools, ordered final
passage identities, logs, configuration, input and code hashes. It requires
nonempty completed active tasks, a loaded reranker, and no logged retrieval/reranking
errors before reporting a healthy comparison. A run passes only if all three
hashes match across repetitions. Reports contain clinical text and are kept
in ignored local `patient_cases/outputs/` folders.

Every repetition searches both clinical backends again; saved passages are
never replayed. The FAISS index loads once per process, as in the application.
The live DynamoDB and Knowledge Base contents are not frozen or version-pinned,
so source updates may legitimately change results. Matching results establish
repeatability for the measured case and session, not medical relevance or
accuracy. Review multiple Development cases before enabling this mode by
default or using it for an examples-off/on comparison. To use it in a later
answer-generation run, add `--versioned-planning` to `run_baseline.py`; keep that
setting the same in both arms.

For a completed retrieval run, build a local review packet or request a paid
automated relevance screen:

```bash
python patient_cases/review_retrieval_relevance.py patient_cases/outputs/retrieval-consistency-<run>
python patient_cases/review_retrieval_relevance.py patient_cases/outputs/retrieval-consistency-<run> --execute
```

The packet contains each patient question and the union of selected passages
across repetitions, with appearance/rank information. The optional Bedrock
screen labels each passage direct, partial or irrelevant and assesses requested
aspect coverage. It must provide a source-matching quote for direct/partial
labels; missing passages, unknown IDs and invented supporting quotes reject
the response. One formatting/validation repair is allowed, with both requests
and responses saved. Successful screens can be resumed without repeating
their model calls; changed run data, rubric or settings cannot be mixed.

This is automated question/passage alignment, not medical correctness or an
independent gold standard: the screen uses the project's Llama writer model.
Coverage labels also need human review. No attending plans, hospital examples
or locked-test cases are used. The review page, prompts and scores remain in
ignored local output folders. A different source category alone does not prove
irrelevance; general methods may transfer, while even a matching disease site
can have an unsuitable population or treatment setting.

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
