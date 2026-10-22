"""Find similar past cases for one case and write a local page.

--case-id queries with an indexed case's stored vector and workbook labels, excluding the case and its patient, so
it makes no model call. --input reads a new case from a JSON file with "vignette", "clinical_question" and any of
"disease_site", "setting_group", "prior_rt", "encounter_type" (drop-down values; missing means Not specified), and
embeds it with Titan v2 (one call). The terminal shows case IDs and similarities only; the page holds clinical text
and stays in the ignored outputs folder.
"""

import argparse
from html import escape
import json
from pathlib import Path
import re
from string import Template
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from core.case_labels import NOT_SPECIFIED, CaseLabels, prior_rt_group
from core.past_cases import LABEL, MIN_SIMILARITY, NumpyPastCaseIndex, search_text, select_examples
from build_past_case_index import INDEX_FOLDER
from check_past_case_quality import regimens

OUTPUT = ROOT / "outputs" / "similar_cases"
LABEL_FIELDS = ("disease_site", "setting_group", "prior_rt", "encounter_type")


def from_index(index, case_id):
    """An indexed case as the query: its stored vector and workbook labels."""
    position = next((i for i, r in enumerate(index.records) if r["case_id"] == case_id), None)
    if position is None:
        raise ValueError(f"{case_id} is not in the past-case index.")
    record = index.records[position]
    query = {"case_id": case_id, "patient_id": record["patient_id"],
             **{k: record[k] for k in ("vignette", "clinical_question", *LABEL_FIELDS)}}
    return query, index.vectors[position]


def from_file(path, embed):
    """A new case from a JSON file, embedded by the same text as the indexed cases."""
    case = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("vignette", "clinical_question") if not str(case.get(k, "")).strip()]
    if missing:
        raise ValueError(f"The case file needs {' and '.join(missing)}.")
    query = {"case_id": None, "patient_id": None, "vignette": case["vignette"],
             "clinical_question": case["clinical_question"],
             **{field: case.get(field, NOT_SPECIFIED) for field in LABEL_FIELDS}}
    return query, embed(search_text(query["vignette"], query["clinical_question"]))


def find(index, query, vector, min_similarity=MIN_SIMILARITY):
    """Up to two examples by the selection rule; the query's own case and patient are never chosen."""
    labels = CaseLabels(**{field: query[field] for field in LABEL_FIELDS})
    return select_examples(index, vector, labels, query["case_id"], query["patient_id"],
                           min_similarity=min_similarity)


def shared_labels(query, record):
    """Which of the query's labels an example shares; Not specified labels are not compared."""
    pairs = (("setting group", query["setting_group"], record["setting_group"]),
             ("prior RT", prior_rt_group(query["prior_rt"]), record["prior_rt_group"]),
             ("encounter type", query["encounter_type"], record["encounter_type"]))
    return [(name, mine == theirs) for name, mine, theirs in pairs if mine != NOT_SPECIFIED]


def regimen_text(plan):
    found = sorted(regimens(plan))
    return ", ".join(f"{dose:g} Gy / {count} fx" for dose, count in found) or "none stated with a fraction count"


def why_none(query, min_similarity):
    if query["disease_site"] == NOT_SPECIFIED:
        return "No disease site was selected, so no past cases are shown: examples never come from another site."
    if min_similarity is not None:
        return f"No past case from this disease site reaches the similarity floor ({min_similarity:.2f})."
    return "No other patient's case from this disease site is indexed."


def page(query, hits, min_similarity):
    """The query beside its examples; every example is labelled as institutional practice, not evidence."""
    labels = " · ".join(escape(query[field]) for field in LABEL_FIELDS)
    title = escape(query["case_id"] or "New case")
    cards = []
    for letter, hit in zip("AB", hits):
        record = hit.record
        shared = " · ".join(f"{'✓' if same else '✗'} {name}" for name, same in shared_labels(query, record))
        # 💡 Examples are lettered, never numbered, so they cannot be mistaken for citations like [1]
        cards.append(f"""<article class="card">
<p class="banner">{escape(LABEL)}</p>
<h2>Example {letter} <span class="meta">{escape(record["case_id"])} · similarity {hit.similarity:.2f}</span></h2>
<p class="meta">{shared}</p>
<dl><dt>Setting</dt><dd>{escape(record["setting"])}</dd><dt>Encounter</dt><dd>{escape(record["encounter_type"])}</dd>
<dt>Technique</dt><dd>{escape(record["technique"])}</dd><dt>Regimen</dt><dd>{escape(regimen_text(record["attending_plan"]))}</dd></dl>
<h3>Attending plan</h3><p>{escape(record["attending_plan"])}</p>
<details><summary>Case description and question</summary><p>{escape(record["vignette"])}</p>
<p><b>Question.</b> {escape(record["clinical_question"])}</p></details>
</article>""")
    examples = "".join(cards) or f'<p class="empty">{escape(why_none(query, min_similarity))}</p>'
    floor = "off" if min_similarity is None else f"{min_similarity:.2f}"
    card = f"""<article class="card query">
<h2>{title}</h2><p class="meta">{labels}</p><p>{escape(query["vignette"])}</p>
<p><b>Question.</b> {escape(query["clinical_question"])}</p></article>"""
    # 💡 One pass: chained .replace() would rescan inserted case text, so a "$examples" in a vignette would be replaced
    return PAGE.substitute(title=title, floor=floor, query=card, examples=examples)


PAGE = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Similar past cases</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
:root { --bg: #f7f7f8; --card: #fff; --text: #1c1c1e; --muted: #6b6b70; --line: #e2e2e4; --warn: #8a5a00; --warnbg: #fff4dc; }
@media (prefers-color-scheme: dark) { :root { --bg: #161618; --card: #232326; --text: #ececee; --muted: #a0a0a6;
  --line: #3a3a3e; --warn: #ffd27a; --warnbg: #3a2f14; } }
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: var(--bg); color: var(--text); }
header { padding: 14px 16px; border-bottom: 1px solid var(--line); }
main { max-width: 1300px; margin: 0 auto; padding: 16px; display: grid; grid-template-columns: 1fr 2fr; gap: 16px; }
.examples { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-content: start; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px;
  font-size: 14px; line-height: 1.5; }
.query { border-color: var(--text); align-self: start; }
h2 { margin: 0 0 4px; font-size: 16px; } h3 { margin: 12px 0 2px; font-size: 14px; }
.meta { color: var(--muted); font-size: 12px; font-weight: 400; }
.banner { margin: 0 0 8px; padding: 4px 8px; border-radius: 6px; background: var(--warnbg); color: var(--warn);
  font-size: 12px; font-weight: 600; }
dl { display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; margin: 8px 0 0; font-size: 13px; }
dt { color: var(--muted); } dd { margin: 0; }
.empty { color: var(--muted); }
@media (max-width: 900px) { main, .examples { grid-template-columns: 1fr; } }
</style></head><body>
<header><strong>Similar past cases</strong> <span class="meta">for $title · same disease site only · similarity
floor $floor · examples show institutional practice and are not evidence</span></header>
<main>$query<section class="examples">$examples</section></main></body></html>""")


def bedrock_embedder(index_folder):
    """Titan v2 embedding of one text; refuses a model other than the one the index was built with."""
    from config import Config, create_boto3_session
    config = Config()
    built_with = json.loads((Path(index_folder) / "manifest.json").read_text(encoding="utf-8"))["embedding_model"]
    # ⚠️ Vectors from different embedding models live in different spaces: their cosine similarity means nothing
    if built_with != config.embedding.model_id:
        raise ValueError(f"The index was built with {built_with}, but the configured model is "
                         f"{config.embedding.model_id}; rebuild the index first.")
    client = create_boto3_session(config.aws).client("bedrock-runtime", region_name=config.aws.region)

    def embed(text):
        response = client.invoke_model(modelId=config.embedding.model_id, body=json.dumps({"inputText": text}))
        return np.array(json.loads(response["body"].read())["embedding"], dtype=np.float32)
    return embed


def main(argv=None, embedder=bedrock_embedder):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--case-id", help="An indexed case, such as an evaluation case; no model call.")
    source.add_argument("--input", help="JSON file with a new case; one Titan embedding call.")
    parser.add_argument("--index", default=str(INDEX_FOLDER), help=argparse.SUPPRESS)
    parser.add_argument("--output", default=str(OUTPUT), help=argparse.SUPPRESS)
    parser.add_argument("--no-floor", action="store_true", help="Show the two closest cases even below the floor.")
    args = parser.parse_args(argv)
    try:
        index = NumpyPastCaseIndex.load(args.index)
        if args.case_id:
            query, vector = from_index(index, args.case_id)
            name = args.case_id
        else:
            query, vector = from_file(args.input, embedder(args.index))
            name = Path(args.input).stem
        min_similarity = None if args.no_floor else MIN_SIMILARITY
        hits = find(index, query, vector, min_similarity)
        folder = Path(args.output)
        folder.mkdir(parents=True, exist_ok=True)
        # ⚠️ The file name comes from user input: keep it to safe characters so it cannot leave the folder
        path = folder / f"{re.sub(r'[^A-Za-z0-9_-]', '_', name)}.html"
        path.write_text(page(query, hits, min_similarity), encoding="utf-8")
        print(json.dumps({"query": query["case_id"] or "new case", "index_version": index.version(),
                          "min_similarity": min_similarity,
                          "examples": [{"case_id": h.record["case_id"], "similarity": round(h.similarity, 3)}
                                       for h in hits],
                          "page": str(path)}, indent=2))
        return 0
    except Exception as exc:
        print(f"Search failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
