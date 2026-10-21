"""Build ladder: register judge runs per stage and compare stages against stage-0 noise.

Reads judge scores only (numbers, labels and case IDs), never answers, reasons or patient text.
No network calls; the report is a local HTML file under outputs/.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REGISTRY = OUTPUTS / "ladder.json"
REPORT = OUTPUTS / "ladder_report.html"
EVALUATION_INPUTS = ROOT / "data" / "processed" / "evaluation_inputs.jsonl"
EVALUATION_SET = ROOT / "data" / "processed" / "evaluation_set.json"
# Stage -> (roadmap step, change); each stage adds one change to the previous one
STAGES = {
    0: ("④", "Today's system, run twice"),
    1: ("⑤", "Case tagging + evidence filtered by disease site"),
    2: ("⑥", "Reference-list passages dropped"),
    3: ("⑦", "Answer sections, rules (a)(e)(f), ≤ 8 passages, re-planning cap"),
    4: ("⑧", "Re-irradiation pathway (A′)"),
    5: ("⑨", "Past cases (B)"),
}
# (key, label, direction): direction +1 means higher is better
METRICS = (
    ("accuracy", "Accuracy", 1),
    ("completeness", "Completeness", 1),
    ("guideline_adherence", "Guideline adherence", 1),
    ("harm_2plus", "Harm ≥ 2", -1),
    ("hallucination", "Hallucination", -1),
    ("citations_verifiable", "Citations verifiable", 1),
    ("abstention_failure", "Abstention = N", -1),
)
# ⚠️ Unanswered cases never reach the judge, so score means alone would improve if a stage broke more cases
NO_ANSWER = ("no_answer", "No answer", -1)
ALL_METRICS = METRICS + (NO_ANSWER,)
SCALES = ("accuracy", "completeness", "guideline_adherence")
# Anything that changes how answers are graded; every stage must share these values
FROZEN = ("judge_model", "prompt_sha256", "judge_code_sha256", "workbook_sha256", "runs_per_case", "max_tokens")
# Answer-writing settings from each generation manifest's config.llm; a stage changes code, never the model
WRITER = ("model_id", "temperature", "max_tokens", "top_p")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_values(final):
    """The seven rubric dimensions as numbers; Harm and Abstention count failures, as the workbook does."""
    return {
        **{key: float(final[key]) for key in SCALES},
        "harm_2plus": float(final["harm"] >= 2),
        "hallucination": float(final["hallucination"] == "Y"),
        "citations_verifiable": float(final["citations_verifiable"] == "Y"),
        "abstention_failure": float(final["appropriate_abstention"] == "N"),
    }


def load_run(judge_dir):
    """Per-case values of one judge run; judge errors are left out, as in the judge's own summary."""
    judge_dir = Path(judge_dir)
    lines = (judge_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    manifest = read_json(judge_dir / "manifest.json")
    summary = judge_dir / "summary.json"
    return {
        "name": judge_dir.name,
        "values": {r["case_id"]: case_values(r["final"]) for r in rows if r["status"] == "scored"},
        # Every evaluation case: the judge skips those whose generation failed or was blocked
        "answers": {case: {"no_answer": float(case in manifest["skipped"])}
                    for case in [*manifest["case_ids"], *manifest["skipped"]]},
        "sites": {r["case_id"]: r["disease_site"] for r in rows},
        "errors": [r["case_id"] for r in rows if r["status"] != "scored"],
        "usage": read_json(summary).get("usage", {}) if summary.exists() else {},
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def stage_values(runs, field="values", metrics=METRICS):
    """Per-case values of a stage; stage 0's two runs are averaged case by case."""
    common = set.intersection(*(set(run[field]) for run in runs))
    return {
        case: {key: mean(run[field][case][key] for run in runs) for key, _, _ in metrics}
        for case in common
    }


def noise(run_a, run_b, metrics=METRICS):
    """Smallest stage change that is not noise: 2 standard errors of the paired stage-0 difference."""
    common = sorted(set(run_a) & set(run_b))
    n = len(common)
    if n < 2:
        raise ValueError("Stage 0 needs at least two cases scored in both runs.")
    thresholds = {}
    for key, _, _ in metrics:
        diffs = [run_a[case][key] - run_b[case][key] for case in common]
        centre = mean(diffs)
        sd = math.sqrt(sum((d - centre) ** 2 for d in diffs) / (n - 1))
        # 💡 Pairing by case removes case difficulty: only run-to-run wobble is left in the differences
        # ⚠️ The standard error assumes cases wobble independently; a whole run can shift too (2026-10-21: Accuracy
        # moved 0.37 against a 0.23 standard-error band), so the observed gap is a floor. So is one case's worth:
        # two identical runs give sd = 0, and one flipped case would then mean "stop".
        thresholds[key] = max(2 * sd / math.sqrt(n), abs(centre), 1 / n)
    return thresholds


def compare(previous, current, thresholds, metrics=METRICS):
    """Mean change per metric over the cases scored in both stages, judged against the noise."""
    common = sorted(set(previous) & set(current))
    if not common:
        raise ValueError("No case was scored in both stages, so they cannot be compared.")
    rows = {}
    for key, _, direction in metrics:
        before = mean(previous[case][key] for case in common)
        after = mean(current[case][key] for case in common)
        # Positive gain is an improvement whichever way the metric points
        gain = direction * (after - before)
        verdict = "worse" if gain < -thresholds[key] else "better" if gain > thresholds[key] else "within noise"
        rows[key] = {"before": before, "after": after, "delta": after - before, "verdict": verdict}
    return {"cases": len(common), "metrics": rows, "stop": any(r["verdict"] == "worse" for r in rows.values())}


def register(stage, judge_dir, registry_path=REGISTRY, inputs_path=EVALUATION_INPUTS):
    """Add one judge run to a stage after checking it scored the evaluation set with the frozen judge."""
    judge_dir = Path(judge_dir)
    if stage not in STAGES:
        raise ValueError(f"Stage must be one of {sorted(STAGES)}.")
    judge = read_json(judge_dir / "manifest.json")
    generation = read_json(judge_dir.parent / judge["baseline_run"] / "manifest.json")
    if generation["input_sha256"] != sha256(inputs_path):
        raise ValueError(f"{judge['baseline_run']} did not answer the current evaluation set.")
    expected = [json.loads(line)["case_id"] for line in Path(inputs_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if sorted(generation["case_ids"]) != sorted(expected):
        raise ValueError(f"{judge['baseline_run']} answered {len(generation['case_ids'])} of {len(expected)} cases.")

    writer = {key: generation["config"]["llm"][key] for key in WRITER}
    registry = read_json(registry_path) if Path(registry_path).exists() else {
        "evaluation_inputs_sha256": sha256(inputs_path),
        "frozen_judge": {key: judge[key] for key in FROZEN},
        "frozen_writer": writer,
        "runs": [],
    }
    if registry["evaluation_inputs_sha256"] != sha256(inputs_path):
        raise ValueError("The evaluation set changed since stage 0; every stage must score the same cases.")
    changed = [key for key in FROZEN if judge[key] != registry["frozen_judge"][key]]
    if changed:
        raise ValueError("Judge changed since stage 0 (" + ", ".join(changed) + "); re-score every stage instead.")
    changed = [key for key in WRITER if writer[key] != registry["frozen_writer"][key]]
    if changed:
        raise ValueError("Answer-writing settings changed since stage 0 (" + ", ".join(changed) + ").")
    if any(r["judge_run"] == judge_dir.name for r in registry["runs"]):
        raise ValueError(f"{judge_dir.name} is already registered.")
    counts = {s: sum(r["stage"] == s for r in registry["runs"]) for s in STAGES}
    if counts[stage] >= (2 if stage == 0 else 1):
        raise ValueError(f"Stage {stage} already has its runs.")
    if stage > 0 and counts[stage - 1] < (2 if stage == 1 else 1):
        raise ValueError(f"Stage {stage - 1} must be complete before stage {stage}.")

    # Scores take the readable name of the answers they grade (stage1_answers -> stage1_scores); the judge
    # itself still writes judge_<id>, because changing its code would break the frozen-judge check above
    answers = judge["baseline_run"]
    if answers.endswith("_answers"):
        target = judge_dir.with_name(answers[: -len("_answers")] + "_scores")
        if target != judge_dir:
            if target.exists():
                raise ValueError(f"{target.name} already exists.")
            judge_dir = judge_dir.rename(target)
    registry["runs"].append({
        "stage": stage, "judge_run": judge_dir.name, "baseline_run": answers,
        "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    Path(registry_path).parent.mkdir(parents=True, exist_ok=True)
    Path(registry_path).write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return registry


def ladder(registry, outputs=OUTPUTS, labels=EVALUATION_SET):
    """Everything the report shows: per-stage values, stage-0 noise and stage-to-stage comparisons."""
    runs = {}
    for entry in registry["runs"]:
        runs.setdefault(entry["stage"], []).append(load_run(Path(outputs) / entry["judge_run"]))
    if len(runs.get(0, [])) < 2:
        raise ValueError("Stage 0 needs both of its runs before the ladder can be built.")
    stages = {stage: stage_values(stage_runs) for stage, stage_runs in sorted(runs.items())}
    answers = {stage: stage_values(stage_runs, "answers", (NO_ANSWER,)) for stage, stage_runs in sorted(runs.items())}
    first, second = runs[0]
    thresholds = {**noise(first["values"], second["values"]),
                  **noise(first["answers"], second["answers"], (NO_ANSWER,))}
    comparisons = {}
    for stage in stages:
        if stage > 0:
            scores = compare(stages[stage - 1], stages[stage], thresholds)
            missing = compare(answers[stage - 1], answers[stage], thresholds, (NO_ANSWER,))
            comparisons[stage] = {"cases": scores["cases"], "metrics": {**scores["metrics"], **missing["metrics"]},
                                  "stop": scores["stop"] or missing["stop"]}
    sites = {}
    for stage_runs in runs.values():
        for run in stage_runs:
            sites.update(run["sites"])
    # Workbook labels also place cases that no run answered
    if Path(labels).exists():
        sites.update({case["case_id"]: case["disease_site"] for case in read_json(labels)["cases"]})
    return {"runs": runs, "stages": stages, "answers": answers, "noise": thresholds,
            "comparisons": comparisons, "sites": sites}


COLORS = {"worse": "#c0392b", "better": "#1e8449", "within noise": "#34495e"}


def fmt(key, value):
    return f"{value:.2f}" if key in SCALES else f"{value * 100:.0f}%"


def fmt_delta(key, value):
    return f"{value:+.2f}" if key in SCALES else f"{value * 100:+.0f} pts"


def fmt_noise(key, value):
    return f"± {value:.2f}" if key in SCALES else f"± {value * 100:.1f} pts"


def field(key):
    """No answer is counted over every case; the rubric metrics over judged cases."""
    return "answers" if key == "no_answer" else "values"


def stage0_means(built, key):
    """Both stage-0 run means over the cases both runs have, the same cases the noise is measured on."""
    first, second = (run[field(key)] for run in built["runs"][0])
    shared = set(first) & set(second)
    return [mean(values[case][key] for case in shared) for values in (first, second)]


def panel(key, label, direction, built):
    """One small chart: stage means, both stage-0 runs and the stage-0 noise band."""
    width, height, left, right, top, bottom = 300, 180, 42, 14, 30, 24
    low, high = (1.0, 5.0) if key in SCALES else (0.0, 1.0)
    # Inset the stages so stage-0 labels do not collide with the y-axis ticks
    pad = 16
    x = lambda stage: left + pad + (width - left - right - 2 * pad) * stage / (len(STAGES) - 1)
    y = lambda value: top + (height - top - bottom) * (high - value) / (high - low)
    stages = built["answers" if key == "no_answer" else "stages"]
    stage_mean = lambda stage: mean(v[key] for v in stages[stage].values())
    better = "higher" if direction > 0 else "lower"
    parts = [f'<text x="4" y="16" class="title">{escape(label)}</text>',
             f'<text x="{width - right}" y="16" class="tick" text-anchor="end">{better} is better</text>']
    for tick in ((1, 2, 3, 4, 5) if key in SCALES else (0, 0.25, 0.5, 0.75, 1)):
        text = f"{tick:g}" if key in SCALES else f"{tick * 100:.0f}%"
        parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" y2="{y(tick):.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 6}" y="{y(tick) + 4:.1f}" class="tick" text-anchor="end">{text}</text>')
    for stage in STAGES:
        parts.append(f'<text x="{x(stage):.1f}" y="{height - 6}" class="tick" text-anchor="middle">{stage}</text>')
    centre, band = stage_mean(0), built["noise"][key]
    upper, lower = y(min(high, centre + band)), y(max(low, centre - band))
    parts.append(f'<rect x="{left}" y="{upper:.1f}" width="{width - left - right}" height="{lower - upper:.1f}" class="band"/>')
    for value in stage0_means(built, key):
        parts.append(f'<circle cx="{x(0):.1f}" cy="{y(value):.1f}" r="3" class="run"/>')
    points = [(stage, stage_mean(stage)) for stage in sorted(built["stages"])]
    parts.append('<polyline class="line" points="' + " ".join(f"{x(s):.1f},{y(v):.1f}" for s, v in points) + '"/>')
    for stage, value in points:
        verdict = built["comparisons"][stage]["metrics"][key]["verdict"] if stage else "within noise"
        parts.append(f'<circle cx="{x(stage):.1f}" cy="{y(value):.1f}" r="4.5" fill="{COLORS[verdict]}">'
                     f'<title>Stage {stage}: {fmt(key, value)}, {verdict}</title></circle>')
        if stage == 0:
            # Right of the point: the two hollow run dots sit above and below it
            parts.append(f'<text x="{x(0) + 8:.1f}" y="{y(value) + 4:.1f}" class="value">{fmt(key, value)}</text>')
            continue
        # Labels go below points near the top, where they would hit the top tick
        label_y = y(value) + 16 if y(value) - 9 < top + 6 else y(value) - 9
        parts.append(f'<text x="{x(stage):.1f}" y="{label_y:.1f}" class="value" text-anchor="middle">{fmt(key, value)}</text>')
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(label)} by stage">{"".join(parts)}</svg>'


def change_table(built):
    later = sorted(built["comparisons"])
    head = ("<tr><th>Metric</th><th>Stage 0 run a</th><th>Run b</th><th>Noise</th>"
            + "".join(f"<th>Stage {stage}</th>" for stage in later) + "</tr>")
    rows = []
    for key, label, direction in ALL_METRICS:
        first, second = stage0_means(built, key)
        cells = [f"<td>{fmt(key, first)}</td><td>{fmt(key, second)}</td><td>{fmt_noise(key, built['noise'][key])}</td>"]
        for stage in later:
            m = built["comparisons"][stage]["metrics"][key]
            cells.append(f'<td class="{m["verdict"].replace(" ", "-")}">{fmt(key, m["after"])} '
                         f'<small>({fmt_delta(key, m["delta"])})</small><br><small>{m["verdict"]}</small></td>')
        better = "higher" if direction > 0 else "lower"
        rows.append(f"<tr><th>{escape(label)}<br><small>{better} is better</small></th>{''.join(cells)}</tr>")
    decision = "".join(f"<td><strong>{'STOP' if built['comparisons'][s]['stop'] else 'continue'}</strong></td>" for s in later)
    rows.append(f'<tr><th>Decision</th><td colspan="3"></td>{decision}</tr>')
    return f'<table class="changes">{head}{"".join(rows)}</table>'


def case_changes(before, after):
    """Case IDs behind a change: accuracy moves of a point or more, and Harm ≥ 2 gained or lost."""
    common = sorted(set(before) & set(after))
    moved = [f"{c} {after[c]['accuracy'] - before[c]['accuracy']:+g}" for c in common
             if abs(after[c]["accuracy"] - before[c]["accuracy"]) >= 1]
    gained = [c for c in common if after[c]["harm_2plus"] and not before[c]["harm_2plus"]]
    lost = [c for c in common if before[c]["harm_2plus"] and not after[c]["harm_2plus"]]
    return moved, gained, lost


def change_lists(built):
    pairs = [("Stage 0, run a → run b", built["runs"][0][0]["values"], built["runs"][0][1]["values"])]
    pairs += [(f"Stage {s - 1} → stage {s}", built["stages"][s - 1], built["stages"][s]) for s in sorted(built["comparisons"])]
    items = []
    for title, before, after in pairs:
        moved, gained, lost = (escape(", ".join(ids) or "none") for ids in case_changes(before, after))
        items.append(f"<li><strong>{escape(title)}</strong><br>Accuracy moved ≥ 1: {moved}<br>"
                     f"New Harm ≥ 2: {gained}<br>Harm ≥ 2 resolved: {lost}</li>")
    return '<ul class="cases">' + "".join(items) + "</ul>"


def heatmap(built):
    """Accuracy per case and run, grouped by disease site; H marks Harm ≥ 2."""
    columns = [("0" + "ab"[i], run) for i, run in enumerate(built["runs"][0])]
    columns += [(str(stage), runs[0]) for stage, runs in sorted(built["runs"].items()) if stage > 0]
    rows, site = [], None
    for case in sorted(built["sites"], key=lambda c: (built["sites"][c], c)):
        if built["sites"][case] != site:
            site = built["sites"][case]
            rows.append(f'<tr class="site"><th colspan="{len(columns) + 1}">{escape(site)}</th></tr>')
        cells = []
        for _, run in columns:
            value = run["values"].get(case)
            if run["answers"].get(case, {}).get("no_answer"):
                cells.append('<td class="noanswer" title="No answer generated">✕</td>')
                continue
            if value is None:
                cells.append('<td class="missing" title="Not judged">–</td>')
                continue
            # 💡 Hue 0 (red) at score 1 to 120 (green) at score 5, so colour alone shows the score
            hue = 120 * (value["accuracy"] - 1) / 4
            mark = '<sup>H</sup>' if value["harm_2plus"] else ""
            cells.append(f'<td style="background:hsl({hue:.0f},65%,82%)">{value["accuracy"]:g}{mark}</td>')
        rows.append(f"<tr><th>{escape(case)}</th>{''.join(cells)}</tr>")
    head = "".join(f"<th>{name}</th>" for name, _ in columns)
    return f'<table class="heat"><tr><th>Case</th>{head}</tr>{"".join(rows)}</table>'


def runs_table(registry, built):
    by_name = {run["name"]: run for runs in built["runs"].values() for run in runs}
    rows = []
    for entry in registry["runs"]:
        run, (step, change) = by_name[entry["judge_run"]], STAGES[entry["stage"]]
        usage = run["usage"]
        tokens = f"{usage['input_tokens']:,} / {usage['output_tokens']:,}" if usage else "–"
        missing = sorted(case for case, value in run["answers"].items() if value["no_answer"])
        rows.append(f"<tr><td>{entry['stage']}</td><td>{step}</td><td>{escape(change)}</td>"
                    f"<td><code>{escape(entry['baseline_run'])}</code></td><td><code>{escape(entry['judge_run'])}</code></td>"
                    f"<td>{len(run['values'])}</td><td>{escape(', '.join(missing)) or '–'}</td>"
                    f"<td>{escape(', '.join(run['errors'])) or '–'}</td><td>{tokens}</td></tr>")
    return ("<table><tr><th>Stage</th><th>Roadmap</th><th>Change</th><th>Answers</th><th>Judge run</th>"
            "<th>Scored</th><th>No answer</th><th>Judge errors</th><th>Judge tokens in / out</th></tr>" + "".join(rows) + "</table>")


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Build ladder</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f7f7f8; color: #1c1c1e; }}
  header {{ padding: 16px 24px; background: #1c1c1e; color: #fff; }}
  header span {{ color: #b0b0b5; font-size: 13px; margin-left: 12px; }}
  main {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
  section {{ background: #fff; border: 1px solid #e2e2e4; border-radius: 10px; padding: 16px 20px; margin-bottom: 20px; overflow-x: auto; }}
  h2 {{ margin-top: 0; font-size: 18px; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  th, td {{ border: 1px solid #e2e2e4; padding: 5px 8px; text-align: left; vertical-align: top; }}
  small, .note {{ color: #6b6b70; }}
  .note {{ font-size: 13px; }}
  .panels {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
  svg {{ width: 100%; height: auto; }}
  svg .title {{ font-size: 13px; font-weight: 600; fill: #1c1c1e; }}
  svg .tick {{ font-size: 10px; fill: #6b6b70; }}
  svg .value {{ font-size: 10px; fill: #1c1c1e; }}
  svg .grid {{ stroke: #ececef; }}
  svg .band {{ fill: #d6dbe0; opacity: 0.7; }}
  svg .line {{ fill: none; stroke: #8a8a8f; stroke-width: 1.5; }}
  svg .run {{ fill: #fff; stroke: #34495e; stroke-width: 1.2; }}
  .worse {{ background: #fdecea; }}
  .better {{ background: #e9f7ef; }}
  .heat td {{ text-align: center; min-width: 34px; }}
  .heat tr.site th {{ background: #f0f0f2; }}
  .heat td.missing {{ color: #b0b0b5; }}
  .heat td.noanswer {{ background: #2c2c2e; color: #fff; }}
  sup {{ color: #c0392b; font-weight: 700; }}
  ul.cases li {{ margin-bottom: 8px; font-size: 13px; }}
</style>
</head>
<body>
<header><strong>Build ladder</strong><span>generated {generated} · scores and case IDs only</span></header>
<main>
<section>
  <h2>Registered runs</h2>
  {runs}
  <p class="note">Evaluation set <code>{inputs}</code> · answers by {writer} · judge {model}, prompt <code>{prompt}</code>.
  The judge has not yet been checked against physicians (step ⑫): use these scores as a development guide only.</p>
</section>
<section>
  <h2>Decision rule</h2>
  <p>Noise for each metric is twice the standard error of the per-case difference between the two stage-0 runs,
  but never less than the gap between the two runs' means, or one case's worth (1 / number of cases). A stage that falls below the previous stage by more than the
  noise on any metric is <strong>worse</strong>, and the decision is <strong>STOP</strong>: find the cause before the
  next stage. Stage 1 is compared with the case-by-case average of the two stage-0 runs. Rubric metrics use the
  cases scored in both stages; <em>No answer</em> counts every evaluation case, so a stage cannot look better by
  failing to answer hard cases.</p>
</section>
<section>
  <h2>Scores by stage</h2>
  <div class="panels">{panels}</div>
  <p class="note">Grey band: stage-0 average ± noise. Hollow dots: the two stage-0 runs. Point colour against the
  previous stage: <span style="color:#c0392b">red worse</span>, <span style="color:#1e8449">green better</span>,
  dark grey within noise.</p>
</section>
<section>
  <h2>Change against the previous stage</h2>
  {changes}
  {lists}
</section>
<section>
  <h2>Accuracy by case</h2>
  <p class="note">Cell colour runs from red (1) to green (5). <sup>H</sup> marks Harm ≥ 2, ✕ no answer generated,
  – not judged. Columns 0a and 0b are the two stage-0 runs.</p>
  {heatmap}
</section>
</main>
</body>
</html>"""


def render(registry, built):
    frozen = registry["frozen_judge"]
    return PAGE_TEMPLATE.format(
        generated=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        runs=runs_table(registry, built), inputs=registry["evaluation_inputs_sha256"][:12],
        model=escape(frozen["judge_model"]), prompt=frozen["prompt_sha256"][:12],
        writer=escape(registry["frozen_writer"]["model_id"]),
        panels="".join(panel(key, label, direction, built) for key, label, direction in ALL_METRICS),
        changes=change_table(built), lists=change_lists(built), heatmap=heatmap(built),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Register a judge run for a stage, then rebuild the report.")
    add.add_argument("stage", type=int)
    add.add_argument("judge_run", type=Path, help="Judge output folder, e.g. outputs/judge_<id>")
    commands.add_parser("build", help="Rebuild the report from the registry.")
    args = parser.parse_args(argv)
    try:
        if args.command == "add":
            registry = register(args.stage, args.judge_run)
            print(f"Registered {args.judge_run.name} as stage {args.stage}.")
        else:
            registry = read_json(REGISTRY)
        if sum(entry["stage"] == 0 for entry in registry["runs"]) < 2:
            print("The report needs both stage-0 runs.")
            return 0
        REPORT.write_text(render(registry, ladder(registry)), encoding="utf-8")
        print(f"Local report: {REPORT}")
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Ladder failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
