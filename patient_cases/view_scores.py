"""Render a stage-0 scores.jsonl as a local, readable HTML dashboard.

No network calls. Output stays on disk next to the input file.
Follows the dataviz skill: sequential blue for magnitude, status
green/red for pass/fail flags, thin marks, hover tooltips, light/dark
surfaces from the validated default palette.
"""

import argparse
from collections import Counter, defaultdict
from html import escape
from json import dumps, loads
from pathlib import Path
from statistics import mean

NUMERIC_METRICS = [
    ("accuracy", "Accuracy", "higher is better"),
    ("completeness", "Completeness", "higher is better"),
    ("guideline_adherence", "Guideline adherence", "higher is better"),
    ("harm", "Harm", "lower is better"),
]
# ⚠️ Manifests written before the 1–5 Harm rubric have no harm_scale: those runs used 0–3, gate 2.
OLD_HARM = {"harm_scale": [0, 3], "harm_gate": 2}


def harm_rubric(manifest):
    return {k: manifest.get(k, default) for k, default in OLD_HARM.items()}


def scale(key, harm):
    lo, hi = harm["harm_scale"] if key == "harm" else (1, 5)
    return list(range(lo, hi + 1))


def load_records(path):
    with path.open("r", encoding="utf-8") as handle:
        return [loads(line) for line in handle if line.strip()]


def build_model(records, harm):
    n_cases = len(records)
    metrics = {}
    for key, _, _ in NUMERIC_METRICS:
        values = [r["final"][key] for r in records if isinstance(r["final"].get(key), (int, float))]
        counts = Counter(int(round(v)) for v in values)
        by_site = defaultdict(list)
        for r in records:
            v = r["final"].get(key)
            if isinstance(v, (int, float)):
                by_site[r.get("disease_site", "Unknown")].append(v)
        metrics[key] = {
            "mean": mean(values) if values else 0,
            "counts": {s: counts.get(s, 0) for s in scale(key, harm)},
            "by_site": {site: mean(vals) for site, vals in by_site.items()},
        }

    flags = Counter()
    for r in records:
        for f in r.get("flags", []):
            flags[f] += 1

    categorical = {}
    for key in ("hallucination", "citations_verifiable", "appropriate_abstention"):
        categorical[key] = dict(Counter(r["final"].get(key) for r in records))

    sites = sorted({r.get("disease_site", "Unknown") for r in records})

    cases = []
    for r in records:
        cases.append(
            {
                "case_id": r.get("case_id", ""),
                "disease_site": r.get("disease_site", ""),
                "final": r.get("final", {}),
                "flags": r.get("flags", []),
                "runs": [
                    {
                        metric: run.get("scores", {}).get(metric, {})
                        for metric in (
                            "accuracy",
                            "completeness",
                            "guideline_adherence",
                            "harm",
                            "hallucination",
                            "citations_verifiable",
                            "appropriate_abstention",
                        )
                    }
                    for run in r.get("runs", [])
                ],
            }
        )

    return {
        "n_cases": n_cases,
        "harm": harm,
        "metrics": metrics,
        "flags": dict(flags),
        "categorical": categorical,
        "sites": sites,
        "cases": cases,
    }


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Scores dashboard &mdash; {title}</title>
<style>
  :root {{
    color-scheme: light;
    --surface-1:      #fcfcfb;
    --page:           #f9f9f7;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --gridline:       #e1e0d9;
    --baseline:       #c3c2b7;
    --border:         rgba(11,11,11,0.10);
    --series-1:       #2a78d6;
    --series-1-wash:  rgba(42,120,214,0.10);
    --status-good:    #0ca30c;
    --status-critical:#d03b3b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface-1:      #1a1a19;
      --page:           #0d0d0d;
      --text-primary:   #f0efec;
      --text-secondary: #c3c2b7;
      --text-muted:     #898781;
      --gridline:       #2c2c2a;
      --baseline:       #383835;
      --border:         rgba(255,255,255,0.10);
      --series-1:       #3987e5;
      --series-1-wash:  rgba(57,135,229,0.14);
      --status-good:    #0ca30c;
      --status-critical:#e66767;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--page); color: var(--text-primary);
         font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
  header {{ padding: 18px 24px; background: var(--surface-1); border-bottom: 1px solid var(--border);
           position: sticky; top: 0; z-index: 10; }}
  header h1 {{ margin: 0; font-size: 16px; }}
  header p {{ margin: 2px 0 0; color: var(--text-secondary); font-size: 13px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 24px; }}
  .card {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
          padding: 18px 20px; margin-bottom: 20px; }}
  .card h2 {{ margin: 0 0 12px; font-size: 14px; color: var(--text-primary); }}
  .card .hint {{ font-size: 12px; color: var(--text-muted); margin: -8px 0 14px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap: 12px; margin-bottom: 20px; }}
  .tile {{ background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
          display: flex; flex-direction: column; }}
  .tile .label {{ font-size: 12px; color: var(--text-secondary); min-height: 2.6em; }}
  .tile .value {{ font-size: 24px; font-weight: 600; color: var(--text-primary); }}
  .tile .value .of {{ font-size: 13px; font-weight: 400; color: var(--text-muted); }}
  .tile .sub {{ font-size: 11px; color: var(--text-muted); margin-top: 4px; display: flex; align-items: center; gap: 6px; }}
  .tile .sub .dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--status-critical); flex: none; }}
  .hist-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; }}
  .hist {{ text-align: center; }}
  .hist .hist-title {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 6px; }}
  svg text {{ fill: var(--text-secondary); font-size: 11px; font-family: inherit; }}
  svg .axis {{ stroke: var(--baseline); stroke-width: 1; }}
  .bar-mark {{ fill: var(--series-1); cursor: pointer; }}
  .bar-mark:hover {{ filter: brightness(1.12); }}
  .site-table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
  .site-table th {{ text-align: left; font-weight: 500; color: var(--text-muted); padding: 6px 10px; border-bottom: 1px solid var(--gridline); }}
  .site-table td {{ padding: 6px 10px; border-bottom: 1px solid var(--gridline); white-space: nowrap; }}
  .cell-bar-wrap {{ display: flex; align-items: center; gap: 6px; min-width: 120px; }}
  .cell-bar-track {{ flex: 1; height: 8px; background: var(--gridline); border-radius: 4px; overflow: hidden; }}
  .cell-bar-fill {{ height: 100%; background: var(--series-1); border-radius: 4px; }}
  .cell-val {{ font-variant-numeric: tabular-nums; color: var(--text-secondary); width: 28px; text-align: right; }}
  .chip-row {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .chip {{ display: flex; align-items: center; gap: 8px; border: 1px solid var(--border); border-radius: 8px;
          padding: 8px 12px; font-size: 12.5px; }}
  .chip .dot {{ width: 9px; height: 9px; border-radius: 50%; flex: none; }}
  .chip .dot.good {{ background: var(--status-good); }}
  .chip .dot.critical {{ background: var(--status-critical); }}
  .chip .dot.neutral {{ background: var(--text-muted); }}
  .chip .n {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
  details.case {{ border: 1px solid var(--gridline); border-radius: 8px; margin: 6px 0; padding: 8px 12px; }}
  details.case summary {{ cursor: pointer; font-size: 13px; display: flex; gap: 10px; align-items: baseline; }}
  details.case summary .site {{ color: var(--text-muted); font-size: 11px; }}
  details.case .run {{ margin-top: 8px; padding-top: 8px; border-top: 1px dashed var(--gridline); font-size: 12.5px; }}
  details.case .metric-row {{ margin: 3px 0; }}
  details.case .metric-row b {{ font-variant-numeric: tabular-nums; }}
  .tooltip {{ position: fixed; pointer-events: none; background: var(--text-primary); color: var(--surface-1);
             font-size: 11.5px; padding: 6px 9px; border-radius: 6px; max-width: 260px; z-index: 50; opacity: 0;
             transition: opacity 80ms; line-height: 1.4; }}
  .tooltip.show {{ opacity: 1; }}
  .tooltip .tt-val {{ font-weight: 700; }}
  .flag-row {{ display: flex; align-items: center; gap: 10px; margin: 6px 0; font-size: 12.5px; }}
  details.case summary code {{ font-family: inherit; font-size: 11px; color: var(--text-secondary);
                               border: 1px solid var(--border); border-radius: 4px; padding: 1px 6px; }}
  .flag-label {{ width: 150px; font-size: 12.5px; color: var(--text-secondary); flex: none; }}
</style>
</head>
<body>
<div class="viz-root">
<header>
  <h1>{title}</h1>
  <p>{n_cases} scored cases &middot; rendered locally, not uploaded</p>
</header>
<main>
{content}
</main>
</div>
<div class="tooltip" id="tt"></div>
<script>
const tt = document.getElementById('tt');
function showTip(e, label, value) {{
  tt.textContent = '';
  const v = document.createElement('span');
  v.className = 'tt-val';
  v.textContent = value;
  tt.appendChild(v);
  tt.appendChild(document.createTextNode(' — '));
  tt.appendChild(document.createTextNode(label));
  tt.classList.add('show');
  moveTip(e);
}}
function moveTip(e) {{
  tt.style.left = (e.clientX + 14) + 'px';
  tt.style.top = (e.clientY + 14) + 'px';
}}
function hideTip() {{ tt.classList.remove('show'); }}
document.querySelectorAll('[data-tip-label]').forEach(function(el) {{
  el.addEventListener('pointermove', function(e) {{ showTip(e, el.dataset.tipLabel, el.dataset.tipValue); }});
  el.addEventListener('pointerleave', hideTip);
}});
</script>
</body>
</html>"""


def histogram_svg(key, label, direction, counts):
    width, height = 220, 140
    pad_bottom = 20
    max_count = max(counts.values()) or 1
    bar_w = 24
    gap = 10
    steps = list(counts)
    n = len(steps)
    total_w = n * bar_w + (n - 1) * gap
    x0 = (width - total_w) / 2
    bars = []
    for i, s in enumerate(steps):
        c = counts[s]
        h = (c / max_count) * (height - pad_bottom - 18) if max_count else 0
        x = x0 + i * (bar_w + gap)
        y = height - pad_bottom - h
        bars.append(
            f'<rect class="bar-mark" x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{max(h, 1):.1f}" '
            f'rx="4" ry="4" data-tip-label="score {s}" data-tip-value="{c} cases"></rect>'
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle">{c}</text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{height - 4}" text-anchor="middle">{s}</text>'
        )
    axis = f'<line class="axis" x1="{x0 - 6}" y1="{height - pad_bottom}" x2="{x0 + total_w + 6}" y2="{height - pad_bottom}"></line>'
    return f"""
    <div class="hist">
      <div class="hist-title">{escape(label)} <span style="color:var(--text-muted)">({direction})</span></div>
      <svg viewBox="0 0 {width} {height}" width="100%" height="{height}">{axis}{''.join(bars)}</svg>
    </div>"""


def site_table(model):
    rows = []
    for site in model["sites"]:
        cells = []
        for key, _, _ in NUMERIC_METRICS:
            v = model["metrics"][key]["by_site"].get(site)
            if v is None:
                cells.append("<td>&mdash;</td>")
                continue
            pct = max(0, min(100, (v / scale(key, model["harm"])[-1]) * 100))
            cells.append(
                f'<td><div class="cell-bar-wrap" data-tip-label="{escape(site)} &middot; {key}" '
                f'data-tip-value="{v:.2f}"><div class="cell-bar-track">'
                f'<div class="cell-bar-fill" style="width:{pct:.0f}%"></div></div>'
                f'<span class="cell-val">{v:.1f}</span></div></td>'
            )
        rows.append(f"<tr><td>{escape(site)}</td>{''.join(cells)}</tr>")
    headers = "".join(f"<th>{escape(label)}</th>" for _, label, _ in NUMERIC_METRICS)
    return f"""
    <table class="site-table">
      <thead><tr><th>Disease site</th>{headers}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


def flags_bars(flags, n_cases):
    if not flags:
        return "<p class='hint'>No disagreement flags recorded.</p>"
    ordered = sorted(flags.items(), key=lambda kv: -kv[1])
    rows = []
    max_v = max(flags.values()) or 1
    for name, count in ordered:
        pct = (count / max_v) * 100
        rows.append(
            f'<div class="flag-row"><span class="flag-label">{escape(name)}</span>'
            f'<div class="cell-bar-track" style="flex:1" data-tip-label="{escape(name)}" '
            f'data-tip-value="{count} of {n_cases} cases"><div class="cell-bar-fill" style="width:{pct:.0f}%"></div></div>'
            f'<span class="cell-val">{count}</span></div>'
        )
    return "".join(rows)


def status_chip(label, count, kind):
    return (
        f'<div class="chip"><span class="dot {kind}"></span>'
        f'<span>{escape(label)}</span><span class="n">{count}</span></div>'
    )


def categorical_section(model):
    hall = model["categorical"].get("hallucination", {})
    cite = model["categorical"].get("citations_verifiable", {})
    abst = model["categorical"].get("appropriate_abstention", {})
    parts = ['<div class="chip-row">']
    parts.append(status_chip(f"Hallucination: N (none detected)", hall.get("N", 0), "good"))
    parts.append(status_chip(f"Hallucination: Y (detected)", hall.get("Y", 0), "critical"))
    parts.append("</div><div class='chip-row' style='margin-top:8px'>")
    parts.append(status_chip("Citations verifiable: Y", cite.get("Y", 0), "good"))
    parts.append(status_chip("Citations verifiable: N", cite.get("N", 0), "critical"))
    parts.append("</div><div class='chip-row' style='margin-top:8px'>")
    for val in ("Y", "N", "NA"):
        if val in abst:
            parts.append(status_chip(f"Appropriate abstention: {val}", abst[val], "neutral"))
    parts.append("</div>")
    return "".join(parts)


def case_detail(case):
    f = case["final"]
    chips = " ".join(f"<code>{escape(fl)}</code>" for fl in case.get("flags", []))
    summary_bits = " &middot; ".join(
        f"{label}: {f.get(key, '—')}" for key, label, _ in NUMERIC_METRICS
    )
    runs_html = ""
    for i, run in enumerate(case["runs"], start=1):
        lines = []
        for key, label, _ in NUMERIC_METRICS:
            s = run.get(key, {})
            lines.append(
                f'<div class="metric-row"><b>{s.get("score", "—")}</b> {escape(label)} &mdash; '
                f'<span style="color:var(--text-muted)">{escape(str(s.get("reason", "")))}</span></div>'
            )
        for key in ("hallucination", "citations_verifiable", "appropriate_abstention"):
            s = run.get(key, {})
            lines.append(
                f'<div class="metric-row"><b>{s.get("value", "—")}</b> {escape(key)} &mdash; '
                f'<span style="color:var(--text-muted)">{escape(str(s.get("reason", "")))}</span></div>'
            )
        runs_html += f'<div class="run"><strong>Run {i}</strong>{"".join(lines)}</div>'
    return f"""
    <details class="case">
      <summary>{escape(case['case_id'])} <span class="site">{escape(case['disease_site'])}</span>
        <span style="color:var(--text-muted)">&middot; {summary_bits}</span>
        {('&middot; ' + chips) if chips else ''}
      </summary>
      {runs_html}
    </details>"""


def build_html(model, title):
    tiles = [
        f'<div class="tile"><div class="label">Scored cases</div><div class="value">{model["n_cases"]}</div></div>'
    ]
    for key, label, direction in NUMERIC_METRICS:
        m = model["metrics"][key]["mean"]
        tiles.append(
            f'<div class="tile"><div class="label">Mean {escape(label.lower())}</div>'
            f'<div class="value">{m:.2f}<span class="of"> / {scale(key, model["harm"])[-1]}</span></div><div class="sub">{direction}</div></div>'
        )
    harm_lo, harm_hi = model["harm"]["harm_scale"]
    n = model["n_cases"]
    hall_y = model["categorical"].get("hallucination", {}).get("Y", 0)
    cite_y = model["categorical"].get("citations_verifiable", {}).get("Y", 0)
    tiles.append(
        f'<div class="tile"><div class="label">Hallucination rate</div>'
        f'<div class="value">{hall_y / n * 100:.0f}%</div>'
        f'<div class="sub"><span class="dot"></span>{hall_y} of {n} cases</div></div>'
    )
    tiles.append(
        f'<div class="tile"><div class="label">Citations verifiable</div>'
        f'<div class="value">{cite_y / n * 100:.0f}%</div>'
        f'<div class="sub"><span class="dot"></span>{cite_y} of {n} cases</div></div>'
    )

    hists = "".join(
        histogram_svg(key, label, direction, model["metrics"][key]["counts"])
        for key, label, direction in NUMERIC_METRICS
    )

    content = f"""
    <div class="tiles">{''.join(tiles)}</div>

    <div class="card">
      <h2>Score distributions</h2>
      <p class="hint">Count of cases at each score. Harm is scored {harm_lo}&ndash;{harm_hi}; the others 1&ndash;5.</p>
      <div class="hist-grid">{hists}</div>
    </div>

    <div class="card">
      <h2>Mean score by disease site</h2>
      <p class="hint">Bar length scaled to each metric's maximum ({harm_hi} for harm, 5 for the others). Harm: lower is better; the others: higher is better.</p>
      {site_table(model)}
    </div>

    <div class="card">
      <h2>Review flags</h2>
      <p class="hint">Cases flagged for review out of {model['n_cases']}: harm score of {model['harm']['harm_gate']} or more, or the two scoring runs disagreed.</p>
      {flags_bars(model['flags'], model['n_cases'])}
    </div>

    <div class="card">
      <h2>Safety &amp; grounding outcomes</h2>
      {categorical_section(model)}
    </div>

    <div class="card">
      <h2>Per-case detail</h2>
      <p class="hint">Expand a case to see both scoring runs and their reasons.</p>
      {''.join(case_detail(c) for c in model['cases'])}
    </div>
    """
    return PAGE_TEMPLATE.format(title=escape(title), n_cases=model["n_cases"], content=content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="Path to a scores.jsonl file")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: alongside input)")
    args = parser.parse_args(argv)

    records = load_records(args.scores)
    manifest_path = args.scores.with_name("manifest.json")
    manifest = loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    model = build_model(records, harm_rubric(manifest))
    output = args.output or args.scores.with_name("scores_dashboard.html")
    output.write_text(build_html(model, args.scores.parent.name), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
