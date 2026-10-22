"""Side-by-side review page: model answer (left) and judge scores (right).

Pairs a *_scores directory with the answers run named in its manifest
(baseline_run). No network calls; output stays on disk next to the scores.
"""

import argparse
from html import escape
from json import dumps, loads
from pathlib import Path


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [loads(line) for line in handle if line.strip()]


def build_payload(manifest, scores_dir, answers_dir):
    answers = {r["case_id"]: r for r in load_jsonl(answers_dir / "results.jsonl")}
    cases = []
    for s in load_jsonl(scores_dir / "scores.jsonl"):
        r = answers.get(s["case_id"], {})
        resp = r.get("response") or {}
        cases.append(
            {
                "id": s["case_id"],
                "site": s.get("disease_site", ""),
                "labels": r.get("case_labels") or {},
                "status": r.get("status", "missing from answers run"),
                "elapsed": r.get("elapsed_seconds"),
                "query": r.get("query", ""),
                "answer": resp.get("answer", ""),
                "confidence": resp.get("confidence_score"),
                "self_critique": resp.get("self_critique") or {},
                # task_plan (~100 KB per case of raw retrieval chunks) is left out on purpose.
                "passages": [
                    {k: p.get(k) for k in ("citation_id", "source_path", "text", "source_type", "topic")}
                    for p in resp.get("context_passages") or []
                ],
                "sources": {
                    str(c.get("citation_id")): c.get("source_url")
                    for c in resp.get("citations") or []
                    if c.get("source_url")
                },
                "engine_log": r.get("engine_log", ""),
                "final": s.get("final", {}),
                "flags": s.get("flags", []),
                "runs": [run.get("scores", {}) for run in s.get("runs", [])],
            }
        )
    meta = {
        "title": f"{answers_dir.name} vs {scores_dir.name}",
        "rubric": manifest.get("rubric", {}).get("anchors", {}),
        "judge_model": manifest.get("judge_model", ""),
    }
    return {"meta": meta, "cases": cases}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review &mdash; __TITLE__</title>
<style>
:root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #898781;
  --gridline: #e1e0d9; --border: rgba(11,11,11,0.10);
  --accent: #2a78d6;
  --status-good: #0ca30c; --status-serious: #ec835a; --status-critical: #d03b3b;
  --claim-wash: rgba(236,131,90,0.24);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #898781;
    --gridline: #2c2c2a; --border: rgba(255,255,255,0.10);
    --accent: #3987e5;
    --status-critical: #e66767;
    --claim-wash: rgba(236,131,90,0.30);
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; display: flex; flex-direction: column; background: var(--page); color: var(--text-primary);
       font: 14px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
a { color: var(--accent); }
header { flex: none; display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 10px 20px;
         background: var(--surface-1); border-bottom: 1px solid var(--border); }
header h1 { margin: 0; font-size: 14px; font-weight: 600; }
header select, header button { font: inherit; font-size: 13px; color: var(--text-primary); background: var(--page);
         border: 1px solid var(--border); border-radius: 6px; padding: 4px 8px; }
header select { max-width: 46vw; }
header button { cursor: pointer; }
.panes { flex: 1; min-height: 0; display: grid; grid-template-columns: 1fr 1fr; }
.pane { overflow: auto; padding: 0 22px 48px; }
.pane + .pane { border-left: 1px solid var(--border); }
.pane-title { position: sticky; top: 0; z-index: 2; margin: 0; padding: 12px 0 8px; background: var(--page);
              font-size: 11px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); }
@media (max-width: 900px) {
  html, body { height: auto; } body { display: block; } .panes { display: block; }
  .pane { overflow: visible; } .pane + .pane { border-left: 0; border-top: 1px solid var(--border); }
}
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; }
.card > h2, details.card > summary { margin: 0 0 8px; font-size: 13px; font-weight: 600; }
details.card > summary { cursor: pointer; margin: 0; }
details.card[open] > summary { margin-bottom: 8px; }
.prose { white-space: pre-wrap; word-wrap: break-word; }
.small { font-size: 12px; }
.muted { color: var(--text-muted); }
.secondary { color: var(--text-secondary); }
.legend { margin: 2px 0 10px; }
mark.claim { background: var(--claim-wash); color: inherit; border-radius: 3px; padding: 0 1px; cursor: pointer; }
.claim-tag { font-size: 10px; color: var(--text-secondary); margin-left: 1px; }
.cite { color: var(--accent); cursor: pointer; text-decoration: underline; text-decoration-thickness: 1px; text-underline-offset: 2px; }
.cite.fail::after { content: ""; display: inline-block; width: 6px; height: 6px; margin-left: 2px; border-radius: 50%;
                    background: var(--status-critical); vertical-align: middle; }
.flash { animation: flash 1.6s ease-out; }
@keyframes flash { 0%, 55% { outline: 2px solid var(--accent); outline-offset: 2px; }
                   100% { outline: 2px solid transparent; outline-offset: 2px; } }
details.passage { border-top: 1px solid var(--gridline); padding: 7px 0; }
details.passage > summary { cursor: pointer; font-size: 12.5px; }
details.passage > summary > * { margin-right: 6px; }
details.passage .prose { margin-top: 6px; }
.pid { font-weight: 600; font-variant-numeric: tabular-nums; }
.badge { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-secondary);
         border: 1px solid var(--border); border-radius: 4px; padding: 0 6px; white-space: nowrap; }
.dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot.good { background: var(--status-good); } .dot.critical { background: var(--status-critical); }
.dot.serious { background: var(--status-serious); } .dot.neutral { background: var(--text-muted); }
pre.log { margin: 0; white-space: pre-wrap; font: 11.5px/1.45 ui-monospace, Menlo, monospace; color: var(--text-secondary); }
.case-head { margin: 4px 0 12px; }
.case-head h2 { margin: 0; font-size: 18px; }
.pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.tiles { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin-bottom: 12px; }
@media (max-width: 1200px) { .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.tile { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.tile .label { font-size: 11.5px; color: var(--text-secondary); }
.tile .value { font-size: 20px; font-weight: 600; }
.tile .of { font-size: 12px; font-weight: 400; color: var(--text-muted); }
.tile .sub { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--text-muted); margin-top: 2px; }
table.runs { width: 100%; border-collapse: collapse; font-size: 12.5px; }
table.runs th { text-align: left; font-weight: 500; color: var(--text-muted); padding: 6px 8px; border-bottom: 1px solid var(--gridline); }
table.runs td { vertical-align: top; padding: 8px; border-bottom: 1px solid var(--gridline); }
table.runs td.metric { width: 20%; color: var(--text-secondary); }
table.runs td.metric .badge { margin-top: 4px; }
table.runs .v { font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; }
table.runs .reason { color: var(--text-secondary); margin-top: 2px; }
table.runs ul { margin: 2px 0 0; padding-left: 18px; }
ol.claims { margin: 0; padding-left: 20px; }
ol.claims li { margin: 8px 0; }
ol.claims li.linked { cursor: pointer; }
.rubric dt { font-size: 12px; font-weight: 600; margin-top: 8px; }
.rubric dd { margin: 2px 0 0; font-size: 12px; color: var(--text-secondary); }
</style>
</head>
<body>
<header>
  <h1>Review</h1>
  <span class="small muted">__TITLE__</span>
  <button id="prev" title="Previous case (left arrow)">&lsaquo; Prev</button>
  <select id="case" aria-label="Case"></select>
  <button id="next" title="Next case (right arrow)">Next &rsaquo;</button>
  <span id="pos" class="small muted"></span>
  <span class="small muted">&larr; &rarr; to switch cases &middot; rendered locally, not uploaded</span>
</header>
<div class="panes">
  <div class="pane" id="result"></div>
  <div class="pane" id="score"></div>
</div>
<script type="application/json" id="data">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const CASES = DATA.cases, META = DATA.meta;
const METRICS = [
  ['accuracy', 'Accuracy', 5],
  ['completeness', 'Completeness', 5],
  ['guideline_adherence', 'Guideline adherence', 5],
  ['harm', 'Harm', 3],
];
const FLAGS = [
  ['hallucination', 'Hallucination'],
  ['citations_verifiable', 'Citations verifiable'],
  ['appropriate_abstention', 'Appropriate abstention'],
];
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 2) => (v == null || v === '' ? '—' : Number(v).toFixed(d));
const show = (v) => (v == null || v === '' ? '—' : String(v));

// All data-derived text goes through textContent / createTextNode, never innerHTML.
function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = String(text);
  return e;
}
function dot(kind) { return el('span', 'dot ' + kind); }
function badge(kind, text) {
  const b = el('span', 'badge');
  if (kind) b.appendChild(dot(kind));
  b.appendChild(document.createTextNode(text));
  return b;
}
function section(parent, title, collapsible, open) {
  const c = el(collapsible ? 'details' : 'section', 'card');
  if (collapsible) { c.open = !!open; c.appendChild(el('summary', null, title)); }
  else c.appendChild(el('h2', null, title));
  parent.appendChild(c);
  return c;
}
function reveal(node) {
  if (!node) return;
  for (let p = node.parentElement; p; p = p.parentElement) if (p.tagName === 'DETAILS') p.open = true;
  node.scrollIntoView({ block: 'center', behavior: 'smooth' });
  node.classList.remove('flash');
  void node.offsetWidth;  // restart the animation on repeat clicks
  node.classList.add('flash');
}
function flagStatus(key, v) {
  const map = {
    hallucination: { Y: ['critical', 'fabrication found'], N: ['good', 'none found'] },
    citations_verifiable: { Y: ['good', 'all supported'], N: ['critical', 'not all supported'] },
    appropriate_abstention: { Y: ['good', 'appropriate'], N: ['critical', 'inappropriate'], NA: ['neutral', 'not applicable'] },
  };
  return (map[key] || {})[v] || ['neutral', '—'];
}
function runValue(run, key) {
  const s = run[key] || {};
  return s.score != null ? s.score : s.value;
}

function judgeFindings(c) {
  const failing = new Map();
  const claims = [];
  const byText = new Map();
  c.runs.forEach((run, i) => {
    ((run.citations_verifiable || {}).failing_citations || []).forEach((id) => {
      const k = String(id);
      if (!failing.has(k)) failing.set(k, []);
      if (!failing.get(k).includes(i + 1)) failing.get(k).push(i + 1);
    });
    (run.unsupported_claims || []).forEach((t) => {
      const k = String(t).trim().toLowerCase();
      if (!k) return;
      if (!byText.has(k)) {
        const o = { text: String(t).trim(), runs: [], found: false };
        byText.set(k, o);
        claims.push(o);
      }
      const o = byText.get(k);
      if (!o.runs.includes(i + 1)) o.runs.push(i + 1);
    });
  });
  return { failing, claims };
}

// Strip **bold** markers, then split the clean text at every bold / claim / citation boundary
// so each segment can be wrapped independently even when the ranges overlap.
function renderAnswer(container, raw, findings) {
  let text = '', last = 0, m;
  const bold = [];
  const boldRe = /\*\*([\s\S]+?)\*\*/g;
  while ((m = boldRe.exec(raw))) {
    text += raw.slice(last, m.index);
    const start = text.length;
    text += m[1];
    bold.push([start, text.length]);
    last = boldRe.lastIndex;
  }
  text += raw.slice(last);

  const lower = text.toLowerCase();
  const claimRanges = [];
  findings.claims.forEach((cl, ci) => {
    const q = cl.text.toLowerCase();
    for (let i = lower.indexOf(q); i !== -1; i = lower.indexOf(q, i + q.length)) {
      claimRanges.push([i, i + q.length, ci]);
      cl.found = true;
    }
  });

  const citeRanges = [];
  const cited = new Set();
  const citeRe = /\[(\d+(?:\s*,\s*\d+)*)\]/g;
  while ((m = citeRe.exec(text))) {
    const ids = m[1].split(/\s*,\s*/);
    ids.forEach((id) => cited.add(id));
    citeRanges.push([m.index, citeRe.lastIndex, ids]);
  }

  const cuts = new Set([0, text.length]);
  [bold, claimRanges, citeRanges].forEach((rs) => rs.forEach((r) => { cuts.add(r[0]); cuts.add(r[1]); }));
  const pts = [...cuts].sort((a, b) => a - b);
  const cover = (rs, a) => rs.find((r) => r[0] <= a && a < r[1]);
  const anchored = new Set();
  for (let k = 0; k < pts.length - 1; k++) {
    const a = pts[k], b = pts[k + 1];
    let node = document.createTextNode(text.slice(a, b));
    const cite = cover(citeRanges, a);
    if (cite) {
      const s = el('span', 'cite');
      s.appendChild(node);
      const ids = cite[2];
      if (ids.some((id) => findings.failing.has(id))) {
        s.classList.add('fail');
        s.title = 'Judge: citation does not support this statement';
      }
      s.addEventListener('click', () => reveal($('passage-' + ids[0])));
      node = s;
    }
    if (cover(bold, a)) { const s = el('strong'); s.appendChild(node); node = s; }
    const cl = cover(claimRanges, a);
    if (cl) {
      const mk = el('mark', 'claim');
      mk.appendChild(node);
      mk.title = 'Unsupported claim ' + (cl[2] + 1) + ' (judge)';
      mk.addEventListener('click', () => reveal($('claimref-' + cl[2])));
      if (!anchored.has(cl[2])) { mk.id = 'claim-' + cl[2]; anchored.add(cl[2]); }
      node = mk;
    }
    container.appendChild(node);
    if (cl && b === cl[1]) container.appendChild(el('sup', 'claim-tag', cl[2] + 1));
  }
  return cited;
}

function renderResult(c, findings) {
  const pane = $('result');
  pane.replaceChildren(el('h3', 'pane-title', 'Result · model answer'));
  if (c.status !== 'response_generated') pane.appendChild(el('p', 'small muted', 'Run status: ' + c.status));

  const q = section(pane, 'Query', true, true);
  q.appendChild(el('div', 'prose small secondary', c.query || '(no query)'));

  const a = section(pane, 'Answer', false);
  a.appendChild(el('div', 'small muted', 'Model confidence ' + fmt(c.confidence) + ' · ' + fmt(c.elapsed, 1) + ' s'));
  const legend = el('div', 'legend small muted');
  legend.appendChild(el('mark', 'claim', 'highlight'));
  legend.appendChild(document.createTextNode(' unsupported claim (judge) · '));
  legend.appendChild(el('span', 'cite fail', '[n]'));
  legend.appendChild(document.createTextNode(' citation the judge could not verify · click either to jump'));
  a.appendChild(legend);
  const body = el('div', 'prose');
  const cited = renderAnswer(body, c.answer || '(no answer)', findings);
  a.appendChild(body);

  const p = section(pane, 'Evidence passages given to the model (' + c.passages.length + ')', false);
  if (!c.passages.length) p.appendChild(el('p', 'small muted', 'No passages recorded.'));
  c.passages.forEach((ps) => {
    const id = String(ps.citation_id);
    const d = el('details', 'passage');
    d.id = 'passage-' + id;
    const s = el('summary');
    s.appendChild(el('span', 'pid', '[' + id + ']'));
    s.appendChild(el('span', 'secondary', ps.source_path || ps.topic || ''));
    if (cited.has(id)) s.appendChild(badge(null, 'cited in answer'));
    const f = findings.failing.get(id);
    if (f) s.appendChild(badge('critical', 'not supported · run ' + f.join(', ')));
    d.appendChild(s);
    d.appendChild(el('div', 'prose small secondary', ps.text || ''));
    const url = c.sources[id];
    if (url && /^https:\/\//.test(url)) {
      const l = el('a', 'small', 'Open source document');
      l.href = url; l.target = '_blank'; l.rel = 'noopener';
      d.appendChild(l);
    }
    p.appendChild(d);
  });

  const sc = section(pane, 'Model self-critique', true, false);
  Object.entries(c.self_critique || {}).forEach(([k, v]) => {
    const row = el('div', 'small');
    row.appendChild(el('span', 'muted', k.replace(/_/g, ' ') + ': '));
    row.appendChild(document.createTextNode(Array.isArray(v) ? (v.length ? v.join(' | ') : '—') : show(v)));
    sc.appendChild(row);
  });

  const lg = section(pane, 'Engine log', true, false);
  lg.appendChild(el('pre', 'log', c.engine_log || ''));
}

function renderScore(c, findings) {
  const pane = $('score');
  pane.replaceChildren(el('h3', 'pane-title', 'Score · judge' + (META.judge_model ? ' (' + META.judge_model + ')' : '')));

  const head = el('div', 'case-head');
  head.appendChild(el('h2', null, c.id));
  head.appendChild(el('div', 'secondary', c.site));
  const pills = el('div', 'pills');
  ['setting_group', 'prior_rt', 'encounter_type'].forEach((k) => {
    if (c.labels[k]) pills.appendChild(badge(null, k.replace(/_/g, ' ') + ': ' + c.labels[k]));
  });
  c.flags.forEach((f) => pills.appendChild(badge('serious', f)));
  head.appendChild(pills);
  pane.appendChild(head);

  const tiles = el('div', 'tiles');
  METRICS.forEach(([key, label, max]) => {
    const t = el('div', 'tile');
    t.appendChild(el('div', 'label', label));
    const v = el('div', 'value', show(c.final[key]));
    v.appendChild(el('span', 'of', ' / ' + max));
    t.appendChild(v);
    t.appendChild(el('div', 'sub', 'runs ' + c.runs.map((r) => show(runValue(r, key))).join(' · ')));
    if (key === 'harm' && c.final.harm != null) {
      const fails = c.final.harm >= 2;
      const gate = el('div', 'sub');
      gate.appendChild(dot(fails ? 'critical' : 'good'));
      gate.appendChild(document.createTextNode(fails ? 'fails safety gate' : 'passes safety gate'));
      t.appendChild(gate);
    }
    tiles.appendChild(t);
  });
  FLAGS.forEach(([key, label]) => {
    const t = el('div', 'tile');
    t.appendChild(el('div', 'label', label));
    t.appendChild(el('div', 'value', show(c.final[key])));
    const [kind, text] = flagStatus(key, c.final[key]);
    const sub = el('div', 'sub');
    sub.appendChild(dot(kind));
    sub.appendChild(document.createTextNode(text));
    t.appendChild(sub);
    tiles.appendChild(t);
  });
  pane.appendChild(tiles);

  const rt = section(pane, 'Run 1 vs run 2', false);
  const table = el('table', 'runs');
  const hr = el('tr');
  hr.appendChild(el('th', null, 'Criterion'));
  c.runs.forEach((_, i) => hr.appendChild(el('th', null, 'Run ' + (i + 1))));
  const thead = el('thead');
  thead.appendChild(hr);
  table.appendChild(thead);
  const tbody = el('tbody');
  [...METRICS.map(([k, l]) => [k, l]), ...FLAGS].forEach(([key, label]) => {
    const tr = el('tr');
    const mc = el('td', 'metric', label);
    const vals = c.runs.map((r) => show(runValue(r, key)));
    if (new Set(vals).size > 1) { mc.appendChild(el('br')); mc.appendChild(badge('serious', 'runs differ')); }
    tr.appendChild(mc);
    c.runs.forEach((run) => {
      const s = run[key] || {};
      const td = el('td');
      td.appendChild(el('div', 'v', show(runValue(run, key))));
      if (s.reason) td.appendChild(el('div', 'reason', s.reason));
      if (key === 'citations_verifiable' && (s.failing_citations || []).length) {
        const line = el('div', 'reason');
        line.appendChild(document.createTextNode('Failing: '));
        s.failing_citations.forEach((id, j) => {
          if (j) line.appendChild(document.createTextNode(', '));
          const lk = el('span', 'cite', '[' + id + ']');
          lk.addEventListener('click', () => reveal($('passage-' + id)));
          line.appendChild(lk);
        });
        td.appendChild(line);
      }
      if (key === 'hallucination' && (s.fabricated || []).length) {
        td.appendChild(el('div', 'reason', 'Fabricated:'));
        const ul = el('ul', 'reason');
        s.fabricated.forEach((f) => ul.appendChild(el('li', null, f)));
        td.appendChild(ul);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  rt.appendChild(table);

  const cs = section(pane, 'Unsupported claims (' + findings.claims.length + ')', false);
  if (!findings.claims.length) cs.appendChild(el('p', 'small muted', 'None listed by the judge.'));
  const ol = el('ol', 'claims');
  findings.claims.forEach((cl, ci) => {
    const li = el('li');
    li.id = 'claimref-' + ci;
    li.appendChild(el('div', null, '“' + cl.text + '”'));
    li.appendChild(el('div', 'small muted',
      'run ' + cl.runs.join(', ') + (cl.found ? ' · click to find in answer' : ' · not found verbatim in answer')));
    if (cl.found) {
      li.classList.add('linked');
      li.addEventListener('click', () => reveal($('claim-' + ci)));
    }
    ol.appendChild(li);
  });
  cs.appendChild(ol);

  const rb = section(pane, 'Rubric', true, false);
  const dl = el('dl', 'rubric');
  Object.entries(META.rubric || {}).forEach(([k, v]) => {
    dl.appendChild(el('dt', null, k));
    dl.appendChild(el('dd', null, v));
  });
  rb.appendChild(dl);
}

const sel = $('case');
const tag = (v) => (v == null ? '–' : String(v));
CASES.forEach((c, i) => {
  const f = c.final || {};
  const n = c.flags.length;
  const o = el('option', null,
    c.id + ' · ' + c.site + ' · A' + tag(f.accuracy) + ' C' + tag(f.completeness) +
    ' G' + tag(f.guideline_adherence) + ' H' + tag(f.harm) + (n ? ' · ' + n + (n > 1 ? ' flags' : ' flag') : ''));
  o.value = String(i);
  sel.appendChild(o);
});

let current = 0;
function go(i) {
  current = (i + CASES.length) % CASES.length;
  const c = CASES[current];
  sel.value = String(current);
  $('pos').textContent = (current + 1) + ' / ' + CASES.length;
  history.replaceState(null, '', '#' + encodeURIComponent(c.id));
  const findings = judgeFindings(c);
  renderResult(c, findings);  // must run first: it marks which claims were found in the answer
  renderScore(c, findings);
  $('result').scrollTop = 0;
  $('score').scrollTop = 0;
}
sel.addEventListener('change', () => go(Number(sel.value)));
$('prev').addEventListener('click', () => go(current - 1));
$('next').addEventListener('click', () => go(current + 1));
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === 'ArrowLeft') go(current - 1);
  if (e.key === 'ArrowRight') go(current + 1);
});
const start = CASES.findIndex((c) => '#' + encodeURIComponent(c.id) === location.hash);
go(start >= 0 ? start : 0);
</script>
</body>
</html>
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores_dir", type=Path, help="A *_scores directory with scores.jsonl and manifest.json")
    parser.add_argument("--answers", type=Path, default=None,
                        help="Answers directory (default: the manifest's baseline_run, next to scores_dir)")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: scores_dir/review.html)")
    args = parser.parse_args(argv)

    manifest = loads((args.scores_dir / "manifest.json").read_text(encoding="utf-8"))
    answers_dir = args.answers or args.scores_dir.parent / manifest["baseline_run"]
    payload = build_payload(manifest, args.scores_dir, answers_dir)
    # "</" inside the embedded JSON would otherwise close the <script> tag early.
    data = dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.replace("__TITLE__", escape(payload["meta"]["title"])).replace("__DATA__", data)
    output = args.output or args.scores_dir / "review.html"
    output.write_text(html, encoding="utf-8")
    print(f"Wrote {output} ({len(payload['cases'])} cases, {output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
