"""Render a baseline results.jsonl as a local, readable HTML viewer.

No network calls. Output stays on disk next to the input file.
"""

import argparse
from html import escape
from json import dumps, loads
from pathlib import Path


def load_records(path):
    with path.open("r", encoding="utf-8") as handle:
        return [loads(line) for line in handle if line.strip()]


def render_chunk(chunk):
    return f"""
      <div class="chunk">
        <div class="chunk-meta">#{chunk.get('faiss_id')} · score {chunk.get('similarity_score', 0):.3f} · {escape(str(chunk.get('path', '')))}</div>
        <pre class="chunk-text">{escape(chunk.get('text', ''))}</pre>
      </div>"""


def render_subtask(task):
    chunks_html = "".join(render_chunk(c) for c in task.get("retrieved_chunks", []))
    return f"""
    <details class="subtask">
      <summary>{escape(task.get('task_id', ''))} — {escape(task.get('description', ''))} <span class="status">[{escape(task.get('status', ''))}]</span></summary>
      <div class="subtask-body">
        <p><strong>Search query:</strong> {escape(task.get('query', ''))}</p>
        <p><strong>Result:</strong> {escape(task.get('result', ''))}</p>
        <details class="chunks">
          <summary>Retrieved chunks ({len(task.get('retrieved_chunks', []))})</summary>
          {chunks_html}
        </details>
      </div>
    </details>"""


def render_citation(c):
    url = c.get("source_url", "")
    label = escape(c.get("source_path", ""))
    link = f'<a href="{escape(url)}" target="_blank" rel="noopener">{label}</a>' if url else label
    return f'<li>[{c.get("citation_id")}] {link} <span class="score">score {c.get("relevance_score", 0):.3f}</span></li>'


def render_case(record):
    resp = record.get("response") or {}
    plan = resp.get("task_plan") or {}
    sub_tasks = plan.get("sub_tasks") or []
    citations = resp.get("citations") or []
    critique = resp.get("self_critique") or {}
    trace = resp.get("reasoning_trace") or []

    subtasks_html = "".join(render_subtask(t) for t in sub_tasks)
    citations_html = "".join(render_citation(c) for c in citations)
    trace_html = "\n".join(escape(line) for line in trace)
    critique_html = escape(dumps(critique, ensure_ascii=False, indent=2))

    return f"""
  <section class="case" id="{escape(record.get('case_id', ''))}">
    <h2>{escape(record.get('case_id', ''))} <span class="status">{escape(record.get('status', ''))}</span></h2>
    <p class="elapsed">{record.get('elapsed_seconds', 0):.1f}s</p>

    <details open class="block">
      <summary>Query</summary>
      <pre class="wrap">{escape(record.get('query', ''))}</pre>
    </details>

    <details open class="block">
      <summary>Answer <span class="score">confidence {resp.get('confidence_score', 0):.2f}</span></summary>
      <pre class="wrap">{escape(resp.get('answer', ''))}</pre>
    </details>

    <details class="block">
      <summary>Citations ({len(citations)})</summary>
      <ol>{citations_html}</ol>
    </details>

    <details class="block">
      <summary>Self-critique</summary>
      <pre class="wrap">{critique_html}</pre>
    </details>

    <details class="block">
      <summary>Sub-tasks / retrieval plan ({len(sub_tasks)})</summary>
      {subtasks_html}
    </details>

    <details class="block">
      <summary>Engine log / reasoning trace</summary>
      <pre class="wrap mono">{trace_html}</pre>
    </details>
  </section>"""


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Results viewer — {title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f7f7f8; color: #1c1c1e; }}
  header {{ padding: 16px 24px; background: #1c1c1e; color: #fff; position: sticky; top: 0; z-index: 10; }}
  header a {{ color: #9ecbff; margin-right: 12px; text-decoration: none; font-size: 14px; }}
  main {{ max-width: 980px; margin: 0 auto; padding: 24px; }}
  .case {{ background: #fff; border: 1px solid #e2e2e4; border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }}
  .case h2 {{ margin-top: 0; }}
  .status {{ font-size: 12px; font-weight: 500; color: #4a4a4f; background: #eee; padding: 2px 8px; border-radius: 6px; }}
  .elapsed {{ color: #6b6b70; font-size: 13px; margin-top: -8px; }}
  .block {{ margin: 10px 0; border: 1px solid #ececef; border-radius: 8px; padding: 6px 12px; }}
  .block summary {{ cursor: pointer; font-weight: 600; padding: 6px 0; }}
  .score {{ font-size: 12px; color: #6b6b70; font-weight: 400; }}
  pre.wrap {{ white-space: pre-wrap; word-wrap: break-word; font-family: inherit; font-size: 14px; line-height: 1.5; }}
  pre.mono {{ font-family: ui-monospace, Menlo, monospace; font-size: 12px; }}
  .subtask {{ border: 1px solid #e8e8ea; border-radius: 6px; margin: 8px 0; padding: 6px 10px; background: #fafafa; }}
  .subtask summary {{ cursor: pointer; font-size: 14px; }}
  .subtask-body {{ margin-top: 8px; font-size: 13px; }}
  .chunks summary {{ font-size: 12px; color: #6b6b70; cursor: pointer; }}
  .chunk {{ border-top: 1px dashed #ddd; padding: 6px 0; }}
  .chunk-meta {{ font-size: 11px; color: #8a8a8f; }}
  .chunk-text {{ white-space: pre-wrap; font-family: inherit; font-size: 12.5px; margin: 4px 0; }}
  ol {{ font-size: 13px; padding-left: 20px; }}
</style>
</head>
<body>
<header>
  <strong>{title}</strong>
  {nav}
</header>
<main>
{cases}
</main>
</body>
</html>"""


def build_html(records, title):
    nav = " ".join(f'<a href="#{escape(r.get("case_id", ""))}">{escape(r.get("case_id", ""))}</a>' for r in records)
    cases = "".join(render_case(r) for r in records)
    return PAGE_TEMPLATE.format(title=escape(title), nav=nav, cases=cases)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Path to a results.jsonl file")
    parser.add_argument("--output", type=Path, default=None, help="Output HTML path (default: alongside input)")
    args = parser.parse_args(argv)

    records = load_records(args.results)
    output = args.output or args.results.with_name("results_viewer.html")
    output.write_text(build_html(records, args.results.parent.name), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
