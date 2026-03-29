"""
Chunk a transcript, label sutta_reading vs teacher_commentary via Ollama, emit tabbed HTML.

  python tag_sutta_commentary_ollama.py --input "001_....txt" --output static/book1_001_layers.html
  python tag_sutta_commentary_ollama.py --no-ollama
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

SUTTA_START = [
    "thus have i heard",
    "on a certain occasion",
    "the exalted one",
    "the buddha said",
    "monks i know",
]

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "llama3.2"


def word_chunks(text: str, words_per_chunk: int = 55) -> list[str]:
    words = text.split()
    out: list[str] = []
    for i in range(0, len(words), words_per_chunk):
        out.append(" ".join(words[i : i + words_per_chunk]))
    return out if out else [text]


def heuristic_label(chunk: str) -> tuple[str, str]:
    s = chunk.strip().lower()
    for start in SUTTA_START:
        if s.startswith(start):
            return "sutta_reading", "formal"
    for start in SUTTA_START:
        if f". {start}" in s or f"? {start}" in s:
            return "sutta_reading", "formal"
    return "teacher_commentary", "colloquial"


def _extract_json_array(raw: str) -> list[dict[str, Any]]:
    raw = raw.strip()
    m = re.search(r"\[[\s\S]*\]", raw)
    if not m:
        raise ValueError("no JSON array in model output")
    return json.loads(m.group(0))


def ollama_tag_batch(
    model: str,
    indexed_chunks: list[tuple[int, str]],
    timeout_s: int = 120,
) -> list[dict[str, Any]]:
    lines = "\n".join(f"{i}: {t[:1200]}" for i, t in indexed_chunks)
    system = (
        "You label spans from a Theravada lecture transcript in English. "
        "sutta_reading = formal translation/narration of the Buddha's words or sutta story "
        "(third person, liturgical). "
        "teacher_commentary = the speaker explaining in their own voice. "
        "Return ONLY a JSON array, one object per input line, keys: i (int), layer "
        "(sutta_reading|teacher_commentary), formality (formal|colloquial)."
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Label each span:\n{lines}"},
        ],
        "stream": False,
        "options": {"temperature": 0.1},
    }
    r = requests.post(OLLAMA_URL, json=body, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    content = (data.get("message") or {}).get("content") or ""
    return _extract_json_array(content)


def merge_runs(
    chunks: list[str],
    labels: list[tuple[str, str]],
) -> tuple[str, str]:
    sutta_parts: list[str] = []
    comm_parts: list[str] = []
    for ch, (layer, _fm) in zip(chunks, labels):
        if layer == "sutta_reading":
            sutta_parts.append(ch)
        else:
            comm_parts.append(ch)
    return " ".join(sutta_parts), " ".join(comm_parts)


def build_html(
    title: str,
    source_path: str,
    model_note: str,
    chunks: list[str],
    labels: list[tuple[str, str]],
    sutta_merged: str,
    commentary_merged: str,
) -> str:
    rows = []
    for i, (ch, (layer, fm)) in enumerate(zip(chunks, labels)):
        rows.append(
            "<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td></tr>"
            % (
                i,
                html.escape(layer),
                html.escape(fm),
                html.escape(ch[:220] + ("…" if len(ch) > 220 else "")),
            )
        )
    table = (
        "<table class='grid'><thead><tr><th>#</th><th>layer</th><th>formality</th>"
        "<th>snippet</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    def tab_body(text: str) -> str:
        body = text if text.strip() else "(none)"
        return f"<div class='body-scroll'><p>{html.escape(body)}</p></div>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <style>
    :root {{ --bg: #0f1419; --panel: #1a222d; --text: #e7ecf1; --muted: #8b98a5; --accent: #c9a227; --tab: #243040; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", system-ui, sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; }}
    header {{ padding: 1rem 1.25rem; border-bottom: 1px solid #2a3544; background: var(--panel); }}
    header h1 {{ margin: 0 0 0.35rem 0; font-size: 1.15rem; font-weight: 600; }}
    header .meta {{ color: var(--muted); font-size: 0.85rem; }}
    .wrap {{ padding: 1rem 1.25rem 2rem; max-width: 960px; margin: 0 auto; }}
    .tabs {{ display: flex; gap: 0; border-bottom: 2px solid #2a3544; margin-bottom: 0; }}
    .tabs button {{ background: var(--tab); color: var(--muted); border: none; padding: 0.65rem 1.1rem; cursor: pointer; font-size: 0.95rem; border-radius: 6px 6px 0 0; margin-right: 4px; }}
    .tabs button.active {{ background: var(--panel); color: var(--text); border-bottom: 2px solid var(--accent); margin-bottom: -2px; font-weight: 600; }}
    .panel {{ display: none; background: var(--panel); border: 1px solid #2a3544; border-top: none; border-radius: 0 0 8px 8px; padding: 0.75rem 1rem 1rem; }}
    .panel.active {{ display: block; }}
    .body-scroll {{ max-height: 62vh; overflow: auto; line-height: 1.55; font-size: 0.92rem; }}
    .body-scroll p {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
    table.grid {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    table.grid th, table.grid td {{ border: 1px solid #2a3544; padding: 0.35rem 0.5rem; vertical-align: top; }}
    table.grid th {{ background: #243040; color: var(--muted); text-align: left; }}
    .stats {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 0.75rem; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="meta">Source: {html.escape(source_path)} · {html.escape(model_note)}</div>
  </header>
  <div class="wrap">
    <div class="stats">
      Chunks: {len(chunks)} · Sutta chars: {len(sutta_merged):,} · Commentary chars: {len(commentary_merged):,}
    </div>
    <div class="tabs" role="tablist">
      <button type="button" class="active" data-tab="sutta">Sutta</button>
      <button type="button" data-tab="commentary">Commentary</button>
      <button type="button" data-tab="chunks">Chunks</button>
    </div>
    <div id="panel-sutta" class="panel active" role="tabpanel">{tab_body(sutta_merged)}</div>
    <div id="panel-commentary" class="panel" role="tabpanel">{tab_body(commentary_merged)}</div>
    <div id="panel-chunks" class="panel" role="tabpanel"><div class="body-scroll">{table}</div></div>
  </div>
  <script>
    document.querySelectorAll('.tabs button').forEach(function(btn) {{
      btn.addEventListener('click', function() {{
        var id = btn.getAttribute('data-tab');
        document.querySelectorAll('.tabs button').forEach(function(b) {{ b.classList.remove('active'); }});
        document.querySelectorAll('.panel').forEach(function(p) {{ p.classList.remove('active'); }});
        btn.classList.add('active');
        document.getElementById('panel-' + id).classList.add('active');
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=Path("001_Anguttara Nikaya Book 1 - Introduction and Overview by Bhante Hye Dhammavuddho Mahathera.txt"),
    )
    ap.add_argument("--output", type=Path, default=Path("static/book1_001_layers.html"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--words", type=int, default=55)
    ap.add_argument("--no-ollama", action="store_true")
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8", errors="replace").strip()
    chunks = word_chunks(text, args.words)
    labels: list[tuple[str, str]] = []

    if args.no_ollama:
        for ch in chunks:
            labels.append(heuristic_label(ch))
        model_note = "Heuristic only (SUTTA_START); no Ollama"
    else:
        batch_size = max(1, args.batch)
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            indexed = [(i, chunks[i]) for i in range(start, end)]
            try:
                arr = ollama_tag_batch(args.model, indexed)
                by_i = {int(x["i"]): x for x in arr}
                for i in range(start, end):
                    obj = by_i.get(i) or {}
                    layer = obj.get("layer", "teacher_commentary")
                    if layer not in ("sutta_reading", "teacher_commentary"):
                        layer = "teacher_commentary"
                    fm = obj.get("formality", "colloquial")
                    if fm not in ("formal", "colloquial"):
                        fm = "colloquial"
                    labels.append((layer, fm))
            except Exception as e:
                print(f"Ollama batch {start}-{end} failed ({e}); heuristic.", file=sys.stderr)
                for i in range(start, end):
                    labels.append(heuristic_label(chunks[i]))
        model_note = f"Ollama {args.model}; failed batches used heuristics"

    while len(labels) < len(chunks):
        labels.append(heuristic_label(chunks[len(labels)]))
    labels = labels[: len(chunks)]

    sutta_merged, commentary_merged = merge_runs(chunks, labels)
    title = f"Sutta vs commentary — {args.input.name}"
    out = build_html(
        title,
        str(args.input),
        model_note,
        chunks,
        labels,
        sutta_merged,
        commentary_merged,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(out, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
