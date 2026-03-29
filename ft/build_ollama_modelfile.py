"""Build an Ollama Modelfile from ft/data/train.jsonl and optionally `ollama create`.

This is Ollama-only (no PyTorch): it sets FROM + SYSTEM from your Cursor-derived
training JSONL so the local model picks up style/context hints. For real weight
changes, merge a LoRA to GGUF elsewhere and set OLLAMA_ADAPTER_GGUF.

Env:
  OLLAMA_MODEL          Base image for FROM (default: mistral:instruct)
  OLLAMA_ADAPTER_GGUF   Optional path to GGUF adapter for ADAPTER line
  DAMA_OLLAMA_NAME      Name for `ollama create` (default: dama2-cursor)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def _load_user_snippets(train_jsonl: Path, max_chars: int) -> str:
    if not train_jsonl.is_file():
        raise SystemExit(f"Missing {train_jsonl}. Run prepare_dataset.py first.")
    parts: list[str] = []
    total = 0
    with train_jsonl.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = row.get("messages") or []
            for m in msgs:
                if m.get("role") != "user":
                    continue
                t = (m.get("content") or "").strip()
                if len(t) < 8:
                    continue
                t = re.sub(r"\s+", " ", t)[:500]
                chunk = f"- {t}\n"
                if total + len(chunk) > max_chars:
                    return "".join(parts).strip()
                parts.append(chunk)
                total += len(chunk)
    return "".join(parts).strip()


def _escape_modelfile_system(text: str) -> str:
    # Modelfile SYSTEM """ ... """ — avoid closing delimiter run
    return text.replace('"""', "'''")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--train-file", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--from-model", default=None, help="FROM line (default: OLLAMA_MODEL)")
    p.add_argument("--max-system-chars", type=int, default=8000)
    p.add_argument(
        "--create",
        action="store_true",
        help="Run: ollama create <name> -f <out>",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Ollama model name (default: DAMA_OLLAMA_NAME or dama2-cursor)",
    )
    args = p.parse_args()

    ft_dir = Path(__file__).resolve().parent
    train = args.train_file or (ft_dir / "data" / "train.jsonl")
    out = args.out or (ft_dir / "data" / "ollama.Modelfile")

    base = args.from_model or os.environ.get("OLLAMA_MODEL", "mistral:instruct")
    adapter = os.environ.get("OLLAMA_ADAPTER_GGUF", "").strip()
    name = args.name or os.environ.get("DAMA_OLLAMA_NAME", "dama2-cursor")

    snippets = _load_user_snippets(train, args.max_system_chars)
    if not snippets:
        raise SystemExit("No user snippets found in train.jsonl.")

    system = _escape_modelfile_system(
        "You are a coding assistant. The user often works on topics reflected in these "
        "recent questions (for context only; answer the actual user message normally):\n\n"
        + snippets
    )

    lines = [f"FROM {base}"]
    if adapter:
        # Path as seen by Ollama (prefer absolute on Windows)
        ap = Path(adapter).resolve()
        if not ap.is_file():
            raise SystemExit(f"OLLAMA_ADAPTER_GGUF not a file: {ap}")
        lines.append(f"ADAPTER {ap.as_posix()}")
    lines.append('SYSTEM """')
    lines.append(system)
    lines.append('"""')

    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"Wrote {out} (FROM {base}" + (f", ADAPTER {adapter}" if adapter else "") + ")")

    if args.create:
        try:
            r = subprocess.run(
                ["ollama", "create", name, "-f", str(out)],
                check=False,
            )
        except FileNotFoundError:
            raise SystemExit(
                "`ollama` not found on PATH. Install Ollama and ensure `ollama` is available."
            ) from None
        if r.returncode != 0:
            raise SystemExit(f"ollama create failed with code {r.returncode}")
        print(f"Created Ollama model: {name}  (set OLLAMA_MODEL={name} for local_app.py)")


if __name__ == "__main__":
    main()
