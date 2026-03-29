"""Build an Ollama Modelfile from ft/data/train.jsonl and optionally `ollama create`.

Weight behavior (be precise):
  • SYSTEM-only (default): **no** weight updates — same base weights as `FROM`;
    only extra context / instructions at inference (not partial fine-tuning).
  • `OLLAMA_ADAPTER_GGUF` set: **partial** weight adjustment — Ollama applies a
    LoRA-style adapter on top of `FROM` (Safetensors adapter **directory** or
    **GGUF file** per Ollama docs); base weights stay frozen aside from that overlay.

Train Cursor chats → JSONL does **not** by itself produce weights; run
`train_sft.py` and point `OLLAMA_ADAPTER_GGUF` at `ft/runs/<model_stem>/` or a
`.gguf` adapter from an external converter (e.g. llama.cpp).

Env:
  OLLAMA_MODEL          Base image for FROM (default: qwen2.5:0.5b-instruct, aligned with DAMA_HF_MODEL)
  OLLAMA_ADAPTER_GGUF   Optional path: GGUF file **or** Safetensors adapter **directory**
  DAMA_OLLAMA_NAME      Name for `ollama create` (default: dama2-cursor)
  DAMA_ADAPTER_PLUS_SNIPPETS  If 1 and adapter set, also add snippet SYSTEM (usually omit; adapter only)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Default Ollama tag aligned with train_sft.py / DAMA_HF_MODEL (Qwen2.5 0.5B Instruct).
_DEFAULT_FROM_OLLAMA = "qwen2.5:0.5b-instruct"


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
    return text.replace('"""', "'''")


def _resolve_adapter(adapter_raw: str) -> Path:
    ap = Path(adapter_raw).resolve()
    if ap.is_dir():
        cfg = ap / "adapter_config.json"
        if not cfg.is_file():
            raise SystemExit(
                f"OLLAMA_ADAPTER_GGUF directory missing adapter_config.json: {ap}"
            )
        return ap
    if ap.is_file():
        return ap
    raise SystemExit(f"OLLAMA_ADAPTER_GGUF not a file or directory: {ap}")


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

    base = args.from_model or os.environ.get("OLLAMA_MODEL", _DEFAULT_FROM_OLLAMA)
    adapter_raw = os.environ.get("OLLAMA_ADAPTER_GGUF", "").strip()
    name = args.name or os.environ.get("DAMA_OLLAMA_NAME", "dama2-cursor")
    adapter_plus = os.environ.get("DAMA_ADAPTER_PLUS_SNIPPETS", "").strip() == "1"

    adapter_path: Path | None = None
    if adapter_raw:
        adapter_path = _resolve_adapter(adapter_raw)

    need_snippets = (not adapter_path) or adapter_plus
    snippets = ""
    if need_snippets:
        snippets = _load_user_snippets(train, args.max_system_chars)

    if adapter_path and not adapter_plus:
        system = "You are a helpful assistant."
    elif adapter_path and adapter_plus:
        if not snippets:
            raise SystemExit("No user snippets found in train.jsonl.")
        system = _escape_modelfile_system(
            "You are a coding assistant. The user often works on topics reflected in these "
            "recent questions (for context only; answer the actual user message normally):\n\n"
            + snippets
        )
    elif snippets:
        system = _escape_modelfile_system(
            "You are a coding assistant. The user often works on topics reflected in these "
            "recent questions (for context only; answer the actual user message normally):\n\n"
            + snippets
        )
    else:
        raise SystemExit("No user snippets found in train.jsonl.")

    lines = [f"FROM {base}"]
    if adapter_path:
        lines.append(f"ADAPTER {adapter_path.as_posix()}")
    lines.append('SYSTEM """')
    lines.append(system)
    lines.append('"""')

    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    adisplay = adapter_raw if adapter_raw else ""
    print(f"Wrote {out} (FROM {base}" + (f", ADAPTER {adisplay}" if adisplay else "") + ")")
    if adapter_path:
        kind = "dir" if adapter_path.is_dir() else "file"
        print(
            f"Weights: partial adjustment via ADAPTER ({kind}).",
            file=sys.stderr,
        )
    else:
        print(
            "Weights: unchanged — SYSTEM only adds context; no partial fine-tuning.",
            file=sys.stderr,
        )

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
