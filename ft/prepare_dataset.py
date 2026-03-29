"""Convert Cursor *.jsonl exports to SFT JSONL with a messages[] field."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


def _extract_text_from_content(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and "text" in block:
                parts.append(str(block["text"]))
            elif "text" in block:
                parts.append(str(block["text"]))
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        inner = content.get("content")
        if inner is not None:
            return _extract_text_from_content(inner)
    return ""


def _message_text(record: dict) -> tuple[str | None, str]:
    role = record.get("role")
    if role not in ("user", "assistant"):
        return None, ""
    msg = record.get("message") or {}
    raw = _extract_text_from_content(msg.get("content"))
    return role, raw.strip()


def _strip_attached_files(text: str) -> str:
    return re.sub(
        r"<attached_files>[\s\S]*?</attached_files>\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )


def _strip_user_query_tags(text: str) -> str:
    m = re.search(r"<user_query>\s*([\s\S]*?)\s*</user_query>", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _clean_user_text(text: str, strip_attachments: bool) -> str:
    if strip_attachments:
        text = _strip_attached_files(text)
    text = _strip_user_query_tags(text)
    return text.strip()


def jsonl_to_messages(path: Path, strip_attachments: bool) -> list[dict]:
    turns: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        role, text = _message_text(record)
        if role is None or not text:
            continue
        if role == "user":
            text = _clean_user_text(text, strip_attachments)
        if not text:
            continue
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] = turns[-1]["content"] + "\n\n" + text
        else:
            turns.append({"role": role, "content": text})
    while turns and turns[0]["role"] != "user":
        turns = turns[1:]
    out: list[dict] = []
    i = 0
    while i < len(turns):
        if turns[i]["role"] != "user":
            i += 1
            continue
        user = turns[i]
        i += 1
        ap: list[str] = []
        while i < len(turns) and turns[i]["role"] == "assistant":
            ap.append(turns[i]["content"])
            i += 1
        if ap:
            out.append({"role": "user", "content": user["content"]})
            out.append({"role": "assistant", "content": "\n\n".join(ap)})
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build SFT JSONL from Cursor chat exports.")
    p.add_argument("--input-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--val-output", type=Path, default=None)
    p.add_argument("--val-files", type=int, default=1)
    p.add_argument("--no-strip-attachments", action="store_true")
    p.add_argument("--min-user-chars", type=int, default=3)
    p.add_argument("--min-assistant-chars", type=int, default=20)
    p.add_argument(
        "--max-file-age-hours",
        type=float,
        default=None,
        metavar="H",
        help="Only include *.jsonl modified within the last H hours.",
    )
    args = p.parse_args()

    ft_dir = Path(__file__).resolve().parent
    repo_root = ft_dir.parent
    input_dir = args.input_dir or (repo_root / "cursor chats")
    output = args.output or (ft_dir / "data" / "train.jsonl")
    strip = not args.no_strip_attachments

    if not input_dir.is_dir():
        raise SystemExit(f"Input dir not found: {input_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(input_dir.glob("*.jsonl"))
    if args.max_file_age_hours is not None and args.max_file_age_hours > 0:
        cutoff = time.time() - args.max_file_age_hours * 3600.0
        files = [f for f in files if f.stat().st_mtime >= cutoff]
        if not files:
            raise SystemExit(
                f"No .jsonl in {input_dir} modified within {args.max_file_age_hours}h"
            )
    if not files:
        raise SystemExit(f"No .jsonl in {input_dir}")

    val_n = max(0, min(args.val_files, len(files) - 1)) if len(files) > 1 else 0
    train_files = files[:-val_n] if val_n else files
    val_files = files[-val_n:] if val_n else []

    def build_rows(paths: list[Path]) -> list[dict]:
        rows: list[dict] = []
        for fp in paths:
            messages = jsonl_to_messages(fp, strip)
            filtered: list[dict] = []
            j = 0
            while j + 1 < len(messages):
                u, a = messages[j], messages[j + 1]
                j += 2
                if len(u["content"]) < args.min_user_chars:
                    continue
                if len(a["content"]) < args.min_assistant_chars:
                    continue
                filtered.extend([u, a])
            if len(filtered) >= 2:
                rows.append({"messages": filtered})
        return rows

    train_rows = build_rows(train_files)
    val_rows = build_rows(val_files) if val_files else []
    if not train_rows:
        raise SystemExit("No training examples after filtering.")

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(output, train_rows)
    print(f"Wrote {len(train_rows)} examples -> {output}")
    if val_rows and args.val_output:
        args.val_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.val_output, val_rows)
        print(f"Wrote {len(val_rows)} -> {args.val_output}")


if __name__ == "__main__":
    main()
