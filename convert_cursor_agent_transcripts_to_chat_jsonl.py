"""
Convert Cursor agent transcript JSONL files into a chat JSONL dataset.

Cursor agent transcripts (observed format):
  Each line is a JSON object like:
    {"role":"user"|"assistant", "message":{"content":[{"type":"text","text":"..."}, {"type":"tool_use", ...}]}}

We extract ONLY the text blocks (type == "text") and ignore tool_use blocks.

Output JSONL: one conversation per transcript file
  {
    "source": "cursor",
    "conversation_id": "<uuid>",
    "path": "<full path to transcript jsonl>",
    "messages": [{"role":"user","content":"..."}, ...]
  }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


Role = Literal["system", "user", "assistant"]


def _as_role(role: Optional[str]) -> Optional[Role]:
    if not role:
        return None
    r = role.strip().lower()
    if r in {"user", "assistant", "system"}:
        return r  # type: ignore[return-value]
    return None


def _extract_text_from_message(obj: dict[str, Any]) -> str:
    msg = obj.get("message") or {}
    content = msg.get("content") or []
    parts: list[str] = []
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            t = item.get("text")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n\n".join(parts).strip()


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


def _iter_messages_from_transcript_jsonl(path: Path) -> list[ChatMessage]:
    msgs: list[ChatMessage] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            role = _as_role(obj.get("role"))
            if not role:
                continue
            text = _extract_text_from_message(obj)
            if not text:
                continue
            msgs.append(ChatMessage(role=role, content=text))

    # Dedup identical consecutive role+content pairs (common when retries happen).
    deduped: list[ChatMessage] = []
    for m in msgs:
        if deduped and deduped[-1].role == m.role and deduped[-1].content == m.content:
            continue
        deduped.append(m)
    return deduped


def _iter_transcript_files(root: Path) -> Iterable[Path]:
    # Layout: agent-transcripts/<uuid>/<uuid>.jsonl
    if not root.exists():
        return
    for p in root.glob("*/*.jsonl"):
        if p.is_file():
            yield p


def convert(root: Path, out_path: Path) -> dict[str, int]:
    total_files = 0
    written = 0
    skipped_empty = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for p in sorted(_iter_transcript_files(root)):
            total_files += 1
            conv_id = p.parent.name
            msgs = _iter_messages_from_transcript_jsonl(p)
            if not msgs:
                skipped_empty += 1
                continue

            record = {
                "source": "cursor",
                "conversation_id": conv_id,
                "path": str(p),
                "messages": [{"role": m.role, "content": m.content} for m in msgs],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {"total_transcripts": total_files, "written": written, "skipped_empty": skipped_empty}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default=str(
            Path.home()
            / ".cursor"
            / "projects"
            / "c-Users-ADMIN-Desktop-dama3"
            / "agent-transcripts"
        ),
        help="Root agent-transcripts directory",
    )
    ap.add_argument(
        "--out",
        default=str(Path("cursor-export") / "cursor_agent_transcripts.chat.jsonl"),
        help="Output JSONL path",
    )
    args = ap.parse_args()

    stats = convert(Path(args.root), Path(args.out))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

