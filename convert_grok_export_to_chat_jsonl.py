"""
Convert a Grok data export JSON into a chat JSONL dataset suitable for LLM fine-tuning.

Input (example): grok-export/prod-grok-backend.json
Output: JSONL where each line is one conversation:
  {
    "source": "grok",
    "conversation_id": "...",
    "title": "...",
    "created_at": "...",
    "messages": [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
  }
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Optional


Role = Literal["system", "user", "assistant"]


def _parse_time(s: Optional[str]) -> tuple[int, str]:
    """
    Return a sortable key + original string.
    Grok timestamps look like: 2026-04-09T07:11:34.833912Z
    """
    if not s:
        return (0, "")
    try:
        # Normalize trailing Z for fromisoformat compatibility
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return (int(dt.timestamp() * 1_000_000), s)
    except Exception:
        return (0, s)


def _role_from_sender(sender: Optional[str]) -> Optional[Role]:
    if not sender:
        return None
    s = sender.strip().lower()
    if s in {"human", "user"}:
        return "user"
    if s in {"assistant"}:
        return "assistant"
    if s in {"system"}:
        return "system"
    # Some exports use uppercase variants
    if sender.strip() == "ASSISTANT":
        return "assistant"
    return None


def _clean_content(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str
    sort_key: int


def _iter_conversation_messages(conv: dict[str, Any]) -> list[ChatMessage]:
    responses = conv.get("responses") or []
    out: list[ChatMessage] = []
    for item in responses:
        r = (item or {}).get("response") or {}
        role = _role_from_sender(r.get("sender"))
        content = _clean_content(r.get("message"))
        if not role or not content:
            continue
        sort_key, _ = _parse_time(r.get("create_time"))
        out.append(ChatMessage(role=role, content=content, sort_key=sort_key))

    out.sort(key=lambda m: m.sort_key)

    # Deduplicate identical consecutive role+content pairs.
    deduped: list[ChatMessage] = []
    for m in out:
        if deduped and deduped[-1].role == m.role and deduped[-1].content == m.content:
            continue
        deduped.append(m)
    return deduped


def _iter_conversations(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for c in data.get("conversations") or []:
        if isinstance(c, dict):
            yield c


def convert(in_path: Path, out_path: Path) -> dict[str, int]:
    data = json.loads(in_path.read_text(encoding="utf-8", errors="replace"))

    written = 0
    skipped_empty = 0
    total = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in _iter_conversations(data):
            total += 1
            meta = c.get("conversation") or {}
            conv_id = meta.get("id") or ""
            title = meta.get("title") or ""
            created_at = meta.get("create_time") or ""

            msgs = _iter_conversation_messages(c)
            if not msgs:
                skipped_empty += 1
                continue

            record = {
                "source": "grok",
                "conversation_id": conv_id,
                "title": title,
                "created_at": created_at,
                "messages": [{"role": m.role, "content": m.content} for m in msgs],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {"total_conversations": total, "written": written, "skipped_empty": skipped_empty}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default=str(Path("grok-export") / "prod-grok-backend.json"),
        help="Path to Grok export backend JSON",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default=str(Path("grok-export") / "grok_conversations.chat.jsonl"),
        help="Output JSONL path",
    )
    args = ap.parse_args()

    stats = convert(Path(args.in_path), Path(args.out_path))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

