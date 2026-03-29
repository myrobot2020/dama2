"""MVP: copy Agent-mode transcripts into `cursor chats/` for the fine-tune pipeline.

Cursor stores workspace metadata in workspaceStorage/*/state.vscdb (composer.composerData)
and full turn-by-turn logs under ~/.cursor/projects/<slug>/agent-transcripts/<uuid>/<uuid>.jsonl.

This script discovers the workspace DB for the repo, optionally filters by composer IDs from
the DB, and copies JSONL files so prepare_dataset.py can consume them.

Requires: Cursor has run Agent/Composer for this project (transcripts on disk).
May break if Cursor changes storage layout (copy known limitation in docstring).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse


def _workspace_path_from_json(data: dict) -> Path | None:
    folder = data.get("folder")
    if not folder or not isinstance(folder, str):
        return None
    if not folder.startswith("file://"):
        return None
    parsed = urlparse(folder)
    raw = unquote(parsed.path or "")
    if len(raw) >= 3 and raw[0] == "/" and raw[2] == ":":
        raw = raw[1:]
    try:
        return Path(raw).resolve()
    except OSError:
        return None


def _cursor_workspace_storage_base() -> Path | None:
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        if not appdata:
            return None
        return Path(appdata) / "Cursor" / "User" / "workspaceStorage"
    if sys.platform == "darwin":
        home = Path.home()
        return home / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"
    return Path.home() / ".config" / "Cursor" / "User" / "workspaceStorage"


def _find_workspace_state_db(repo_root: Path) -> Path | None:
    base = _cursor_workspace_storage_base()
    if base is None or not base.is_dir():
        return None
    target = repo_root.resolve()
    for child in base.iterdir():
        if not child.is_dir():
            continue
        wj = child / "workspace.json"
        if not wj.is_file():
            continue
        try:
            data = json.loads(wj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        wp = _workspace_path_from_json(data)
        if wp == target:
            db = child / "state.vscdb"
            if db.is_file():
                return db
    return None


def _repo_to_cursor_slug(repo_root: Path) -> str:
    p = repo_root.resolve()
    drive = p.drive.replace(":", "").lower()
    tail = [drive] + [part for part in p.parts[1:]]
    return "-".join(tail)


def _composer_ids_from_db(db_path: Path) -> set[str]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT value FROM ItemTable WHERE key = ?", ("composer.composerData",)
        )
        row = cur.fetchone()
        if not row:
            return set()
        raw = row[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        data = json.loads(raw)
    except (sqlite3.Error, json.JSONDecodeError, TypeError):
        return set()
    finally:
        conn.close()

    out: set[str] = set()
    for c in data.get("allComposers") or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("composerId")
        if isinstance(cid, str) and len(cid) > 10:
            out.add(cid)
    sel = data.get("selectedComposerId")
    if isinstance(sel, str):
        out.add(sel)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Project root (default: parent of ft/)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write db_<uuid>.jsonl (default: <repo>/cursor chats)",
    )
    p.add_argument(
        "--slug",
        default=None,
        help="Cursor projects folder slug (default: derived from repo path)",
    )
    p.add_argument(
        "--mode",
        choices=("workspace", "all"),
        default="workspace",
        help="workspace: only composer IDs listed in workspace state.vscdb; "
        "all: every transcript under agent-transcripts for this project",
    )
    p.add_argument(
        "--max-age-hours",
        type=float,
        default=None,
        metavar="H",
        help="Skip transcript files older than H hours (mtime).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing db_*.jsonl in output dir before copy (avoids stale sessions).",
    )
    args = p.parse_args()

    ft_dir = Path(__file__).resolve().parent
    repo_root = (args.repo_root or ft_dir.parent).resolve()
    out_dir = args.output_dir or (repo_root / "cursor chats")
    slug = args.slug or _repo_to_cursor_slug(repo_root)
    agents_root = Path.home() / ".cursor" / "projects" / slug / "agent-transcripts"

    if not agents_root.is_dir():
        raise SystemExit(
            f"Agent transcripts folder not found:\n  {agents_root}\n"
            "Open this repo in Cursor and run at least one Agent chat, or pass --slug if "
            "your projects path uses a different slug."
        )

    state_db = _find_workspace_state_db(repo_root)
    if args.mode == "workspace" and state_db is None:
        raise SystemExit(
            "Could not find workspace state.vscdb for this repo under "
            "%APPDATA%\\Cursor\\User\\workspaceStorage. Use --mode all to copy every "
            "transcript without DB filtering."
        )

    allow_ids: set[str] | None = None
    if args.mode == "workspace" and state_db is not None:
        allow_ids = _composer_ids_from_db(state_db)
        if not allow_ids:
            print(
                "Warning: composer.composerData had no composer IDs; falling back to --mode all",
                file=sys.stderr,
            )
            allow_ids = None

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean and not args.dry_run:
        for stale in out_dir.glob("db_*.jsonl"):
            try:
                stale.unlink()
            except OSError:
                pass
    cutoff = None
    if args.max_age_hours is not None and args.max_age_hours > 0:
        cutoff = time.time() - args.max_age_hours * 3600.0

    copied = 0
    skipped = 0
    for sub in sorted(agents_root.iterdir()):
        if not sub.is_dir():
            continue
        cid = sub.name
        if allow_ids is not None and cid not in allow_ids:
            skipped += 1
            continue
        src = sub / f"{cid}.jsonl"
        if not src.is_file():
            skipped += 1
            continue
        if cutoff is not None and src.stat().st_mtime < cutoff:
            skipped += 1
            continue
        dest = out_dir / f"db_{cid}.jsonl"
        if args.dry_run:
            print(f"copy {src} -> {dest}")
        else:
            shutil.copy2(src, dest)
        copied += 1

    print(
        f"export_cursor_db: copied {copied} transcript(s) -> {out_dir}"
        + (f" (skipped {skipped})" if skipped else "")
        + (f"\n  workspace DB: {state_db}" if state_db else "")
        + f"\n  agents root: {agents_root}"
    )
    if copied == 0:
        raise SystemExit(
            "No transcripts copied. Try --mode all, fix --slug, or remove --max-age-hours."
        )


if __name__ == "__main__":
    main()
