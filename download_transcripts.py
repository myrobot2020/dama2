"""
Download all Anguttara Nikaya lecture transcripts from the YouTube playlist.

Prerequisites:
    pip install yt-dlp

Usage:
    python download_transcripts.py

This will download auto-generated subtitles for each video in the playlist
and save them as numbered .txt files in the current directory.
"""

import re
import subprocess
import sys
from pathlib import Path

PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLD8I9vPmsYXxR_Qt36EbquMkYTOZbXWpM"
OUTPUT_DIR = Path(__file__).resolve().parent


def check_yt_dlp():
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("yt-dlp is not installed. Install it with:\n  pip install yt-dlp")
        sys.exit(1)


def clean_subtitle_text(vtt_path: Path) -> str:
    """Strip VTT headers, timestamps, and dedup repeated lines."""
    raw = vtt_path.read_text(encoding="utf-8", errors="ignore")
    lines = []
    prev = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->", line):
            continue
        if re.match(r"^<\d{2}:\d{2}:\d{2}\.\d{3}>", line):
            line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", "", line).strip()
        cleaned = re.sub(r"<[^>]+>", "", line).strip()
        if cleaned and cleaned != prev:
            lines.append(cleaned)
            prev = cleaned
    return " ".join(lines)


def main():
    check_yt_dlp()

    tmp_dir = OUTPUT_DIR / "_subtitle_tmp"
    tmp_dir.mkdir(exist_ok=True)

    print(f"Downloading subtitles from playlist:\n  {PLAYLIST_URL}\n")

    result = subprocess.run(
        [
            "yt-dlp",
            "--flat-playlist",
            "--print", "%(playlist_index)s|||%(title)s|||%(id)s",
            PLAYLIST_URL,
        ],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        print(f"Failed to list playlist:\n{result.stderr}")
        sys.exit(1)

    entries = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|||")
        if len(parts) == 3:
            idx, title, vid_id = parts
            entries.append((idx.strip(), title.strip(), vid_id.strip()))

    print(f"Found {len(entries)} videos in playlist.\n")

    for idx_str, title, vid_id in entries:
        idx = int(idx_str) if idx_str.isdigit() else 0
        safe_title = re.sub(r'[<>:"/\\|?*]', '', title)[:120]
        out_name = f"{idx:03d}_{safe_title} by Bhante Hye Dhammavuddho Mahathera.txt"
        out_path = OUTPUT_DIR / out_name

        if out_path.exists():
            print(f"  [{idx:03d}] Already exists, skipping: {out_name}")
            continue

        print(f"  [{idx:03d}] Downloading subtitles: {title}")

        sub_result = subprocess.run(
            [
                "yt-dlp",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--skip-download",
                "--output", str(tmp_dir / f"{vid_id}"),
                f"https://www.youtube.com/watch?v={vid_id}",
            ],
            capture_output=True, text=True,
        )

        vtt_files = list(tmp_dir.glob(f"{vid_id}*.vtt"))
        if not vtt_files:
            srt_files = list(tmp_dir.glob(f"{vid_id}*.srt"))
            if srt_files:
                vtt_files = srt_files

        if not vtt_files:
            print(f"         No subtitles found for {title}")
            continue

        text = clean_subtitle_text(vtt_files[0])
        if text.strip():
            out_path.write_text(text, encoding="utf-8")
            print(f"         Saved: {out_name} ({len(text)} chars)")
        else:
            print(f"         Empty transcript for {title}")

        for f in tmp_dir.iterdir():
            f.unlink()

    tmp_dir.rmdir()
    print(f"\nDone. Transcripts saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
