"""
Fetch public web pages and save readable plain text (no browser / no JS execution).

Dependencies:
    pip install requests beautifulsoup4

Usage:
    python scrape_web.py https://example.com/article
    python scrape_web.py https://a.com https://b.com --out-dir scraped
    python scrape_web.py https://example.com --selector "article" --out article.txt

Static HTML only. Pages that need JavaScript to render will look empty unless you
use another tool (e.g. playwright). Be polite: obey robots.txt and site terms,
and use --delay when fetching many URLs.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

DEFAULT_UA = (
    "Mozilla/5.0 (compatible; dama2-scraper/1.0; +https://example.local) "
    "Python-requests"
)


def url_to_basename(url: str) -> str:
    p = urlparse(url)
    base = f"{p.netloc}{p.path or '/'}".strip("/").replace("/", "_")
    base = re.sub(r'[^\w\-.]', "_", base).strip("_") or "page"
    return base[:180] + ".txt"


def fetch_html(
    url: str, timeout: float, headers: dict[str, str], *, verify_ssl: bool
) -> str:
    r = requests.get(url, timeout=timeout, headers=headers, verify=verify_ssl)
    r.raise_for_status()
    r.encoding = r.apparent_encoding or "utf-8"
    return r.text


def html_to_text(html: str, selector: str | None) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    root = soup.select_one(selector) if selector else soup.body or soup
    if root is None:
        return ""
    lines: list[str] = []
    for chunk in root.stripped_strings:
        t = str(chunk).strip()
        if t:
            lines.append(t)
    return "\n".join(lines)


def scrape_one(
    url: str,
    *,
    selector: str | None,
    timeout: float,
    user_agent: str,
    verify_ssl: bool,
) -> str:
    headers = {"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"}
    html = fetch_html(url, timeout=timeout, headers=headers, verify_ssl=verify_ssl)
    return html_to_text(html, selector).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scrape public HTML pages to plain text.")
    parser.add_argument("urls", nargs="+", help="One or more page URLs")
    parser.add_argument(
        "--out",
        type=Path,
        help="Write a single URL to this file (only when exactly one URL is given)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("scraped"),
        help="Directory for output when using multiple URLs (default: ./scraped)",
    )
    parser.add_argument(
        "--selector",
        help="CSS selector for the root element to extract text from (default: whole body)",
    )
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds")
    parser.add_argument("--user-agent", default=DEFAULT_UA, help="User-Agent header")
    parser.add_argument(
        "--no-verify-ssl",
        action="store_true",
        help="Disable TLS certificate verification (insecure; use only if you hit SSL errors)",
    )
    args = parser.parse_args()
    verify_ssl = not args.no_verify_ssl

    if len(args.urls) > 1 and args.out:
        print("Error: --out can only be used with a single URL.", file=sys.stderr)
        sys.exit(2)

    if len(args.urls) == 1 and args.out:
        text = scrape_one(
            args.urls[0],
            selector=args.selector,
            timeout=args.timeout,
            user_agent=args.user_agent,
            verify_ssl=verify_ssl,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + ("\n" if text else ""), encoding="utf-8")
        print(f"Wrote {len(text)} characters to {args.out}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i, url in enumerate(args.urls):
        if i and args.delay > 0:
            time.sleep(args.delay)
        try:
            text = scrape_one(
                url,
                selector=args.selector,
                timeout=args.timeout,
                user_agent=args.user_agent,
                verify_ssl=verify_ssl,
            )
        except requests.RequestException as e:
            print(f"Failed {url}: {e}", file=sys.stderr)
            continue
        path = args.out_dir / url_to_basename(url)
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        print(f"Saved {len(text)} chars -> {path}")


if __name__ == "__main__":
    main()
