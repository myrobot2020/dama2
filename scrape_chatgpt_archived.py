"""
Scrape ChatGPT **Archived chats** (or similar long virtualized list) with Playwright.

Designed for very long lists, long waits, and multi-hour/multi-day runs:
  - Checkpoints to disk after each batch (resume after crash / Ctrl+C)
  - Plateau detection: stop when item count stops growing
  - Retries, timeouts, structured logging
  - Optional Ollama: health check + optional embedding-based near-duplicate flagging

Performance (see also --fetch-all-resources / --checkpoint-pretty):
  - One evaluate per iteration for full row payload; post-scroll polls use count-only JS
  - Dialog scroll targets the largest scrollable child (no full-subtree querySelectorAll('*'))
  - Optional blocking of image/font/media routes; compact checkpoint JSON; sorted(set) removed
  - JSONL file kept open across iterations; Ollama duplicate ring uses bounded deque

Legal / ToS: Only use on data you are allowed to export. This automates your own UI.

Setup
-----
  pip install -r requirements-browser-scrape.txt
  playwright install chromium

First-time login (saves cookies/local storage; use **Continue with Google** in the browser):
  python scrape_chatgpt_archived.py --login --storage-state auth_chatgpt.json
  python scrape_chatgpt_archived.py --login --login-url https://chatgpt.com/

Scrape (you must already be able to open Archived chats in that profile):
  python scrape_chatgpt_archived.py --storage-state auth_chatgpt.json \\
      --url "https://chatgpt.com/c/YOUR_CONV_ID#settings/DataControls/ArchivedChats" \\
      --out-dir scraped_archived

Chrome CDP (use your logged-in Chrome; quit other Chrome first):
  chrome.exe --remote-debugging-port=9222
  python scrape_chatgpt_archived.py --connect-cdp http://127.0.0.1:9222 \\
      --skip-navigation --out-dir scraped_archived

Scroll archived list to the end only (no JSONL; same plateau detection as scrape):
  python scrape_chatgpt_archived.py --connect-cdp http://127.0.0.1:9222 \\
      --skip-navigation --scroll-to-end-only
  # From current scroll position: add --no-start-top
  # Start Chrome after the script: keep retrying CDP for up to 30 minutes:
  python scrape_chatgpt_archived.py --connect-cdp http://127.0.0.1:9222 \\
      --wait-for-cdp-s 1800 --skip-navigation --scroll-to-end-only

Manual step: open **Manage Archived chats** if the URL alone does not open the modal;
the script waits for dialog links before starting.

Environment:
  OLLAMA_HOST   default http://127.0.0.1:11434
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# One round-trip: extract rows + link count (for logging without second evaluate)
_EXTRACT_AND_COUNT_JS = """() => {
  const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
  const roots = dialogs.length ? dialogs : [document.body];
  const out = [];
  const seen = new Set();
  for (const root of roots) {
    root.querySelectorAll('a[href*="/c/"]').forEach(a => {
      const href = a.getAttribute('href') || '';
      if (!href.includes('/c/')) return;
      const u = href.split('?')[0].split('#')[0];
      if (seen.has(u)) return;
      seen.add(u);
      const t = (a.innerText || '').trim().replace(/\\s+/g, ' ');
      if (t.length < 1) return;
      out.push({ href: u, title: t });
    });
  }
  return { rows: out, count: out.length };
}"""

_COUNT_ONLY_JS = """() => {
  const dialogs = Array.from(document.querySelectorAll('[role="dialog"]'));
  const roots = dialogs.length ? dialogs : [document.body];
  const seen = new Set();
  let n = 0;
  for (const root of roots) {
    root.querySelectorAll('a[href*="/c/"]').forEach(a => {
      const href = a.getAttribute('href') || '';
      if (!href.includes('/c/')) return;
      const u = href.split('?')[0].split('#')[0];
      if (seen.has(u)) return;
      seen.add(u);
      const t = (a.innerText || '').trim();
      if (t.length < 1) return;
      n++;
    });
  }
  return n;
}"""

# mode: "top" | "end" — one parse, child-walk only (no dialog.querySelectorAll('*')).
_DIALOG_SCROLL_JS = """(mode) => {
  function bestScrollable(root) {
    let best = null, bestH = 0;
    const stack = [root];
    while (stack.length) {
      const el = stack.pop();
      if (!el || el.nodeType !== 1) continue;
      const st = getComputedStyle(el);
      const oy = st.overflowY;
      if ((oy === 'auto' || oy === 'scroll') && el.scrollHeight > el.clientHeight + 40) {
        if (el.scrollHeight > bestH) { bestH = el.scrollHeight; best = el; }
      }
      for (let i = 0; i < el.children.length; i++) stack.push(el.children[i]);
    }
    return best;
  }
  const d = document.querySelector('[role="dialog"]');
  if (!d) return;
  const el = bestScrollable(d);
  if (!el) return;
  el.scrollTop = mode === 'top' ? 0 : el.scrollHeight;
}"""

_SCROLL_LAST_INTO_VIEW_JS = """() => {
  const d = document.querySelector('[role="dialog"]') || document.body;
  const links = d.querySelectorAll('a[href*="/c/"]');
  if (!links.length) return false;
  links[links.length - 1].scrollIntoView({ block: 'end', inline: 'nearest' });
  return true;
}"""

try:
    import requests
except ImportError:
    requests = None  # type: ignore

try:
    from playwright.sync_api import TimeoutError as PlaywrightTimeout
    from playwright.sync_api import sync_playwright
except ImportError as e:
    raise SystemExit(
        "Install Playwright: pip install -r requirements-browser-scrape.txt && playwright install chromium"
    ) from e


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Checkpoint:
    seen_hrefs: list[str]
    total_rows_written: int
    iterations: int
    last_plateau_hits: int
    last_dom_link_count: int
    updated_at: str

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            seen_hrefs=list(data.get("seen_hrefs", [])),
            total_rows_written=int(data.get("total_rows_written", 0)),
            iterations=int(data.get("iterations", 0)),
            last_plateau_hits=int(data.get("last_plateau_hits", 0)),
            last_dom_link_count=int(data.get("last_dom_link_count", 0)),
            updated_at=str(data.get("updated_at", "")),
        )

    def save(self, path: Path, *, compact: bool = True) -> None:
        data = asdict(self)
        if compact:
            path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        else:
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def setup_logging(log_file: Path | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=handlers,
    )


def ollama_health(host: str) -> bool:
    if requests is None:
        return False
    try:
        r = requests.get(f"{host.rstrip('/')}/api/tags", timeout=5)
        return r.ok
    except OSError:
        return False


def ollama_embed(host: str, model: str, text: str) -> list[float] | None:
    """Single-text embedding; used sparingly (optional duplicate hints)."""
    if requests is None:
        return None
    try:
        r = requests.post(
            f"{host.rstrip('/')}/api/embeddings",
            json={"model": model, "prompt": text[:8000]},
            timeout=120,
        )
        if not r.ok:
            return None
        data = r.json()
        emb = data.get("embedding")
        return list(emb) if isinstance(emb, list) else None
    except OSError:
        return None


def cosine_sim(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _parse_row_items(raw_rows: Any) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not isinstance(raw_rows, list):
        return rows
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        href = str(item.get("href", "")).strip()
        title = str(item.get("title", "")).strip()
        if href and title:
            rows.append({"href": href, "title": title})
    return rows


def extract_visible_batch(page) -> tuple[list[dict[str, str]], int]:
    """Single round-trip: visible rows + count (same work as count-only path)."""
    try:
        raw = page.evaluate(_EXTRACT_AND_COUNT_JS)
    except Exception as e:
        logging.warning("extract_visible_batch evaluate failed: %s", e)
        return [], 0
    if not isinstance(raw, dict):
        return [], 0
    rows = _parse_row_items(raw.get("rows"))
    try:
        c = int(raw.get("count", len(rows)))
    except (TypeError, ValueError):
        c = len(rows)
    return rows, c


def count_dialog_chat_links(page) -> int:
    """Lightweight poll during wait (count only; avoids serializing full row payloads)."""
    try:
        raw = page.evaluate(_COUNT_ONLY_JS)
        return int(raw) if isinstance(raw, (int, float)) else 0
    except Exception as e:
        logging.debug("count_dialog_chat_links failed: %s", e)
        return 0


def scroll_dialog_to_top(page) -> None:
    page.evaluate(_DIALOG_SCROLL_JS, "top")


def scroll_dialog_to_end(page) -> None:
    page.evaluate(_DIALOG_SCROLL_JS, "end")


def scroll_last_link_into_view(page) -> bool:
    return bool(page.evaluate(_SCROLL_LAST_INTO_VIEW_JS))


def connect_chromium_cdp(
    playwright,
    cdp_url: str,
    *,
    wait_s: float,
    interval_s: float,
):
    """
    Connect to Chrome/Edge over CDP. If ``wait_s`` is 0, fail on the first error.
    If ``wait_s`` > 0, retry every ``interval_s`` until success or ``wait_s`` wall seconds elapse.
    """
    t0 = time.time()
    last_err: Exception | None = None
    while True:
        try:
            return playwright.chromium.connect_over_cdp(cdp_url)
        except Exception as e:
            last_err = e
            if wait_s <= 0:
                logging.error("connect_over_cdp(%r) failed: %s", cdp_url, e)
                raise
            elapsed = time.time() - t0
            if elapsed >= wait_s:
                logging.error(
                    "connect_over_cdp(%r) still failing after %.1fs: %s",
                    cdp_url,
                    wait_s,
                    last_err,
                )
                raise last_err
            logging.info(
                "CDP not ready (%s); retrying in %.1fs (~%.0fs left)...",
                e,
                interval_s,
                max(0.0, wait_s - elapsed),
            )
            time.sleep(interval_s)


def wait_until_stable_count(
    page,
    get_count,
    *,
    min_wait_s: float,
    max_wait_s: float,
    poll_s: float,
    stable_rounds: int,
) -> tuple[int, int]:
    """
    After an action, wait until `get_count()` is unchanged for `stable_rounds` polls
    (or max_wait_s). Returns (final_count, polls_used).
    """
    time.sleep(min_wait_s)
    t0 = time.time()
    last = -1
    stable = 0
    polls = 0
    while time.time() - t0 < max_wait_s:
        polls += 1
        try:
            cur = int(get_count())
        except Exception as e:
            logging.debug("get_count error: %s", e)
            cur = last
        if cur == last:
            stable += 1
        else:
            stable = 0
            last = cur
        if stable >= stable_rounds:
            return cur, polls
        time.sleep(poll_s + random.uniform(0, poll_s * 0.2))
    return last if last >= 0 else cur, polls


def scroll_archived_dialog_to_end(page, args: argparse.Namespace, shutdown: dict[str, bool]) -> None:
    """
    Scroll the archived-chats dialog until no new conversation links appear in the visible
    batch (same plateau idea as full scrape, but no JSONL/checkpoint writes).
    """
    seen: set[str] = set()
    plateau_hits = 0
    iteration = 0
    batch_limit = args.max_iterations if args.max_iterations else None

    def _count() -> int:
        return count_dialog_chat_links(page)

    while not shutdown["stop"]:
        iteration += 1
        batch_rows, dom_count = extract_visible_batch(page)
        new_in_batch = 0
        for row in batch_rows:
            href = row["href"]
            if href in seen:
                continue
            seen.add(href)
            new_in_batch += 1

        if new_in_batch == 0:
            plateau_hits += 1
        else:
            plateau_hits = 0

        logging.info(
            "scroll-only iter=%s dom_links=%s new_unique=%s total_unique=%s plateau=%s/%s",
            iteration,
            dom_count,
            new_in_batch,
            len(seen),
            plateau_hits,
            args.plateau_iterations,
        )

        if plateau_hits >= args.plateau_iterations:
            logging.info("scroll-only: end of list (plateau).")
            break

        if batch_limit is not None and iteration >= batch_limit:
            logging.info("scroll-only: stopped at --max-iterations %s.", args.max_iterations)
            break

        scrolled = scroll_last_link_into_view(page)
        if not scrolled:
            scroll_dialog_to_end(page)

        final_c, polls = wait_until_stable_count(
            page,
            _count,
            min_wait_s=args.min_wait_s,
            max_wait_s=args.max_wait_s,
            poll_s=args.poll_s,
            stable_rounds=args.stable_rounds,
        )
        logging.debug("scroll-only post-scroll stable dom link count=%s polls=%s", final_c, polls)
        time.sleep(args.between_iteration_sleep_s)

    logging.info("scroll-only finished; cumulative unique hrefs (visible batches)=%s", len(seen))


def run_login(storage_state: Path, login_url: str) -> None:
    storage_state.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.goto(login_url, wait_until="domcontentloaded")
        logging.info(
            "In the browser: open **Log in** if you see the home page, then choose "
            "**Continue with Google** (or your provider). Finish Google sign-in and any 2FA "
            "until ChatGPT is fully loaded. Then return here and press Enter."
        )
        input()
        context.storage_state(path=str(storage_state))
        browser.close()
    logging.info("Saved storage state to %s", storage_state)


def run_scrape(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / args.out_jsonl
    ckpt_path = out_dir / args.checkpoint
    log_path = out_dir / args.log_file if args.log_file else None
    setup_logging(log_path, args.verbose)

    ollama_host = (args.ollama_host or "http://127.0.0.1:11434").rstrip("/")
    if args.ollama_check:
        if ollama_health(ollama_host):
            logging.info("Ollama reachable at %s", ollama_host)
        else:
            logging.warning("Ollama not reachable at %s (continuing without it)", ollama_host)

    seen: set[str] = set()
    if ckpt_path.is_file() and not args.fresh:
        ckpt = Checkpoint.load(ckpt_path)
        seen = set(ckpt.seen_hrefs)
        logging.info(
            "Resume: %d known hrefs, %d rows written, iter %d",
            len(seen),
            ckpt.total_rows_written,
            ckpt.iterations,
        )
    else:
        ckpt = Checkpoint(
            seen_hrefs=[],
            total_rows_written=0,
            iterations=0,
            last_plateau_hits=0,
            last_dom_link_count=0,
            updated_at=_utc_iso(),
        )

    shutdown = {"stop": False}

    def _handle_sig(*_a: Any) -> None:
        logging.warning("Signal received; will stop after current iteration.")
        shutdown["stop"] = True

    signal.signal(signal.SIGINT, _handle_sig)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_sig)

    embed_ring: deque[tuple[str, list[float]]] = deque(maxlen=args.ollama_embed_ring)
    dup_hints = 0

    with sync_playwright() as p:
        uses_cdp = bool(args.connect_cdp)
        browser = None
        try:
            if uses_cdp:
                try:
                    browser = connect_chromium_cdp(
                        p,
                        args.connect_cdp,
                        wait_s=args.wait_for_cdp_s,
                        interval_s=args.wait_for_cdp_interval_s,
                    )
                except Exception:
                    return 4
                if not browser.contexts:
                    logging.error(
                        "CDP: no browser contexts. Start Chrome with "
                        "--remote-debugging-port=9222 (quit other Chrome windows first)."
                    )
                    browser.close()
                    return 4
                context = browser.contexts[0]
                page = None
                for pg in context.pages:
                    try:
                        u = pg.url or ""
                    except Exception:
                        u = ""
                    if "chatgpt.com" in u:
                        page = pg
                        break
                if page is None and context.pages:
                    page = context.pages[0]
                if page is None:
                    page = context.new_page()
                logging.info("CDP attached; page url=%s", page.url)
            else:
                browser = p.chromium.launch(headless=args.headless)
                context = browser.new_context(storage_state=str(args.storage_state))
                page = context.new_page()

            if args.block_heavy_resources:

                def _route_block(route) -> None:
                    if route.request.resource_type in ("image", "media", "font"):
                        route.abort()
                    else:
                        route.continue_()

                context.route("**/*", _route_block)

            page.set_default_timeout(args.navigation_timeout_ms)

            try:
                if args.skip_navigation:
                    logging.info("--skip-navigation: not calling goto; url=%s", page.url)
                else:
                    page.goto(args.url, wait_until="domcontentloaded")
            except PlaywrightTimeout:
                logging.error("Navigation timeout to %s", args.url)
                return 2

            logging.info("Waiting for archived modal links (timeout %ss)...", args.modal_wait_s)
            try:
                page.wait_for_function(
                    """() => {
              const d = document.querySelector('[role="dialog"]');
              const root = d || document.body;
              return root.querySelectorAll('a[href*="/c/"]').length >= 1;
            }""",
                    timeout=int(args.modal_wait_s * 1000),
                )
            except PlaywrightTimeout:
                logging.error(
                    "No conversation links found. Open **Manage Archived chats** in the UI, then re-run."
                )
                return 3

            if args.start_top:
                scroll_dialog_to_top(page)
                time.sleep(args.after_scroll_wait_s)

            if args.scroll_to_end_only:
                scroll_archived_dialog_to_end(page, args, shutdown)
                return 0

            plateau_hits = ckpt.last_plateau_hits
            iteration = ckpt.iterations
            rows_written = ckpt.total_rows_written
            ck_compact = not args.checkpoint_pretty
            batch_limit = (ckpt.iterations + args.max_iterations) if args.max_iterations else None

            def save_ckpt(dom_links: int) -> None:
                Checkpoint(
                    seen_hrefs=list(seen),
                    total_rows_written=rows_written,
                    iterations=iteration,
                    last_plateau_hits=plateau_hits,
                    last_dom_link_count=dom_links,
                    updated_at=_utc_iso(),
                ).save(ckpt_path, compact=ck_compact)

            jsonl_f = jsonl_path.open("a", encoding="utf-8")
            try:
                while not shutdown["stop"]:
                    iteration += 1
                    batch_rows, dom_count = extract_visible_batch(page)
                    new_in_batch = 0

                    for row in batch_rows:
                        href = row["href"]
                        if href in seen:
                            continue
                        seen.add(href)
                        rec = {
                            "href": href,
                            "title": row["title"],
                            "scraped_at": _utc_iso(),
                            "iteration": iteration,
                        }
                        jsonl_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        jsonl_f.flush()
                        new_in_batch += 1
                        rows_written += 1

                        if args.ollama_embed_model and args.ollama_dup_threshold > 0:
                            tkey = row["title"][:500]
                            emb = ollama_embed(ollama_host, args.ollama_embed_model, tkey)
                            if emb:
                                for prev_title, prev_emb in embed_ring:
                                    sim = cosine_sim(emb, prev_emb)
                                    if sim >= args.ollama_dup_threshold:
                                        dup_hints += 1
                                        logging.debug(
                                            "Ollama dup hint (sim=%.3f): %r ~ %r",
                                            sim,
                                            tkey[:80],
                                            prev_title[:80],
                                        )
                                        break
                                embed_ring.append((tkey, emb))

                    if new_in_batch == 0:
                        plateau_hits += 1
                    else:
                        plateau_hits = 0

                    logging.info(
                        "iter=%s dom_links=%s new_unique=%s total_unique=%s plateau=%s/%s",
                        iteration,
                        dom_count,
                        new_in_batch,
                        len(seen),
                        plateau_hits,
                        args.plateau_iterations,
                    )

                    if plateau_hits >= args.plateau_iterations:
                        logging.info(
                            "Stopping: no new unique hrefs for %s iterations (end of list or stuck).",
                            args.plateau_iterations,
                        )
                        save_ckpt(dom_count)
                        break

                    if batch_limit is not None and iteration >= batch_limit:
                        logging.info(
                            "Stopping: --max-iterations %s (iteration=%s).",
                            args.max_iterations,
                            iteration,
                        )
                        save_ckpt(dom_count)
                        break

                    scrolled = scroll_last_link_into_view(page)
                    if not scrolled:
                        scroll_dialog_to_end(page)

                    def _count() -> int:
                        return count_dialog_chat_links(page)

                    final_c, polls = wait_until_stable_count(
                        page,
                        _count,
                        min_wait_s=args.min_wait_s,
                        max_wait_s=args.max_wait_s,
                        poll_s=args.poll_s,
                        stable_rounds=args.stable_rounds,
                    )
                    logging.debug("post-scroll stable dom link count=%s polls=%s", final_c, polls)

                    save_ckpt(final_c)

                    time.sleep(args.between_iteration_sleep_s)

                    if shutdown["stop"]:
                        logging.info("Shutdown requested; checkpoint already saved for this iteration.")
                        break
            finally:
                jsonl_f.close()

        finally:
            if browser is not None:
                browser.close()

    if dup_hints:
        logging.info("Ollama near-duplicate hints: %s", dup_hints)
    logging.info("Done. unique=%s rows_written=%s out=%s", len(seen), rows_written, jsonl_path)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--login", action="store_true", help="Interactive login; save --storage-state")
    p.add_argument(
        "--login-url",
        default="https://chatgpt.com/auth/login",
        help="Page to open for --login (default: ChatGPT sign-in; use https://chatgpt.com/ for home)",
    )
    p.add_argument(
        "--storage-state",
        type=Path,
        default=Path("auth_chatgpt.json"),
        help="Playwright storage state path (not used with --connect-cdp)",
    )
    p.add_argument(
        "--connect-cdp",
        default="",
        metavar="URL",
        help="Attach to Chrome/Edge, e.g. http://127.0.0.1:9222 (start browser with --remote-debugging-port=9222)",
    )
    p.add_argument(
        "--wait-for-cdp-s",
        type=float,
        default=0.0,
        metavar="SEC",
        help="With --connect-cdp: retry until Chrome answers or this many seconds elapse (0=fail on first error)",
    )
    p.add_argument(
        "--wait-for-cdp-interval-s",
        type=float,
        default=3.0,
        help="Seconds between CDP retries when --wait-for-cdp-s > 0",
    )
    p.add_argument(
        "--skip-navigation",
        action="store_true",
        help="Do not page.goto (Archived chats must already be open on the attached tab)",
    )
    p.add_argument(
        "--scroll-to-end-only",
        action="store_true",
        help=(
            "Only scroll the archived dialog until plateau (same logic as scrape, no JSONL/checkpoint). "
            "Use with --connect-cdp or storage-state browser; Archived chats must be open."
        ),
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        metavar="N",
        help="Stop after N scrape iterations (0=unlimited). Stops before next scroll.",
    )
    p.add_argument(
        "--url",
        default="https://chatgpt.com/#settings/DataControls/ArchivedChats",
        help="Page URL (open Archived chats first if needed)",
    )
    p.add_argument("--out-dir", type=Path, default=Path("scraped_archived"))
    p.add_argument("--out-jsonl", default="archived_chats.jsonl")
    p.add_argument("--checkpoint", default="checkpoint.json")
    p.add_argument("--log-file", default="scrape_archived.log")
    p.add_argument("--fresh", action="store_true", help="Ignore existing checkpoint")
    p.add_argument("--headless", action="store_true", help="Headless (may break login walls)")
    p.add_argument("--start-top", action="store_true", default=True, help="Scroll dialog to top first")
    p.add_argument("--no-start-top", action="store_false", dest="start_top")
    p.add_argument("--navigation-timeout-ms", type=int, default=120_000)
    p.add_argument("--modal-wait-s", type=float, default=300.0, help="Wait for first /c/ link in dialog")
    p.add_argument("--min-wait-s", type=float, default=4.0, help="Minimum wait after each scroll")
    p.add_argument("--max-wait-s", type=float, default=120.0, help="Max wait for DOM to stabilize")
    p.add_argument("--poll-s", type=float, default=3.0, help="Poll interval while waiting for load")
    p.add_argument(
        "--stable-rounds",
        type=int,
        default=3,
        help="Consecutive equal counts required to treat load as stable",
    )
    p.add_argument("--between-iteration-sleep-s", type=float, default=1.0)
    p.add_argument("--after-scroll-wait-s", type=float, default=2.0)
    p.add_argument(
        "--plateau-iterations",
        type=int,
        default=5,
        help="Stop after this many iterations with zero new unique hrefs",
    )
    p.add_argument("--ollama-host", default=None, help="Default OLLAMA_HOST or 127.0.0.1:11434")
    p.add_argument("--ollama-check", action="store_true", help="Ping Ollama /api/tags at startup")
    p.add_argument(
        "--ollama-embed-model",
        default="",
        help="If set (e.g. nomic-embed-text), optional near-duplicate hints (slow)",
    )
    p.add_argument(
        "--ollama-dup-threshold",
        type=float,
        default=0.0,
        help="Cosine similarity threshold for duplicate hint (0=disabled)",
    )
    p.add_argument(
        "--ollama-embed-ring",
        type=int,
        default=200,
        metavar="N",
        help="Max prior embeddings to compare per new title (ring buffer)",
    )
    p.add_argument(
        "--fetch-all-resources",
        action="store_false",
        dest="block_heavy_resources",
        help="Do not block images/fonts/media (slower, more bandwidth)",
    )
    p.set_defaults(block_heavy_resources=True)
    p.add_argument(
        "--checkpoint-pretty",
        action="store_true",
        help="Write indented checkpoint JSON (larger/slower; default is compact)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.login:
        setup_logging(None, True)
        run_login(args.storage_state, args.login_url)
        return 0
    if not args.connect_cdp and not args.storage_state.is_file():
        print(
            "Missing storage state. Use --login, or --connect-cdp with Chrome on a debug port.",
            file=sys.stderr,
        )
        return 1
    return run_scrape(args)


if __name__ == "__main__":
    raise SystemExit(main())
