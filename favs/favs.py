#!/usr/bin/env python3
"""Your own TikTok Favorites -> NAS "TikTok Saved" series. Deterministic, zero-LLM; runs after each DM poll,
under the poller's lock, attached over CDP to the poller's logged-in browser.

How (measured 2026-09-28): the logged-in profile page `/@<you>` has a "Favorites" tab (role=tab, no
data-e2e). Opening it makes the web app call `/api/user/collect/item_list/` (16 items per page, newest
favorite first, `cursor` = the favorite time in unix seconds of the page's last item, `hasMore`). Replaying
that request ourselves works once and then returns an empty body (it is signed), so we do NOT: we scroll the
grid like a person and read the app's own responses.

State: /data/favs_index.json = {"complete": bool, "items": {id: {author, desc, created, fav_ts, first_seen,
submitted?}}}. `fav_ts` is a lower bound (the cursor of the page the item arrived in). Once the index is
complete, a run stops at the first page with no new ids, so a normal run costs one page load.
New ids go to MeTube (folder "TikTok Saved"), newest favorite first, MAX_SUBMIT per run.

Your handle: $FAVS_HANDLE or the first line of /data/favs_handle.

Exit codes: 0 ok · 3 not logged in / wrong account / no handle configured · 5 MeTube unreachable · 6 browser (CDP) unreachable ·
7 Favorites tab or its API not found (markup changed). Off-switch: /data/../FAVS_OFF (host ~/tiktok-rag/FAVS_OFF).
"""
import json, os, re, sys, threading, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import sync_playwright

CDP_URL = os.environ.get("CDP_URL", "http://127.0.0.1:9222")
DATA = Path(os.environ.get("DATA_DIR", "/data"))
INDEX = DATA / "favs_index.json"
OFF = DATA.parent / "FAVS_OFF"
HANDLE = os.environ.get("FAVS_HANDLE", "")
if not HANDLE and (DATA / "favs_handle").exists():
    HANDLE = next(iter((DATA / "favs_handle").read_text().split()), "")
METUBE = os.environ.get("METUBE_BASE", "http://127.0.0.1:8081").rstrip("/")
FOLDER = os.environ.get("FAVS_FOLDER", "TikTok Saved")
MAX_SUBMIT = int(os.environ.get("FAVS_MAX_SUBMIT", "25"))
MAX_PAGES = int(os.environ.get("FAVS_MAX_PAGES", "80"))  # ~1280 favorites; grid tiles are light
DEADLINE = int(os.environ.get("FAVS_DEADLINE", "420"))
READ_EVERY = int(os.environ.get("FAVS_READ_EVERY", "3600"))  # browser read at most hourly once complete
DRY = os.environ.get("FAVS_DRY", "0") == "1"  # read + index only, submit nothing
API = "/api/user/collect/item_list/"
VIDEO_RE = re.compile(r"/video/(\d+)")
NOW = int(time.time())


def log(msg):
    print(f"{datetime.now(timezone.utc):%FT%TZ} favs: {msg}", flush=True)


def load_index():
    try:
        return json.loads(INDEX.read_text())
    except (OSError, ValueError):
        return {"complete": False, "items": {}}


def save_index(ix):
    tmp = INDEX.with_suffix(".tmp")
    tmp.write_text(json.dumps(ix, ensure_ascii=False, indent=0))
    os.replace(tmp, INDEX)


def metube_seen():
    try:
        with urllib.request.urlopen(METUBE + "/history", timeout=20) as r:
            hist = json.load(r)
    except Exception as e:  # noqa: BLE001
        log(f"MeTube unreachable: {e!r}")
        return None
    seen = set()
    for bucket in ("done", "queue", "pending"):
        for rec in hist.get(bucket, []) or []:
            m = VIDEO_RE.search(rec.get("url", "") or "")
            if m:
                seen.add(m.group(1))
    return seen


def metube_add(vid):
    body = json.dumps({"url": f"https://www.tiktok.com/@_/video/{vid}", "quality": "best", "format": "any",
                       "folder": FOLDER, "auto_start": True}).encode()
    req = urllib.request.Request(METUBE + "/add", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def arm_watchdog(page, seconds):
    """A stalled renderer never answers, so close our own tab after `seconds`."""
    try:
        tid = page.context.new_cdp_session(page).send("Target.getTargetInfo")["targetInfo"]["targetId"]
    except Exception as e:  # noqa: BLE001
        log(f"watchdog not armed: {e!r}"[:160])
        return

    def fire():
        time.sleep(seconds)
        log(f"WATCHDOG: still running after {seconds}s, closing our tab")
        try:
            urllib.request.urlopen(urllib.request.Request(f"{CDP_URL}/json/close/{tid}", method="PUT"), timeout=10).read()
        except Exception as e:  # noqa: BLE001
            log(f"watchdog close failed: {e!r}"[:160])

    threading.Thread(target=fire, daemon=True).start()


def read_favorites(page, ix):
    """Scroll the Favorites grid; return (pages_read, new_ids, reached_end). Mutates ix['items']."""
    pages = []

    def on_resp(r):
        if API in r.url:
            try:
                pages.append(r.json())
            except Exception as e:  # noqa: BLE001
                pages.append({"_err": repr(e)[:120]})

    page.on("response", on_resp)
    page.get_by_role("tab", name="Favorites").first.click(timeout=15000)
    items, new, done, prev_cursor, handled, idle = ix["items"], [], False, NOW, 0, 0
    while handled < MAX_PAGES:
        t0 = time.time()
        while len(pages) <= handled and time.time() - t0 < 10:
            page.wait_for_timeout(500)
        if len(pages) <= handled:
            idle += 1
            if idle >= 3:
                break
            page.mouse.wheel(0, 4000)
            page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
            continue
        idle = 0
        j = pages[handled]
        handled += 1
        if "_err" in j or j.get("statusCode", 0) != 0:
            log(f"page {handled}: bad response {str(j)[:160]}")
            break
        cursor = int(j.get("cursor") or 0)
        fresh = 0
        for it in j.get("itemList") or []:
            vid = str(it.get("id"))
            if vid in items:
                continue
            fresh += 1
            new.append(vid)
            items[vid] = {"author": (it.get("author") or {}).get("uniqueId"), "desc": (it.get("desc") or "")[:300],
                          "created": it.get("createTime"), "fav_ts": cursor or None, "fav_before": prev_cursor,
                          "first_seen": NOW}
        prev_cursor = cursor or prev_cursor
        if not j.get("hasMore"):
            done = True
            break
        if ix.get("complete") and fresh == 0:
            break  # steady state: everything below is already indexed
        page.wait_for_timeout(1500)
        page.mouse.wheel(0, 4000)
        page.evaluate("() => window.scrollTo(0, document.body.scrollHeight)")
    return handled, new, done


def main():
    if OFF.exists():
        log("FAVS_OFF present, exiting")
        return 0
    if not HANDLE:
        log("no handle: set FAVS_HANDLE or write it to /data/favs_handle")
        return 3
    seen = metube_seen()
    if seen is None:
        return 5
    ix = load_index()
    n, new, done = 0, [], False
    if ix.get("complete") and NOW - ix.get("last_read", 0) < READ_EVERY:
        log("read not due; submitting backlog only")
    else:
        rc = read_step(ix)
        if rc is not None:
            return rc
        n, new, done = ix.pop("_run")
        ix["last_read"] = NOW
    return submit_step(ix, seen, n, new, done)


def read_step(ix):
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(CDP_URL, timeout=15000)
        except Exception as e:  # noqa: BLE001
            log(f"browser unreachable: {e!r}"[:200])
            return 6
        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
        page = ctx.new_page()  # our own tab; never touch the poller's or a human's
        arm_watchdog(page, DEADLINE)
        try:
            page.goto(f"https://www.tiktok.com/@{HANDLE}?lang=en", wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(4000)
            who = page.evaluate("""() => { try { return JSON.parse(document.getElementById('__UNIVERSAL_DATA_FOR_REHYDRATION__')
                .textContent).__DEFAULT_SCOPE__['webapp.app-context'].user.uniqueId } catch (e) { return null } }""")
            if who != HANDLE:
                log(f"logged in as {who!r}, expected {HANDLE!r}")
                return 3
            try:
                n, new, done = read_favorites(page, ix)
            except Exception as e:  # noqa: BLE001
                log(f"favorites not readable: {e!r}"[:200])
                return 7
            if n == 0:
                log("Favorites tab opened but no API page arrived")
                return 7
            ix["_run"] = (n, new, done)
        finally:
            try:
                page.close()
            except Exception:  # noqa: BLE001
                pass
    return None


def submit_step(ix, seen, n, new, done):
    if done:
        ix["complete"] = True
    # Submit: newest favorite first (dict order = read order = newest first), skipping known downloads.
    todo = [v for v, r in ix["items"].items() if not r.get("submitted") and v not in seen]
    todo.sort(key=lambda v: -(ix["items"][v].get("fav_before") or 0))  # newest favorite first (stable in-page)
    for v in [v for v, r in ix["items"].items() if not r.get("submitted") and v in seen]:
        ix["items"][v]["submitted"] = "already"
    sent = 0
    for vid in ([] if DRY else todo[:MAX_SUBMIT]):
        try:
            r = metube_add(vid)
            ix["items"][vid]["submitted"] = NOW if r.get("status") == "ok" else f"err {str(r)[:80]}"
            sent += 1
        except Exception as e:  # noqa: BLE001
            log(f"submit {vid} failed: {e!r}"[:160])
            break
    save_index(ix)
    log(f"pages {n} · new {len(new)} · total {len(ix['items'])} (complete={ix['complete']}) · submitted {sent} · "
        f"backlog {max(0, len(todo) - sent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
