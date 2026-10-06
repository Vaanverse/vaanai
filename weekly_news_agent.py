"""
weekly_news_agent.py

VaanAI's weekly environment-news agent. Every Sunday at 00:00 Japan time it:

1. Collects every environment story published in the previous week
   (Sunday 00:00 -> Saturday 23:59:59 JST) from a list of open RSS feeds.
2. Asks the model (Groq, same as the water agent) to act as an editor and pick
   the 5 most significant stories worldwide, plus a few reserves.
3. Opens each picked article, reads it, and asks the model to write a short
   headline, a 1-2 sentence tile description and a 3-5 sentence summary using
   only facts from the article.
4. Saves a resized copy of the article's preview image to Supabase Storage
   (bucket `news-images`) with a credit line, and writes the 5 stories to the
   `weekly_top_news` table (ranks 1-5). Re-running for the same week updates
   those 5 rows in place and bumps modified_at.

The website reads the `latest_top_news` view, which always returns just the
newest week's 5 stories.

Usage
-----
    python weekly_news_agent.py                    # normal weekly run
    python weekly_news_agent.py --dry-run          # pick + write, print, save nothing
    python weekly_news_agent.py --week-end 2026-10-04   # redo a past week (a Sunday)

Needs in .env: GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY (service_role).
Run sql/2026-10-06_weekly_top_news.sql in Supabase once before the first run.
"""

import argparse
import calendar
import html
import io
import json
import logging
import os
import re
import sys
import time
import uuid
from datetime import date, datetime, timedelta, timezone

import feedparser
import requests
import trafilatura
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

import storage_utils as storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

# Japan has no daylight saving, so a fixed +9h offset is exact (and avoids
# needing the tzdata package on Windows).
JST = timezone(timedelta(hours=9), "JST")

TABLE = "weekly_top_news"
BUCKET = "news-images"
TOP_N = 5
RESERVES = 4                 # extra picks used if an article can't be read or has no image
MAX_CANDIDATES = 50          # keeps the selection prompt inside Groq's free-tier token limit
MAX_PER_SOURCE = 8           # so one busy feed can't crowd out the others
MIN_CANDIDATES = 8           # fewer than this means the feeds are broken — fail the run
ARTICLE_CHARS = 3500         # how much article text the writer sees
IMAGE_MAX_WIDTH = 1200
IMAGE_MIN_WIDTH = 300        # smaller than this is a thumbnail, not a usable tile image

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0 Safari/537.36 VaanAI-NewsAgent/1.0 (+https://github.com/Vaanverse/vaanai)"
)

# Open RSS feeds of reputable environment desks around the world. A feed that
# is down or blocks us is logged and skipped. Add or remove freely.
FEEDS = [
    ("The Guardian", "https://www.theguardian.com/environment/rss"),
    ("BBC News", "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml"),
    ("Mongabay", "https://news.mongabay.com/feed/"),
    ("Mongabay India", "https://india.mongabay.com/feed/"),
    ("Carbon Brief", "https://www.carbonbrief.org/feed/"),
    ("UN News", "https://news.un.org/feed/subscribe/en/news/topic/climate-change/feed/rss.xml"),
    ("Inside Climate News", "https://insideclimatenews.org/feed/"),
    ("Yale Environment 360", "https://e360.yale.edu/feed.xml"),
    ("NASA Earth Observatory", "https://earthobservatory.nasa.gov/feeds/earth-observatory.rss"),
    ("Grist", "https://grist.org/feed/"),
    ("The Hindu", "https://www.thehindu.com/sci-tech/energy-and-environment/feeder/default.rss"),
]

CATEGORIES = [
    "climate", "weather & disasters", "water", "forests", "wildlife", "oceans",
    "pollution", "energy", "ice & polar", "agriculture", "policy",
]

# Headlines that are almost never "news of the week".
SKIP_TITLE = re.compile(
    r"\b(podcast|quiz|newsletter|live updates?|as it happened|in pictures|photos of the week|"
    r"crossword|letters?:|opinion|webinar|sponsored)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Week window
# ---------------------------------------------------------------------------


def week_window(now: datetime | None = None, week_end: date | None = None):
    """Returns (start, end) as aware datetimes for the previous full week in
    Japan time: start = Sunday 00:00 JST a week ago, end = the most recent
    Sunday 00:00 JST (exclusive). A run at Sunday 00:00 JST therefore covers
    the 7 days that just finished. A manual re-run later in the week covers
    the SAME week, so it updates those rows instead of adding new ones."""
    if week_end is not None:
        end = datetime(week_end.year, week_end.month, week_end.day, tzinfo=JST)
    else:
        now_jst = (now or datetime.now(timezone.utc)).astimezone(JST)
        days_since_sunday = (now_jst.weekday() + 1) % 7      # Mon=0..Sun=6 -> Sun=0
        end = (now_jst - timedelta(days=days_since_sunday)).replace(hour=0, minute=0, second=0, microsecond=0)
    return end - timedelta(days=7), end


# ---------------------------------------------------------------------------
# Collecting candidate stories
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(html.unescape(text).split())


def _entry_time(entry) -> datetime | None:
    t = entry.get("published_parsed") or entry.get("updated_parsed")
    return datetime.fromtimestamp(calendar.timegm(t), timezone.utc) if t else None


def _entry_image(entry) -> str | None:
    """Largest image the feed itself offers for this entry (media:content,
    media:thumbnail or an image enclosure)."""
    found = []
    for m in entry.get("media_content", []) + entry.get("media_thumbnail", []):
        if m.get("url") and (m.get("medium") in (None, "image") or "image" in (m.get("type") or "")):
            found.append((int(m.get("width") or 0), m["url"]))
    for enc in entry.get("enclosures", []):
        if "image" in (enc.get("type") or "") and enc.get("href"):
            found.append((0, enc["href"]))
    return max(found)[1] if found else None


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower())[:80]


def collect_candidates(start: datetime, end: datetime) -> list[dict]:
    per_source: dict[str, list[dict]] = {}
    seen_titles: set[str] = set()
    seen_links: set[str] = set()

    for source, url in FEEDS:
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=25)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
        except Exception as exc:  # noqa: BLE001 - one broken feed must not stop the run
            log.warning("Feed skipped — %s (%s): %s", source, url, storage.short_error(exc, 150))
            continue

        kept = []
        for e in feed.entries:
            published = _entry_time(e)
            title = _clean(e.get("title", ""))
            link = e.get("link", "")
            if not (published and title and link) or not (start <= published < end):
                continue
            if SKIP_TITLE.search(title):
                continue
            key = _norm_title(title)
            if key in seen_titles or link in seen_links:
                continue
            seen_titles.add(key)
            seen_links.add(link)
            kept.append({
                "source": source,
                "title": title,
                "url": link,
                "published_at": published,
                "snippet": _clean(e.get("summary", "")),
                "rss_image": _entry_image(e),
            })
        kept.sort(key=lambda c: c["published_at"], reverse=True)
        per_source[source] = kept[:MAX_PER_SOURCE]
        log.info("Feed %-24s %3d stories in the week (kept %d)", source, len(kept), len(per_source[source]))

    # Interleave sources (newest of each first) so the list stays diverse.
    candidates, round_ = [], 0
    while len(candidates) < MAX_CANDIDATES and any(round_ < len(v) for v in per_source.values()):
        for items in per_source.values():
            if round_ < len(items) and len(candidates) < MAX_CANDIDATES:
                candidates.append(items[round_])
        round_ += 1

    for i, c in enumerate(candidates, 1):
        c["id"] = f"c{i}"
    return candidates


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------


def call_groq(messages: list, tools: list, tool_name: str, api_key: str, max_retries: int = 4) -> dict:
    """Calls Groq with the tool FORCED, and returns that tool's parsed arguments.
    Waits and retries on free-tier rate limits (429) and on GPT-OSS's known
    intermittent 'tool_use_failed' / 'output_parse_failed' glitches."""
    for attempt in range(max_retries + 1):
        response = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": {"type": "function", "function": {"name": tool_name}},
                "reasoning_format": "hidden",
                "reasoning_effort": "low",
            },
            timeout=90,
        )

        if response.status_code == 429 and attempt < max_retries:
            try:
                wait = min(90, float(response.headers.get("retry-after", 0))) or 15 * (attempt + 1)
            except ValueError:
                wait = 15 * (attempt + 1)
            log.warning("Groq rate limit (attempt %d/%d) — waiting %.0fs.", attempt + 1, max_retries, wait)
            time.sleep(wait)
            continue

        if response.status_code == 413:
            raise RuntimeError("Prompt too large for this Groq model's limit — lower MAX_CANDIDATES or ARTICLE_CHARS.")

        if response.status_code == 400 and ("tool_use_failed" in response.text or "output_parse_failed" in response.text):
            recovered = _recover_failed_generation(response, tool_name)
            if recovered is not None:
                return recovered
            if attempt < max_retries:
                log.warning("Model produced unparseable tool output (attempt %d/%d) — retrying.", attempt + 1, max_retries)
                time.sleep(3)
                continue

        if not response.ok:
            log.error("Groq API error %s: %s", response.status_code, response.text[:500])
        response.raise_for_status()

        message = response.json()["choices"][0]["message"]
        for call in message.get("tool_calls") or []:
            if call["function"]["name"].split("<|")[0].strip() == tool_name:
                return json.loads(call["function"]["arguments"])
        if attempt < max_retries:
            log.warning("Model did not call %s (attempt %d/%d) — retrying.", tool_name, attempt + 1, max_retries)
            continue
        raise RuntimeError(f"Model never called {tool_name}.")

    raise RuntimeError("Exceeded retries talking to Groq.")


def _recover_failed_generation(response, tool_name: str):
    """GPT-OSS sometimes leaks a formatting token into the tool name while the
    arguments are fine — rescue those arguments instead of retrying."""
    try:
        failed = response.json().get("error", {}).get("failed_generation")
        parsed = json.loads(failed)
        if parsed.get("name", "").split("<|")[0].strip() == tool_name:
            args = parsed.get("arguments", {})
            return json.loads(args) if isinstance(args, str) else args
    except Exception:  # noqa: BLE001
        pass
    return None


# ---------------------------------------------------------------------------
# Step 1: the editor picks the stories
# ---------------------------------------------------------------------------

SELECT_TOOL = [{
    "type": "function",
    "function": {
        "name": "select_top_stories",
        "description": "Submit the ranked list of chosen story ids, most important first.",
        "parameters": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "description": f"{TOP_N + RESERVES} distinct candidate ids, best first. "
                                   f"The first {TOP_N} are the top stories; the rest are reserves.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Candidate id, e.g. 'c12'."},
                            "reason": {"type": "string", "description": "Why it matters, max 20 words."},
                        },
                        "required": ["id", "reason"],
                    },
                },
            },
            "required": ["picks"],
        },
    },
}]


def select_stories(candidates: list[dict], start: datetime, end: datetime, api_key: str) -> list[dict]:
    lines = [
        f"{c['id']} | {c['source']} | {c['published_at'].astimezone(JST):%d %b} | {c['title']} — {c['snippet'][:200]}"
        for c in candidates
    ]
    messages = [
        {"role": "system", "content": (
            "You are the news editor of VaanAI, a website about changes on planet Earth. "
            "From the candidate list, choose the most significant ENVIRONMENT news stories of the week "
            "for a worldwide audience. Judge by: real-world impact on people, ecosystems or the climate; "
            "new findings, events or decisions (not opinion columns, explainers, reviews, podcasts or "
            "evergreen features); and scientific credibility. Keep the selection diverse: never two stories "
            "about the same event, mix topics (e.g. climate, disasters, wildlife, oceans, pollution, policy) "
            "and regions, and at most two stories focused on the same country. "
            "Use only ids that appear in the list."
        )},
        {"role": "user", "content": (
            f"Week: {start:%d %b %Y} to {(end - timedelta(days=1)):%d %b %Y} (Japan time).\n"
            f"Candidates (id | source | date | headline — teaser):\n" + "\n".join(lines) +
            f"\n\nCall select_top_stories with {TOP_N + RESERVES} ids, the {TOP_N} best first."
        )},
    ]
    args = call_groq(messages, SELECT_TOOL, "select_top_stories", api_key)

    by_id = {c["id"]: c for c in candidates}
    picks, used = [], set()
    for p in args.get("picks", []):
        cid = str(p.get("id", "")).strip()
        if cid in by_id and cid not in used:
            used.add(cid)
            picks.append({**by_id[cid], "reason": p.get("reason", "")})
    log.info("Editor picked %d valid stories: %s", len(picks), ", ".join(p["id"] for p in picks))

    # Top up with the newest unused candidates if the model returned too few.
    for c in candidates:
        if len(picks) >= TOP_N + RESERVES:
            break
        if c["id"] not in used:
            used.add(c["id"])
            picks.append({**c, "reason": "(added automatically as a reserve)"})
    return picks


# ---------------------------------------------------------------------------
# Step 2: read the article, find its image, write the story
# ---------------------------------------------------------------------------

_META_RE = re.compile(r"<meta\b[^>]*>", re.I)
_ATTR_RE = re.compile(r'([a-zA-Z:_-]+)\s*=\s*("([^"]*)"|\'([^\']*)\')')


def _meta_image(page_html: str, base_url: str) -> str | None:
    wanted = {"og:image": 0, "og:image:url": 0, "og:image:secure_url": 0, "twitter:image": 1, "twitter:image:src": 1}
    best = None
    for tag in _META_RE.findall(page_html[:300_000]):
        attrs = {m.group(1).lower(): (m.group(3) if m.group(3) is not None else m.group(4)) for m in _ATTR_RE.finditer(tag)}
        key = (attrs.get("property") or attrs.get("name") or "").lower()
        if key in wanted and attrs.get("content"):
            if best is None or wanted[key] < best[0]:
                best = (wanted[key], requests.compat.urljoin(base_url, html.unescape(attrs["content"])))
    return best[1] if best else None


def fetch_article(url: str) -> dict:
    """Returns {'text': article text or '', 'image': preview image URL or None}."""
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        page = resp.text
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not open article %s: %s", url, storage.short_error(exc, 150))
        return {"text": "", "image": None}
    text = trafilatura.extract(page, include_comments=False, include_tables=False, favor_precision=True) or ""
    return {"text": " ".join(text.split()), "image": _meta_image(page, resp.url)}


def download_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        resp.raise_for_status()
        if len(resp.content) > 15_000_000:
            return None
        img = Image.open(io.BytesIO(resp.content))
        img.load()
        if img.width < IMAGE_MIN_WIDTH:
            log.info("Image too small (%dpx wide): %s", img.width, url)
            return None
        return img
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not use image %s: %s", url, storage.short_error(exc, 150))
        return None


WRITE_TOOL = [{
    "type": "function",
    "function": {
        "name": "write_story",
        "description": "Submit the website copy for this news story.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Clear, factual headline in your own words, max 90 characters."},
                "description": {"type": "string", "description": "1-2 sentence teaser for the tile, max 200 characters."},
                "summary": {"type": "string", "description": "3-5 sentences: what happened, where, the key numbers, and why it matters. Max 900 characters."},
                "category": {"type": "string", "enum": CATEGORIES},
                "region": {"type": "string", "description": "Main country or region, e.g. 'Brazil', 'East Africa', 'Global'."},
            },
            "required": ["title", "description", "summary", "category", "region"],
        },
    },
}]


def _trim(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",;:-—")
    return cut + "…"


def write_story(pick: dict, article_text: str, api_key: str) -> dict:
    body = article_text[:ARTICLE_CHARS] if len(article_text) >= 300 else (pick["snippet"] + "\n" + article_text)
    messages = [
        {"role": "system", "content": (
            "You write short environment news briefs for the VaanAI website. Use ONLY facts stated in the "
            "article below — never add numbers, names, causes or quotes that are not in it. Write in your own "
            "words (do not copy sentences), neutral tone, plain English for a general audience. "
            "If the text is short, write a shorter summary rather than guessing."
        )},
        {"role": "user", "content": (
            f"Source: {pick['source']}\nPublished: {pick['published_at'].astimezone(JST):%d %B %Y}\n"
            f"Original headline: {pick['title']}\n\nArticle text:\n{body}\n\nCall write_story."
        )},
    ]
    a = call_groq(messages, WRITE_TOOL, "write_story", api_key)
    category = a.get("category") if a.get("category") in CATEGORIES else "climate"
    return {
        "title": _trim(a.get("title") or pick["title"], 110),
        "description": _trim(a.get("description") or pick["snippet"], 240),
        "summary": _trim(a.get("summary") or pick["snippet"], 1200),
        "category": category,
        "region": _trim(a.get("region") or "Global", 60),
    }


def placeholder_image(title: str) -> Image.Image:
    """Last resort when no picked story has a usable image: a blue VaanAI card."""
    w, h = 1200, 675
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        for x in range(0, w):
            t = (x / w + y / h) / 2
            px[x, y] = (int(10 + 20 * t), int(47 + 60 * t), int(107 + 110 * t))
    draw = ImageDraw.Draw(img)
    try:
        font_big, font_small = ImageFont.load_default(size=54), ImageFont.load_default(size=30)
    except TypeError:  # Pillow < 10.1
        font_big = font_small = ImageFont.load_default()
    draw.text((70, 70), "VaanAI · Environment news", fill=(190, 215, 255), font=font_small)
    words, line, y = title.split(), "", 260
    for word in words:
        if len(line) + len(word) > 34:
            draw.text((70, y), line, fill="white", font=font_big)
            line, y = "", y + 70
        line = f"{line} {word}".strip()
    draw.text((70, y), line, fill="white", font=font_big)
    return img


def build_stories(picks: list[dict], api_key: str) -> list[dict]:
    stories, no_image = [], []
    for pick in picks:
        if len(stories) == TOP_N:
            break
        log.info("Reading %s — %s", pick["source"], pick["title"])
        article = fetch_article(pick["url"])
        if len(article["text"]) < 300 and len(pick["snippet"]) < 80:
            log.warning("Skipping — article text unavailable and the feed teaser is too short.")
            continue

        img, img_from = None, None
        for candidate in (article["image"], pick["rss_image"]):
            if candidate and (img := download_image(candidate)) is not None:
                img_from = candidate
                break

        try:
            copy = write_story(pick, article["text"], api_key)
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping — writer failed: %s", storage.short_error(exc, 200))
            continue

        story = {**copy, "pick": pick, "image": img, "image_source_url": img_from}
        if img is None:
            no_image.append(story)       # keep, but prefer a reserve that has a real image
            log.info("No usable image — kept aside in case reserves run out.")
            continue
        stories.append(story)
        log.info("Accepted #%d: %s", len(stories), story["title"])

    # Not enough stories with images: fall back to imageless ones with a placeholder.
    for story in no_image:
        if len(stories) == TOP_N:
            break
        story["image"] = placeholder_image(story["title"])
        stories.append(story)
        log.warning("Using a placeholder image for: %s", story["title"])
    return stories


# ---------------------------------------------------------------------------
# Step 3: save to Supabase
# ---------------------------------------------------------------------------


def _to_jpeg(img: Image.Image) -> bytes:
    if img.mode not in ("RGB", "L"):
        background = Image.new("RGB", img.size, "white")
        rgba = img.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        img = background
    img = img.convert("RGB")
    if img.width > IMAGE_MAX_WIDTH:
        img = img.resize((IMAGE_MAX_WIDTH, round(img.height * IMAGE_MAX_WIDTH / img.width)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=84, optimize=True, progressive=True)
    return buf.getvalue()


def save_week(stories: list[dict], start: datetime, end: datetime) -> None:
    client = storage.get_supabase_client()
    table = client.table(TABLE)
    week_start = start.date().isoformat()
    week_end = (end - timedelta(days=1)).date().isoformat()

    old_paths = {
        r["image_path"]
        for r in storage.with_retries(
            "Reading this week's existing rows",
            lambda: table.select("image_path").eq("week_start", week_start).execute(),
        ).data
        if r.get("image_path")
    }

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for rank, s in enumerate(stories, 1):
        path = f"{week_start}/rank{rank}-{uuid.uuid4().hex[:10]}.jpg"
        data = _to_jpeg(s["image"])

        def _upload(path=path, data=data):
            client.storage.from_(BUCKET).upload(path, data, {"content-type": "image/jpeg", "upsert": "true"})

        storage.with_retries(f"Uploading image for rank {rank}", _upload)
        image_url = client.storage.from_(BUCKET).get_public_url(path)
        pick = s["pick"]
        rows.append({
            "week_start": week_start,
            "week_end": week_end,
            "rank": rank,
            "title": s["title"],
            "description": s["description"],
            "summary": s["summary"],
            "category": s["category"],
            "region": s["region"],
            "source_name": pick["source"],
            "source_url": pick["url"],
            "published_at": pick["published_at"].isoformat(),
            "image_url": image_url,
            "image_path": path,
            "image_credit": f"Image: {pick['source']}" if s["image_source_url"] else "Image: VaanAI",
            "modified_at": now,
        })

    # One row per (week_start, rank): a re-run for the same week updates the
    # rows in place — created_at is not sent, so it keeps its original value.
    response = storage.with_retries(
        "Saving weekly_top_news rows",
        lambda: table.upsert(rows, on_conflict="week_start,rank").execute(),
    )
    if not response.data or len(response.data) != len(rows):
        raise RuntimeError("Supabase wrote fewer rows than expected — check SUPABASE_KEY is the service_role "
                           "key and that sql/2026-10-06_weekly_top_news.sql has been run.")
    log.info("Saved %d stories for the week %s – %s.", len(rows), week_start, week_end)

    # Remove last run's images for this week that are no longer used.
    stale = sorted(old_paths - {r["image_path"] for r in rows})
    if stale:
        try:
            client.storage.from_(BUCKET).remove(stale)
            log.info("Removed %d replaced image(s).", len(stale))
        except Exception as exc:  # noqa: BLE001 - clean-up is best effort
            log.warning("Could not remove old images %s: %s", stale, storage.short_error(exc))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL, help=f"Groq model (default: {MODEL})")
    p.add_argument("--dry-run", action="store_true", help="Pick and write the stories, print them, save nothing.")
    p.add_argument("--week-end", type=date.fromisoformat,
                   help="Sunday (YYYY-MM-DD, Japan time) that ENDS the week to cover. Default: the most recent Sunday.")
    return p.parse_args()


def main() -> None:
    global MODEL
    load_dotenv()
    args = parse_args()
    MODEL = args.model

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY not set in .env. Get a free key at https://console.groq.com")
        sys.exit(1)

    start, end = week_window(week_end=args.week_end)
    log.info("Covering %s – %s (Japan time).", f"{start:%a %d %b %Y %H:%M}", f"{end:%a %d %b %Y %H:%M}")

    candidates = collect_candidates(start, end)
    log.info("%d candidate stories from %d feeds.", len(candidates), len(FEEDS))
    if len(candidates) < MIN_CANDIDATES:
        print(f"::error::Only {len(candidates)} candidate stories found — the news feeds may be down.")
        sys.exit(1)

    picks = select_stories(candidates, start, end, api_key)
    stories = build_stories(picks, api_key)
    if len(stories) < TOP_N:
        print(f"::error::Only {len(stories)} of {TOP_N} stories could be prepared — nothing saved.")
        sys.exit(1)

    print("\n=== TOP ENVIRONMENT NEWS ===")
    for rank, s in enumerate(stories, 1):
        print(f"{rank}. [{s['category']} · {s['region']}] {s['title']}\n   {s['description']}\n"
              f"   {s['pick']['source']} — {s['pick']['url']}")

    if args.dry_run:
        print("\n(Dry run — nothing saved.)")
        return

    try:
        save_week(stories, start, end)
    except Exception as exc:  # noqa: BLE001
        message = f"Saving the weekly news failed: {storage.short_error(exc)}"
        print(f"::error::{message}")
        log.error(message)
        sys.exit(1)


if __name__ == "__main__":
    main()
