"""
agentic_water_monitor_groq.py

Same agent loop as the Claude and Ollama versions, but running on Groq's
free, hosted API instead of a paid API or a locally-installed model.

Why this exists (vs. Ollama): Groq requires NO local install at all — no
GPU, no driver, no download. It's a plain HTTPS call to Groq's servers,
which host open-weight models (Llama 3.3, GPT-OSS, etc.) on fast hardware.
This makes it the version that actually works from a GitHub Actions
runner or any other machine with zero setup, unlike Ollama which needs
its own installed service running locally.

Prerequisites
-------------
1. Create a free account at https://console.groq.com (no card required)
2. Generate an API key from the console
3. Add it to your .env: GROQ_API_KEY=your_key_here

Usage
-----
    python agentic_water_monitor_groq.py \
        --location "Lake Mead, NV/AZ" \
        --bbox -114.75 36.00 -114.30 36.25 \
        --goal "Compare water extent between 2015 and 2023, decide if the change is significant, and explain a likely driver."
"""

import argparse
import calendar
import json
import logging
import math
import os
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests
from dotenv import load_dotenv

import extract_eo_data as eo
import compute_water_change as wc
import sentinel2_data as s2
import storage_utils as storage
import outline_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"  # confirmed available on this account; use --model to try openai/gpt-oss-120b for higher quality
AOI_BBOX = None
MEDIA_NEEDED = True  # set in main() from --media-needed; False = save the numbers only, no before/after images
IST = timezone(timedelta(hours=5, minutes=30))   # India Standard Time; the daily run starts at 00:00 IST
LOCATION_ID = None   # set in main() from --location-id; saved as water_insights.location_id
OUTLINE = None       # the water body's shoreline (GeoJSON) from monitored_locations_water.outline, if mapped
_OUTLINE_MASKS: dict = {}   # pixel size -> mask of the (widened) outline on the area grid
RUN_INDEX = None     # set in main() from --run-index; saved as water_insights.run_index (which yearly cycle this is)

# Quality gate: a result is only PUBLISHED if both pictures are mostly clear over
# the water body. Above this share of cloud/missing pixels the measurement is not
# trustworthy (a cloud-covered lake measures as "no water"), so nothing is saved.
MAX_CLOUD_TO_PUBLISH_PCT = 20
FOLDER_CLOUD: dict = {}   # scene folder -> % of the area hidden by cloud/snow/no-data
_skip_reason = None       # set when a run ends without a result for an expected reason (no clear picture)

# Month-vs-same-month-a-year-earlier comparison (set in main()).
#
# Cycle 1 (run_index 1, or no --run-index): the PREVIOUS calendar month of the
# run date is compared with the same month a year earlier, e.g. a run in October:
#   "after"  = September this year  -> the LATEST clear scene in 1-30 Sep
#   "before" = September last year  -> the LATEST clear scene in 1-30 Sep last year
# "Latest" = closest to the month's last day. Satellites don't image a place every
# day (Sentinel-2 ~5 days, HLS ~2-3 days) and clouds hide some days, so the scene
# picked is the last CLEAR one in the month.
#
# Cycle N > 1 steps one more year back: the before_date saved by cycle N-1 for the
# same location becomes this run's "after" target date, and "before" is the same
# date one year earlier. 5 cycles -> 5 yearly comparisons per water body.
#
# fetch_scene only accepts date ranges inside these two months.
WINDOWS: dict = {}   # {"after": (start, end, target), "before": (start, end, target)} as date objects

# Data source: NASA HLS (30 m) for large areas, Sentinel-2 L2A (10 m) for small
# ones such as village reservoirs. "auto" decides from the bbox size.
DATA_SOURCE = "hls"            # set in main(): "hls" or "s2"
DATA_RES = 30                  # metres per pixel of the chosen source
SMALL_AOI_KM2 = 400            # bbox smaller than this (~20 x 20 km) -> Sentinel-2 in "auto"
SOURCE_LABELS = {"hls": "NASA HLS Landsat (30 m)", "s2": "Sentinel-2 L2A (10 m)"}

# Results whose water area is under this many pixels are saved with confidence
# "low": ~1.0 km² at 30 m (HLS), ~0.11 km² at 10 m (Sentinel-2).
MIN_RELIABLE_PIXELS = 1100
MAX_TURNS = 16  # cloud/snow avoidance can legitimately need several retries per time period

# ---------------------------------------------------------------------------
# Tool definitions - OpenAI-compatible format (same shape Groq expects).
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_scene",
            "description": (
                "Find the clearest satellite scene over a bounding box within a date range, "
                "download what is needed to cover the box for that day, and return a local "
                "folder path plus scene_date (the actual day the picture was taken). The data "
                "source (NASA HLS 30 m for large areas, Sentinel-2 10 m for small ones) is chosen "
                "automatically. Use this whenever you need imagery for a new date range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "short_name": {"type": "string", "description": "Optional and ignored — the data source is chosen automatically."},
                    "min_lon": {"type": "number"},
                    "min_lat": {"type": "number"},
                    "max_lon": {"type": "number"},
                    "max_lat": {"type": "number"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["min_lon", "min_lat", "max_lon", "max_lat", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_water_extent",
            "description": (
                "Compute water surface area (km^2) for a previously fetched scene "
                "folder using NDWI, automatically excluding cloud/snow-covered pixels. "
                "Call after fetch_scene, passing the folder it returned. The result "
                "includes cloud_snow_masked_pct — if this is high (e.g. above ~10%), "
                "the result may be unreliable and you should consider fetching a "
                "different, clearer date instead of relying on it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {"type": "string"},
                    "green_band": {
                        "type": "string",
                        "description": "Zero-padded HLS band code for Green, e.g. 'B03'. Default: B03.",
                    },
                    "nir_band": {
                        "type": "string",
                        "description": "Zero-padded HLS band code for Near-Infrared, e.g. 'B05'. Default: B05.",
                    },
                },
                "required": ["folder"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_report",
            "description": (
                "Call this ONLY when confident enough to conclude. Ends the "
                "investigation — no more tools will be called after this. "
                "You must reference the exact folder paths and dates you used "
                "from your earlier fetch_scene/compute_water_extent calls, so "
                "the result and both images can be permanently saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "A plain-language write-up for the public, 130-200 words, in THREE short paragraphs "
                            "separated by a blank line: (1) what changed — the water area on both dates in km², the "
                            "change in km² and %, and the two dates in words (e.g. '27 September 2026'); "
                            "(2) what it means for the people, farms and industries that rely on this water body "
                            "(use the 'Used for' context given); (3) the most likely reasons (season, monsoon "
                            "rainfall, releases from upstream dams, irrigation or drinking-water draw-down, silt, "
                            "encroachment), worded as likely rather than certain, plus any caution in everyday words. "
                            "Do NOT name satellites, sensors, data products or agencies (no Sentinel, Landsat, HLS, "
                            "NASA, ESA) and avoid technical terms (no NDWI, index, pixels, bands, tiles, cloud mask); "
                            "say 'satellite pictures' instead."
                        ),
                    },
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                    "before_folder": {"type": "string", "description": "Folder path from the earlier (older-date) fetch_scene call."},
                    "after_folder": {"type": "string", "description": "Folder path from the later (newer-date) fetch_scene call."},
                    "before_date": {"type": "string", "description": "scene_date (YYYY-MM-DD) returned by the earlier fetch_scene call."},
                    "after_date": {"type": "string", "description": "scene_date (YYYY-MM-DD) returned by the later fetch_scene call."},
                    "before_value": {"type": "number", "description": "water_area_km2 result for the earlier scene."},
                    "after_value": {"type": "number", "description": "water_area_km2 result for the later scene."},
                },
                "required": [
                    "summary", "confidence", "before_folder", "after_folder",
                    "before_date", "after_date", "before_value", "after_value",
                ],
            },
        },
    },
]

def _granule_overlap(granule, bbox) -> float:
    """Fraction (0-1) of our bbox that this tile's footprint covers."""
    try:
        pts = granule["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"]["Points"]
    except (KeyError, IndexError, TypeError):
        return 0.0
    lons = [p["Longitude"] for p in pts]
    lats = [p["Latitude"] for p in pts]
    min_lon, min_lat, max_lon, max_lat = bbox
    w = max(0.0, min(max_lon, max(lons)) - max(min_lon, min(lons)))
    h = max(0.0, min(max_lat, max(lats)) - max(min_lat, min(lats)))
    return (w * h) / ((max_lon - min_lon) * (max_lat - min_lat))


def _cloud_cover(granule) -> float:
    for attr in granule["umm"].get("AdditionalAttributes", []):
        if attr.get("Name") == "CLOUD_COVERAGE":
            return float(attr["Values"][0])
    return 100.0


def _acq_day(granule) -> str:
    """'HLS.L30.T11SQA.2024258T181234.v2.0' -> '2024258' (year + day-of-year)."""
    return granule["umm"]["GranuleUR"].split(".")[3][:7]


def month_bounds(year: int, month: int) -> tuple[date, date]:
    """First and last day of a calendar month."""
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def set_comparison_windows(run_date: date) -> None:
    """Previous calendar month (relative to run_date) vs the same month one year
    earlier. The target in each is the month's last day, so the LATEST clear
    scene of each month is used."""
    prev_month_last_day = run_date.replace(day=1) - timedelta(days=1)
    y, m = prev_month_last_day.year, prev_month_last_day.month
    start, end = month_bounds(y, m)
    ly_start, ly_end = month_bounds(y - 1, m)
    WINDOWS["after"] = (start, end, end)
    WINDOWS["before"] = (ly_start, ly_end, ly_end)


def years_before(day: date, years: int) -> date:
    """Same calendar date `years` earlier (29 Feb -> 28 Feb in a non-leap year)."""
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, day=28)


def set_cycle_windows(after_target: date) -> None:
    """Cycle 2+: "after" = the month containing after_target, aiming for that exact
    day; "before" = the same date one year earlier, within its month."""
    before_target = years_before(after_target, 1)
    a_start, a_end = month_bounds(after_target.year, after_target.month)
    b_start, b_end = month_bounds(before_target.year, before_target.month)
    WINDOWS["after"] = (a_start, a_end, after_target)
    WINDOWS["before"] = (b_start, b_end, before_target)


def cycle_after_target(location_id: int | None, run_index: int, run_date: date) -> tuple[date, str]:
    """Works out the "after" date for cycle `run_index` of one location.

    Normally: the before_date saved by cycle run_index-1. If that row is missing
    (e.g. the earlier run was skipped because of cloud), it steps back from the
    newest earlier cycle that does exist, one year per missing cycle. With no
    earlier cycle at all, it counts back from cycle 1's month (the month before
    the run date)."""
    if location_id is not None:
        try:
            for row in storage.get_cycle_insights(location_id):   # newest cycle first
                k = row.get("run_index")
                if k is None or k >= run_index or not row.get("before_date"):
                    continue
                prev_before = datetime.strptime(str(row["before_date"])[:10], "%Y-%m-%d").date()
                return (years_before(prev_before, run_index - 1 - k),
                        f"cycle {k} before_date {prev_before}"
                        + ("" if k == run_index - 1 else f", stepped back {run_index - 1 - k} more year(s)"))
        except Exception as exc:  # noqa: BLE001 - fall back to the calculated date below
            log.warning("Could not read earlier cycles: %s", storage.short_error(exc))
    month_end = run_date.replace(day=1) - timedelta(days=1)
    return (years_before(month_end, run_index - 1),
            f"no earlier cycle found - counted back {run_index - 1} year(s) from {month_end:%B %Y}")


MAX_MONTH_SHIFTS = 2   # no clear picture in the month -> try up to 2 months earlier (same month in both years)


def months_before(day: date, months: int) -> date:
    """Same day `months` earlier; a month-end date stays a month-end date
    (30 Sep -> 31 Aug), other days are capped at the shorter month's end."""
    y, m = divmod(day.year * 12 + day.month - 1 - months, 12)
    m += 1
    last = calendar.monthrange(y, m)[1]
    is_month_end = day.day == calendar.monthrange(day.year, day.month)[1]
    return date(y, m, last if is_month_end else min(day.day, last))


def apply_month_shift(base_windows: dict, shift: int) -> None:
    """Moves both comparison months `shift` months earlier than base_windows."""
    for side, (_, _, target) in base_windows.items():
        t = months_before(target, shift)
        start, end = month_bounds(t.year, t.month)
        WINDOWS[side] = (start, end, t)


def period_instructions(run_date: date, shift: int) -> str:
    """The fixed-period part of the prompt, for the current WINDOWS."""
    a, b = WINDOWS["after"], WINDOWS["before"]
    a_label, b_label = a[0].strftime("%B %Y"), b[0].strftime("%B %Y")
    log.info("Run date %s, cycle %s -> comparing %s (%s..%s, target %s) with %s (%s..%s, target %s).",
             run_date, RUN_INDEX or 1, a_label, a[0], a[1], a[2], b_label, b[0], b[1], b[2])
    note = ("" if not shift else
            f"(The planned month had no clear satellite picture, so this comparison uses {a[0]:%B}, "
            f"{shift} month(s) earlier. Mention this in the summary.)\n")
    return (
        f"COMPARISON PERIOD IS FIXED — compare {a_label} with {b_label}, using the clear scene "
        "of each month closest to its target date:\n"
        f"- AFTER ({a_label}): call fetch_scene with start_date={a[0]}, end_date={a[1]} (target {a[2]}).\n"
        f"- BEFORE ({b_label}): call fetch_scene with start_date={b[0]}, end_date={b[1]} (target {b[2]}).\n"
        f"{note}"
        "fetch_scene automatically picks the clear day closest to the target date and reports "
        "days_from_target. If a scene turns out too cloudy, fetch again with a narrower range "
        "INSIDE the same month that leaves out that scene_date. Ranges outside these two months "
        "are rejected. Mention both scene dates and the months compared in your summary.\n\n"
        "Fetch both scenes, compute water extent for each, and only call finalize_report once "
        "confident in a conclusion."
    )


def _snap_to_window(args: dict):
    """Keeps a fetch_scene request inside one of the two comparison windows.
    Returns (start_iso, end_iso, target_iso) or an {"error": ...} dict."""
    if not WINDOWS:
        return args["start_date"], args["end_date"], None   # no fixed dates (manual/legacy use)
    try:
        start = datetime.strptime(args["start_date"], "%Y-%m-%d").date()
        end = datetime.strptime(args["end_date"], "%Y-%m-%d").date()
    except (KeyError, ValueError):
        return {"error": "start_date and end_date must be YYYY-MM-DD."}
    best = None
    for w_start, w_end, target in WINDOWS.values():
        s, e = max(start, w_start), min(end, w_end)
        overlap = (e - s).days
        if overlap >= 0 and (best is None or overlap > best[0]):
            best = (overlap, s, e, target)
    if best is None:
        a, b = WINDOWS["after"], WINDOWS["before"]
        return {"error": (f"Date range {start}..{end} is outside the comparison windows. Use "
                          f"{a[0]}..{a[1]} for the recent scene (target {a[2]}) or "
                          f"{b[0]}..{b[1]} for last year's scene (target {b[2]}).")}
    _, s, e, target = best
    if (s, e) != (start, end):
        log.info("Requested %s..%s -> limited to %s..%s (target date %s).", start, end, s, e, target)
    return s.isoformat(), e.isoformat(), target.isoformat()


def _no_scene_message(source: str, args: dict) -> str:
    global _skip_reason
    if not WINDOWS:
        return f"No usable {source} scenes found for that range. Try a different or wider date range."
    start = datetime.strptime(args["start_date"], "%Y-%m-%d").date()
    end = datetime.strptime(args["end_date"], "%Y-%m-%d").date()
    w_start, w_end, _ = next((w for w in WINDOWS.values() if w[0] <= start and end <= w[1]),
                             (start, end, None))
    _skip_reason = f"no clear picture found in {w_start:%B %Y}"
    if (start, end) == (w_start, w_end):
        return (f"No usable {source} scene exists anywhere in {start:%B %Y}. Do NOT retry this month. "
                "Stop now and explain that no comparison is possible for this month - the system will "
                "automatically try an earlier month.")
    rest = []
    if start > w_start:
        rest.append(f"{w_start}..{start - timedelta(days=1)}")
    if end < w_end:
        rest.append(f"{end + timedelta(days=1)}..{w_end}")
    return (f"No usable {source} scene in {start}..{end}, which is only PART of {w_start:%B %Y}. "
            f"The rest of the month ({' and '.join(rest)}) may still have a clear day: if you have not "
            "searched that part yet, fetch it (leave out scene dates already found too cloudy). If the "
            "whole month has now been searched without a clear picture, stop and explain that no "
            "comparison is possible for this month - the system will automatically try an earlier month.")


_TECH_TERMS = [
    (r"\b(?:the\s+)?(?:Copernicus\s+)?Sentinel[- ]?2[A-C]?(?:\s*L2A)?(?:\s*\(\s*\d+\s*m\s*\))?", "satellite"),
    (r"\b(?:NASA\s+)?(?:HLS|Harmonized Landsat(?: and)? Sentinel(?:[- ]?2)?)(?:\s*\(\s*\d+\s*m\s*\))?", "satellite"),
    (r"\bLandsat[- ]?\d*\b", "satellite"),
    (r"\b(?:NASA|ESA|USGS)\b", ""),
    (r"\b(?:NDWI|water[- ]index)[- ]based\s+", ""),
    (r"\bNDWI\b|\bnormali[sz]ed difference water index\b", "satellite measurement"),
    (r"\bcloud[- ]?mask(?:ed|ing)?\b", "cloud cover"),
    (r"\bpixels?\b", "image points"),
]


def clean_summary(text: str) -> str:
    """Safety net for the public write-up: removes satellite/product names and
    jargon the model may still use, and tidies the leftovers."""
    import re
    for pattern, repl in _TECH_TERMS:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    text = re.sub(r"\bsatellite(\s+satellite)+\b", "satellite", text, flags=re.IGNORECASE)
    text = re.sub(r"\bsatellite (imagery|images|scenes?|data)\b", "satellite pictures", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(satellite pictures)((?:,| and) satellite pictures)+\b", r"\1", text, flags=re.IGNORECASE)
    # capitalise sentence starts that a replacement may have lower-cased
    text = re.sub(r"(^|[.!?]\s+|\n\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\(\s*\)", "", text)
    return text.strip()


def _days_from(day_iso: str, target_iso: str | None):
    if not target_iso:
        return None
    return abs((datetime.strptime(day_iso, "%Y-%m-%d").date()
                - datetime.strptime(target_iso, "%Y-%m-%d").date()).days)


def _fetch_scene_s2(args: dict, target: str | None = None) -> dict:
    """Sentinel-2 L2A (10 m): picks the day closest to the target date (or the
    clearest, without a target) that is clear OVER THE BBOX ITSELF (not the
    whole tile), then saves only the bbox pixels — a few MB instead of ~120 MB."""
    bbox = (args["min_lon"], args["min_lat"], args["max_lon"], args["max_lat"])
    try:
        best = s2.best_scene(bbox, args["start_date"], args["end_date"],
                             max_candidates=12 if target else 6, target_date=target)
        if best is None:
            return {"no_scene": "Sentinel-2"}
        day, items, bad, covered = best
        folder = tempfile.mkdtemp(prefix="vaanai_scene_")
        _created_folders.append(folder)
        s2.download_scene(items, AOI_BBOX or bbox, folder)
        return {
            "folder": folder,
            "source": SOURCE_LABELS["s2"],
            "scene_date": day,
            "tile_count": len(items),
            "aoi_coverage_pct": round(covered * 100),
            "aoi_cloud_pct": round(bad * 100, 1),
            "target_date": target,
            "days_from_target": _days_from(day, target),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc) or repr(exc) or "Unknown error during Sentinel-2 fetch (see logs above)."}


def _window_for(start_iso: str, end_iso: str):
    """The comparison window (month) a snapped date range lies in, as (start, end) ISO strings."""
    for w_start, w_end, _ in WINDOWS.values():
        if w_start.isoformat() <= start_iso and end_iso <= w_end.isoformat():
            return w_start.isoformat(), w_end.isoformat()
    return None


def tool_fetch_scene(args: dict) -> dict:
    snapped = _snap_to_window(args)
    if isinstance(snapped, dict):
        return snapped   # error: range outside both comparison windows
    start_date, end_date, target = snapped
    args = {**args, "start_date": start_date, "end_date": end_date}
    result = _fetch_scene_once(args, target)

    # The model often asks for only part of the month (e.g. 15-30 Sep). If that
    # part has no clear picture, search the WHOLE month before giving up -
    # otherwise a clear day earlier in the month is never looked at.
    window = _window_for(start_date, end_date) if WINDOWS else None
    if "no_scene" in result and window and window != (start_date, end_date):
        log.info("No usable scene in %s..%s - searching the whole month %s..%s instead.",
                 start_date, end_date, *window)
        args = {**args, "start_date": window[0], "end_date": window[1]}
        result = _fetch_scene_once(args, target)
        if "folder" in result:
            result["note"] = (f"Nothing usable in {start_date}..{end_date}, so the whole month "
                              f"{window[0]}..{window[1]} was searched and this scene was found.")

    if "no_scene" in result:
        return {"error": _no_scene_message(result["no_scene"], args)}
    return result


def _fetch_scene_once(args: dict, target: str | None) -> dict:
    """One search + download for an already-snapped date range. Returns the scene,
    {"no_scene": <source>} when nothing usable exists in the range, or {"error": ...}."""
    if DATA_SOURCE == "s2":
        return _fetch_scene_s2(args, target)
    bbox = (args["min_lon"], args["min_lat"], args["max_lon"], args["max_lat"])
    short_name = "HLSL30"   # compute_water_extent assumes Landsat band codes (NIR = B05)
    try:
        # HLS downloads whole 110 km tiles (hundreds of MB), so prefer tiles that
        # are mostly clear. In the monsoon there may be none under 20% in a month,
        # so fall back to tiles up to 60% cloudy — compute_water_extent masks the
        # clouds and warns if the water body itself is covered.
        granules = []
        for max_cloud in (20, 60):
            granules = eo.search_granules(
                short_name=short_name, bbox=bbox,
                start_date=args["start_date"], end_date=args["end_date"],
                max_results=50, cloud_cover=max_cloud,
            )
            if granules:
                if max_cloud > 20:
                    log.info("No HLS tile under 20%% cloud in %s..%s - using tiles up to %d%%.",
                             args["start_date"], args["end_date"], max_cloud)
                break
        if not granules:
            return {"no_scene": "HLS"}

        # The bbox can span several tiles. Group tiles by the day they were
        # captured, then pick the day whose tiles together cover the bbox best,
        # and among those, the clearest. ALL tiles of that day are downloaded
        # and later stitched together onto one fixed grid.
        by_day: dict = {}
        for g in granules:
            by_day.setdefault(_acq_day(g), []).append(g)

        def day_cover(gs) -> float:
            return min(1.0, sum(_granule_overlap(g, bbox) for g in gs))

        def day_iso(d: str) -> str:
            return datetime.strptime(d, "%Y%j").date().isoformat()

        def day_score(d):
            gs = by_day[d]
            if target:   # same-date comparison: full coverage first, then CLOSEST to the target, then clearest
                return (day_cover(gs) >= 0.95, -_days_from(day_iso(d), target),
                        -max(_cloud_cover(g) for g in gs), day_cover(gs))
            return (day_cover(gs) >= 0.95, -max(_cloud_cover(g) for g in gs), day_cover(gs))

        best_day = max(by_day, key=day_score)
        chosen = by_day[best_day]

        folder = tempfile.mkdtemp(prefix="vaanai_scene_")
        _created_folders.append(folder)
        eo.download_granules(chosen, folder)

        return {
            "folder": folder,
            "source": SOURCE_LABELS["hls"],
            "scene_date": datetime.strptime(best_day, "%Y%j").date().isoformat(),
            "tile_count": len(chosen),
            "aoi_coverage_pct": round(day_cover(chosen) * 100),
            "tile_cloud_pct": max(_cloud_cover(g) for g in chosen),
            "target_date": target,
            "days_from_target": _days_from(day_iso(best_day), target),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc) or repr(exc) or "Unknown error during fetch (see logs above)."}


def tool_compute_water_extent(args: dict) -> dict:
    folder = args["folder"]
    green_band = args.get("green_band", "B03")
    nir_band = args.get("nir_band", "B05")
    try:
        bad_mask, bad_pct = wc.read_bad_pixel_mask(folder, bbox=AOI_BBOX)
        green_files = wc.find_band_files(folder, green_band)
        ndwi = wc.compute_ndwi(
            green_files,
            wc.find_band_files(folder, nir_band),
            bad_pixel_mask=bad_mask, bbox=AOI_BBOX,
        )
        res = wc.band_resolution(green_files[0])   # 30 m (HLS) or 10 m (Sentinel-2)
        if OUTLINE is not None and AOI_BBOX is not None:
            # Count only this water body: water inside its outline (+ a margin), and
            # judge cloud over the water body itself, not over the whole picture.
            if res not in _OUTLINE_MASKS:
                _OUTLINE_MASKS[res] = outline_utils.outline_mask(OUTLINE, AOI_BBOX, res)
            inside = _OUTLINE_MASKS[res]
            if inside.shape == ndwi.shape and inside.any():
                ndwi = np.where(inside, ndwi, np.nan)
                bad_pct = float(bad_mask[inside].mean())
            else:
                log.warning("Outline does not fit the area grid - measuring the whole area instead.")
        FOLDER_CLOUD[folder] = round(bad_pct * 100, 1)
        result = {
            "water_area_km2": round(wc.water_area_km2(ndwi, pixel_area_m2=res * res), 3),   # 3 decimals: small ponds are < 0.01 km²
            "cloud_snow_masked_pct": round(bad_pct * 100, 1),
        }
        if bad_pct * 100 > MAX_CLOUD_TO_PUBLISH_PCT:
            result["warning"] = (
                f"UNUSABLE: {bad_pct*100:.0f}% of the area is hidden by cloud, so this value is not a real "
                f"measurement and a report using it will NOT be saved (limit {MAX_CLOUD_TO_PUBLISH_PCT}%). "
                "Fetch a different day inside the same month that avoids this scene_date."
            )
        elif bad_pct > 0.10:
            result["warning"] = (
                f"{bad_pct*100:.0f}% of this scene is cloud-covered and was excluded. Usable, but a clearer "
                "day in the same month would be better if one exists."
            )
        return result
    except ValueError as exc:
        return {"error": str(exc) or "Band lookup failed (see logs above)."}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc) or repr(exc) or "Unknown error computing water extent (see logs above)."}


TOOL_FUNCTIONS = {
    "fetch_scene": tool_fetch_scene,
    "compute_water_extent": tool_compute_water_extent,
}

_created_folders: list[str] = []  # tracks every temp folder this run creates, for cleanup at the end
_insight_saved = False  # set True only once a result is actually written to Supabase; main() exits with
                        # an error if it's still False, so GitHub marks the run red and emails you


def cleanup_temp_folders() -> None:
    """Deletes every temp folder this run created. Harmless on GitHub Actions
    (the whole machine is destroyed anyway) but keeps a local machine's disk
    clean across many weekly runs."""
    import shutil

    for folder in _created_folders:
        shutil.rmtree(folder, ignore_errors=True)
    log.info("Cleaned up %d temp folder(s).", len(_created_folders))
    _created_folders.clear()


VALID_TOOL_NAMES = {"fetch_scene", "compute_water_extent", "finalize_report"}


def _try_recover_corrupted_tool_call(response) -> dict | None:
    """GPT-OSS on Groq occasionally leaks an internal formatting token (e.g.
    'finalize_report<|channel|>commentary') into the tool call's *name* field
    — a known harmony-format parsing issue. The arguments it generated are
    usually still fully correct and complete; only the name is corrupted.
    Rather than discard a fully-computed result (which, for this pipeline,
    means repeating expensive multi-minute NASA downloads), try to recover
    the intended call directly from the error body. Returns None if recovery
    isn't possible, so the caller can fall back to a normal retry."""
    try:
        error_body = response.json().get("error", {})
        failed_generation = error_body.get("failed_generation")
        if not failed_generation:
            return None

        parsed = json.loads(failed_generation)
        raw_name = parsed.get("name", "")
        clean_name = raw_name.split("<|channel|>")[0].split("<|")[0].strip()

        if clean_name not in VALID_TOOL_NAMES:
            return None

        log.warning("Recovered a corrupted tool call name: '%s' -> '%s'", raw_name, clean_name)
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "recovered_call_1",
                                "type": "function",
                                "function": {
                                    "name": clean_name,
                                    "arguments": json.dumps(parsed.get("arguments", {})),
                                },
                            }
                        ],
                    }
                }
            ]
        }
    except Exception:  # noqa: BLE001 - recovery is best-effort; fall back on any parse issue
        return None


# ---------------------------------------------------------------------------
# Groq token usage. Every model reply carries a "usage" block; it is recorded
# per call so each run can print what it used (per call, per minute, total) and
# save it to USAGE_FILE for the workflow's end-of-night summary (usage_report.py).
# Calls Groq rejected (rate limited, or broken output it could not parse) carry
# no usage block, so they are only counted, not measured.
# ---------------------------------------------------------------------------
USAGE_CALLS: list[dict] = []   # one entry per successful model call
USAGE_FAILED_CALLS = 0         # rejected calls (429 / unparseable output)
USAGE_LIMITS: dict = {}        # latest x-ratelimit-* headers from Groq


def _record_usage(response) -> None:
    global USAGE_LIMITS
    usage = (response.json() or {}).get("usage") or {}
    USAGE_CALLS.append({
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    })
    USAGE_LIMITS = {k.lower(): v for k, v in response.headers.items() if k.lower().startswith("x-ratelimit")}
    c = USAGE_CALLS[-1]
    log.info("Groq call %d: %d tokens (%d in, %d out). Left this minute: %s tokens; left today: %s requests.",
             len(USAGE_CALLS), c["total_tokens"], c["prompt_tokens"], c["completion_tokens"],
             USAGE_LIMITS.get("x-ratelimit-remaining-tokens", "?"),
             USAGE_LIMITS.get("x-ratelimit-remaining-requests", "?"))


def per_minute(calls: list[dict]) -> dict:
    """{"HH:MM" (UTC): tokens} for the given calls."""
    out: dict = {}
    for c in calls:
        minute = c["time"][11:16]
        out[minute] = out.get(minute, 0) + c["total_tokens"]
    return out


def peak_60s(calls: list[dict]) -> int:
    """Most tokens used inside any rolling 60 seconds (what the per-minute limit counts)."""
    times = [(datetime.fromisoformat(c["time"]), c["total_tokens"]) for c in calls]
    best = 0
    for t0, _ in times:
        best = max(best, sum(n for t, n in times if t0 <= t < t0 + timedelta(seconds=60)))
    return best


def report_usage(location: str, location_id, saved: bool) -> None:
    """Prints this run's Groq usage and saves it to USAGE_FILE (default groq_usage.json)."""
    total = sum(c["total_tokens"] for c in USAGE_CALLS)
    p_in = sum(c["prompt_tokens"] for c in USAGE_CALLS)
    p_out = sum(c["completion_tokens"] for c in USAGE_CALLS)
    minutes = per_minute(USAGE_CALLS)
    lines = [
        f"=== GROQ TOKEN USAGE: {location} ===",
        f"Model: {MODEL}",
        f"Calls: {len(USAGE_CALLS)} answered, {USAGE_FAILED_CALLS} rejected (rate limit / broken output, not measured)",
        f"Tokens: {total:,} total ({p_in:,} in, {p_out:,} out)",
        f"Busiest 60 seconds: {peak_60s(USAGE_CALLS):,} tokens",
        "Per minute (UTC): " + (", ".join(f"{m} {n:,}" for m, n in minutes.items()) or "-"),
    ]
    if USAGE_LIMITS:
        lines.append("Groq limits: " + ", ".join(f"{k.replace('x-ratelimit-', '')}={v}"
                                                  for k, v in sorted(USAGE_LIMITS.items())))
    print("\n" + "\n".join(lines))
    try:
        with open(os.getenv("USAGE_FILE", "groq_usage.json"), "w", encoding="utf-8") as f:
            json.dump({"location": location, "location_id": location_id, "run_index": RUN_INDEX,
                       "model": MODEL, "saved": saved, "calls": USAGE_CALLS,
                       "failed_calls": USAGE_FAILED_CALLS, "limits": USAGE_LIMITS}, f, indent=1)
    except OSError as exc:
        log.warning("Could not write the usage file: %s", exc)


def call_groq(messages: list, api_key: str, max_retries: int = 3, tool_choice="auto") -> dict:
    """Calls Groq, automatically waiting and retrying on free-tier rate limits (429),
    and on the GPT-OSS-specific 'output_parse_failed' error (a known Groq issue where
    the model's internal reasoning leaks into the response and breaks tool-call
    parsing) — both are common on the free tier and worth handling gracefully."""
    for attempt in range(max_retries + 1):
        response = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": tool_choice,
                # GPT-OSS-specific settings (per Groq docs) that keep the model's
                # internal reasoning out of the response body, which otherwise
                # sometimes breaks structured tool-call parsing:
                "reasoning_format": "hidden",
                "reasoning_effort": "low",
            },
            timeout=60,
        )

        global USAGE_FAILED_CALLS
        if not response.ok:
            USAGE_FAILED_CALLS += 1
        if response.status_code == 429 and attempt < max_retries:
            wait_seconds = 10 * (attempt + 1)  # simple backoff: 10s, 20s, 30s
            log.warning(
                "Rate limited (attempt %d/%d). Waiting %ds before retrying...",
                attempt + 1, max_retries, wait_seconds,
            )
            time.sleep(wait_seconds)
            continue

        is_known_groq_glitch = response.status_code == 400 and (
            "tool_use_failed" in response.text or "output_parse_failed" in response.text
        )

        if is_known_groq_glitch:
            recovered = _try_recover_corrupted_tool_call(response)
            if recovered is not None:
                return recovered

            # Recovery wasn't possible (output was too corrupted/truncated to parse
            # at all, not just mislabeled) — this is a known intermittent GPT-OSS
            # generation glitch on Groq, and a plain retry usually produces clean
            # output the second time, since it isn't a deterministic/repeatable failure.
            if attempt < max_retries:
                log.warning(
                    "Model produced corrupted/unparseable tool output (attempt %d/%d) "
                    "— retrying, this is a known intermittent GPT-OSS/Groq issue.",
                    attempt + 1, max_retries,
                )
                time.sleep(3)
                continue

        if not response.ok:
            log.error("Groq API error %s: %s", response.status_code, response.text)
        response.raise_for_status()
        _record_usage(response)
        return response.json()

    raise RuntimeError("Exceeded max retries after repeated rate limiting / parse failures.")


def persist_result(final_args: dict, location: str) -> None:
    """Saves one row into the `water_insights` table — called once finalize_report
    is invoked. If MEDIA_NEEDED, also generates before/after preview images and
    uploads them to Supabase Storage; otherwise the image URL columns stay empty.
    Keeps the model's own reported values as the source of truth for what to
    save, since it's the one that knows which folders/dates it used."""
    import tempfile as _tempfile

    before_url = after_url = None
    if MEDIA_NEEDED:
        before_dir = _tempfile.mkdtemp(prefix="vaanai_preview_")
        after_dir = _tempfile.mkdtemp(prefix="vaanai_preview_")
        _created_folders.extend([before_dir, after_dir])
        # JPEG, not PNG: a lossless PNG of a large area can exceed 20 MB (see
        # storage_utils.generate_preview_png); JPEG keeps previews around 1 MB.
        before_png = os.path.join(before_dir, "before.jpg")
        after_png = os.path.join(after_dir, "after.jpg")

        storage.generate_preview_png(final_args["before_folder"], before_png, bbox=AOI_BBOX, outline=OUTLINE)
        storage.generate_preview_png(final_args["after_folder"], after_png, bbox=AOI_BBOX, outline=OUTLINE)

        before_url = storage.upload_image(before_png)
        after_url = storage.upload_image(after_png)
    else:
        log.info("media_needed = false for %s — saving the numbers only, no images.", location)

    before_value = final_args["before_value"]
    after_value = final_args["after_value"]
    change_pct = ((after_value - before_value) / before_value * 100) if before_value else 0.0

    # Small water bodies are only a few hundred pixels, so a handful of
    # shoreline pixels can swing the % change a lot. Never publish those as
    # "high"/"medium" confidence, whatever the model says.
    min_reliable_km2 = MIN_RELIABLE_PIXELS * DATA_RES * DATA_RES / 1_000_000
    confidence = final_args["confidence"]
    if max(before_value, after_value) < min_reliable_km2 and confidence != "low":
        log.warning(
            "Water area below %.2f km² at %d m (%.3f -> %.3f) — lowering confidence from '%s' to 'low'.",
            min_reliable_km2, DATA_RES, before_value, after_value, confidence,
        )
        confidence = "low"

    record = {
        "location": location,
        "before_date": final_args["before_date"],
        "after_date": final_args["after_date"],
        "before_value": before_value,
        "after_value": after_value,
        "change_pct": round(change_pct, 2),
        "summary": final_args["summary"],
        "confidence": confidence,
        "before_image_url": before_url,
        "after_image_url": after_url,
        # link to monitored_locations_water.id (None for manual runs without --location-id)
        "location_id": LOCATION_ID,
    }
    if RUN_INDEX is not None:
        record["run_index"] = RUN_INDEX   # which yearly cycle this result belongs to
    storage.save_insight(record)


def _cloud_of(folder: str):
    """% cloud over the area for a scene folder (measures it if not done yet)."""
    if folder and folder not in FOLDER_CLOUD and os.path.isdir(folder):
        tool_compute_water_extent({"folder": folder})
    return FOLDER_CLOUD.get(folder)


def _handle_finalize(tool_args: dict, location: str) -> None:
    global _insight_saved, _skip_reason
    # Quality gate — enforced in code, whatever the model decided.
    for side in ("before", "after"):
        cloud = _cloud_of(tool_args.get(f"{side}_folder"))
        if cloud is None or cloud > MAX_CLOUD_TO_PUBLISH_PCT:
            _skip_reason = (f"the {side} picture ({tool_args.get(f'{side}_date')}) was "
                            f"{'unmeasured' if cloud is None else f'{cloud:.0f}% cloud-covered'} over the area "
                            f"(limit {MAX_CLOUD_TO_PUBLISH_PCT}%) - not published, to avoid a false result")
            log.warning("Not saving the result for %s: %s.", location, _skip_reason)
            return
    tool_args["summary"] = clean_summary(tool_args.get("summary", ""))
    print("\n=== FINAL REPORT ===")
    print(f"Confidence: {tool_args.get('confidence', 'unknown')}")
    print(tool_args.get("summary", ""))
    try:
        persist_result(tool_args, location=location)
        _insight_saved = True
        print("(Saved image pair + result to Supabase.)" if MEDIA_NEEDED else "(Saved result to Supabase - no images, media_needed = false.)")
    except Exception as exc:  # noqa: BLE001 - don't let a save failure hide the report itself
        msg = storage.short_error(exc)   # one line, not a whole Cloudflare HTML page
        log.error("Failed to persist insight to Supabase: %s", msg)
        print(f"(Warning: could not save to Supabase — {msg})")

def _compute_missing(folder_dates: dict, computed: dict, rejected: set) -> None:
    """The model sometimes fetches a scene and then stops (or writes its report)
    without ever calling compute_water_extent on it — e.g. Lake Powell, where it
    fetched the 2023 scene, never measured it, and simply copied the 2024 value
    as the 'before' number. The download is already done, so measure any such
    folder ourselves instead of throwing the run away."""
    for folder in folder_dates:
        if folder in computed:
            continue
        log.warning("Scene %s was fetched but never measured — measuring it now.", folder)
        result = tool_compute_water_extent({"folder": folder})
        log.info("Tool result: %s", result)
        if "water_area_km2" in result:
            computed[folder] = result["water_area_km2"]
            if "warning" in result:
                rejected.add(folder)


def _extract_report_from_text(text: str, computed: dict):
    """If the model wrote its finalize_report arguments as text instead of a tool
    call, recover them — but only if the folders really were computed, and use
    the computed numbers (not the model's typed ones) as the source of truth."""
    if not text or "{" not in text:
        return None
    try:
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    required = {"summary", "confidence", "before_folder", "after_folder",
                "before_date", "after_date", "before_value", "after_value"}
    if not required.issubset(data) or data["before_folder"] not in computed \
            or data["after_folder"] not in computed:
        return None
    # If the numbers the model typed don't match what was really measured, its
    # summary was written about made-up values — don't publish that text.
    # Returning None hands over to _force_finalize, which shows the model the
    # real measurements and asks it to write the report again.
    for side in ("before", "after"):
        real = computed[data[f"{side}_folder"]]
        try:
            typed = float(data[f"{side}_value"])
        except (TypeError, ValueError):
            return None
        if abs(typed - real) > max(0.01 * abs(real), 0.01):
            log.warning("Model's text report says %s_value=%s but the measured value is %s — "
                        "not using its summary.", side, typed, real)
            return None
    data["before_value"] = computed[data["before_folder"]]
    data["after_value"] = computed[data["after_folder"]]
    return data

def _force_finalize(messages: list, api_key: str, computed: dict, folder_dates: dict, location: str, rejected=frozenset()) -> bool:
    """Fallback for when the model has already computed enough evidence but
    stopped without calling finalize_report (e.g. it wrote a prose summary
    instead). Rather than lose a fully correct, already-computed result,
    nudge it once more with tool_choice FORCED to finalize_report specifically.
    Returns True if a report was successfully forced and saved."""
    # Only consider folders that were actually computed, not just fetched —
    # a folder can be fetched and then abandoned (e.g. the agent moved on to
    # try a different date) without ever being analyzed, and picking that as
    # "after" would silently use a result that doesn't exist.
    # Skip scenes whose result came with a cloud/snow warning, so a rejected
    # cloudy scene can never be picked as "before" or "after".
    # Only clear scenes may be used — never fall back to cloudy ones (a cloud-covered
    # lake measures as "no water" and produced a false 100% change once).
    candidates = {f: d for f, d in folder_dates.items()
                  if f in computed and (FOLDER_CLOUD.get(f) or 0) <= MAX_CLOUD_TO_PUBLISH_PCT}
    if len(candidates) < 2:
        return False

    ordered = sorted(candidates.items(), key=lambda kv: kv[1]["start_date"])
    before_folder, before_meta = ordered[0]
    after_folder, after_meta = ordered[-1]

    log.warning("Model stopped without calling finalize_report despite having enough evidence — forcing it.")
    messages.append(
        {
            "role": "user",
            "content": (
                "You already computed these results:\n"
                f"- scene_date={before_meta.get('scene_date') or before_meta['start_date']}: "
                f"folder={before_folder}, water_area_km2={computed[before_folder]}\n"
                f"- scene_date={after_meta.get('scene_date') or after_meta['start_date']}: "
                f"folder={after_folder}, water_area_km2={computed[after_folder]}\n\n"
                "Call finalize_report now using these exact folders, values and scene dates."
            ),
        }
    )

    # Same wait-and-retry on rate limits (429) as every other model call — the
    # forced call used to give up on the first 429, losing a finished result.
    try:
        response = call_groq(
            messages, api_key,
            tool_choice={"type": "function", "function": {"name": "finalize_report"}},
        )
        message = response["choices"][0]["message"]
        for call in message.get("tool_calls") or []:
            if call["function"]["name"] == "finalize_report":
                tool_args = json.loads(call["function"]["arguments"])
                # Trust our own measurements over numbers the model retyped.
                if tool_args.get("before_folder") in computed:
                    tool_args["before_value"] = computed[tool_args["before_folder"]]
                if tool_args.get("after_folder") in computed:
                    tool_args["after_value"] = computed[tool_args["after_folder"]]
                _handle_finalize(tool_args, location)
                return True
        log.warning("Forced finalize returned no finalize_report call.")
    except Exception as exc:  # noqa: BLE001 - fall through to the no-model save below
        log.error("Forced finalize call failed: %s", exc)

    # Last resort: the measurements are already done, so save them without the
    # model. The summary is a plain factual sentence (no guessed driver), and
    # confidence is "medium" at most since the model never confirmed the pair.
    log.warning("Saving the computed result directly, without a model-written summary.")
    before_date = before_meta.get("scene_date") or before_meta["start_date"]
    after_date = after_meta.get("scene_date") or after_meta["start_date"]
    b, a = computed[before_folder], computed[after_folder]
    pct = ((a - b) / b * 100) if b else 0.0
    dec = 3 if max(a, b) < 1 else 2   # small ponds: 0.023 km², not "0.02"
    _handle_finalize(
        {
            "summary": (
                f"Satellite pictures show the water surface at {location} measured {b:.{dec}f} km² on {before_date} "
                f"and {a:.{dec}f} km² on {after_date}, a change of {a - b:+.{dec}f} km² ({pct:+.1f}%).\n\n"
                "This compares the same month one year apart, so normal seasonal ups and downs are largely "
                "taken out of the comparison. A detailed explanation of the likely causes could not be "
                "generated for this reading."
            ),
            "confidence": "medium",
            "before_folder": before_folder, "after_folder": after_folder,
            "before_date": before_date, "after_date": after_date,
            "before_value": b, "after_value": a,
        },
        location,
    )
    return True


def _note_cloudy_only() -> None:
    """Ending without a result after finding only cloud-covered pictures counts
    as "no clear picture" (so main() tries an earlier month), not as an error."""
    global _skip_reason
    if not _skip_reason and any(c > MAX_CLOUD_TO_PUBLISH_PCT for c in FOLDER_CLOUD.values()):
        _skip_reason = "only cloud-covered pictures were found"


def run_agent(goal: str, api_key: str, location: str) -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful research analyst investigating satellite data. "
                "Use the available tools to gather evidence before concluding. "
                "If a computed result has a high cloud_snow_masked_pct or comes with a "
                "warning, don't trust it — fetch a different date range instead of "
                "concluding from a heavily clouded scene. However, budget your attempts: "
                "try at most 2-3 scenes per time period before accepting the clearest one "
                "you've found so far and moving on to the other period — don't exhaust all "
                "your attempts repeatedly retrying a single period while neglecting the other. "
                "You need at least one usable result from EACH period to conclude anything."
                "Once you have computed both before and after values from clear scenes, "
                "you MUST call finalize_report as a tool call — do not just write your "
                "conclusion as text."
            ),
        },
        {"role": "user", "content": goal},
    ]

    computed: dict = {}       # folder -> water_area_km2
    rejected: set = set()     # folders whose result came with a cloud/snow warning
    folder_dates: dict = {}   # folder -> {"start_date":..., "end_date":...}

    for turn in range(1, MAX_TURNS + 1):
        log.info("--- Turn %d: asking the model what to do next ---", turn)
        response = call_groq(messages, api_key)
        message = response["choices"][0]["message"]
        messages.append(message)

        if message.get("content"):
            print(f"\n[Model]: {message['content'].strip()}")

        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            _compute_missing(folder_dates, computed, rejected)
            recovered = _extract_report_from_text(message.get("content"), computed)
            if recovered:
                log.warning("Model wrote its report as text — using its own chosen scenes.")
                _handle_finalize(recovered, location)
                return
            if _force_finalize(messages, api_key, computed, folder_dates, location, rejected):
                return
            _note_cloudy_only()
            log.info("Model stopped, and fewer than two scenes were measured — nothing to compare; ending.")
            return

        stop = False
        for call in tool_calls:
            name = call["function"]["name"]
            tool_args = json.loads(call["function"]["arguments"])
            call_id = call["id"]

            log.info("Model called tool: %s(%s)", name, tool_args)

            if name == "finalize_report":
                _handle_finalize(tool_args, location)
                stop = True
                break

            func = TOOL_FUNCTIONS.get(name)
            result = func(tool_args) if func else {"error": f"Unknown tool '{name}'"}
            log.info("Tool result: %s", result)
            if "unexpected keyword argument" in str(result.get("error", "")):
                log.error("Code error in a tool — stopping instead of retrying: %s", result["error"])
                return

            # Track evidence ourselves so we can force a conclusion later if the
            # model computes everything it needs but forgets the final tool call.
            if name == "fetch_scene" and "folder" in result:
                folder_dates[result["folder"]] = {
                    "start_date": tool_args.get("start_date"),
                    "end_date": tool_args.get("end_date"),
                    "scene_date": result.get("scene_date"),
                }
            if name == "compute_water_extent" and "water_area_km2" in result:
                computed[tool_args["folder"]] = result["water_area_km2"]
                if "warning" in result:
                    rejected.add(tool_args["folder"])

            # Groq/OpenAI format requires tool_call_id to match the call being answered.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result),
                }
            )

        if stop:
            return

    _compute_missing(folder_dates, computed, rejected)
    if _force_finalize(messages, api_key, computed, folder_dates, location, rejected):
        return
    _note_cloudy_only()
    log.warning("Hit MAX_TURNS (%d) without a final report. Ending.", MAX_TURNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", required=True)
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"), required=True)
    parser.add_argument(
        "--goal",
        default="Investigate how water extent has changed over time in this area and explain why.",
    )
    parser.add_argument("--model", default=MODEL, help=f"Groq model to use (default: {MODEL})")
    parser.add_argument(
        "--source", choices=["auto", "hls", "s2"], default="auto",
        help=f"Satellite data: 'hls' = NASA HLS 30 m, 's2' = Sentinel-2 10 m, "
             f"'auto' (default) = Sentinel-2 if the bbox is under {SMALL_AOI_KM2} km², else HLS.",
    )
    parser.add_argument(
        "--media-needed", choices=["true", "false"], default="true", type=str.lower,
        help="'true' (default) = also save before/after preview images; 'false' = save the numbers only.",
    )
    parser.add_argument(
        "--run-date", default=None, metavar="YYYY-MM-DD",
        help="Pretend the run happens on this date (default: today in India time, IST). In cycle 1 the PREVIOUS month of "
             "this date is compared with the same month a year earlier, e.g. 2026-10-12 -> Sep 2026 vs Sep 2025. "
             "Use 'none' to let the model choose its own dates (old behaviour).",
    )
    parser.add_argument(
        "--location-id", type=int, default=None,
        help="id of the row in monitored_locations_water; when given, its last_run_at is updated after the run.",
    )
    parser.add_argument(
        "--run-index", type=int, default=None,
        help="Yearly cycle number from monitored_locations_water.run_index (1-5). 1 = previous month vs "
             "a year earlier; N > 1 = continues one year further back from cycle N-1's before_date "
             "(needs --location-id). Saved as water_insights.run_index.",
    )
    return parser.parse_args()


def bbox_area_km2(bbox) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2)
    return (max_lon - min_lon) * 111.32 * math.cos(mid_lat) * (max_lat - min_lat) * 110.57


def main() -> None:
    global MODEL, AOI_BBOX, DATA_SOURCE, DATA_RES, MEDIA_NEEDED, LOCATION_ID, RUN_INDEX, OUTLINE
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY not set in .env. Get a free key at https://console.groq.com")
        sys.exit(1)

    args = parse_args()
    MODEL = args.model
    AOI_BBOX = tuple(args.bbox)
    MEDIA_NEEDED = args.media_needed == "true"
    LOCATION_ID = args.location_id
    RUN_INDEX = args.run_index
    if RUN_INDEX is not None and RUN_INDEX < 1:
        log.error("--run-index must be 1 or more (got %s).", RUN_INDEX)
        sys.exit(1)

    area = bbox_area_km2(AOI_BBOX)
    DATA_SOURCE = args.source if args.source != "auto" else ("s2" if area < SMALL_AOI_KM2 else "hls")
    DATA_RES = 10 if DATA_SOURCE == "s2" else 30
    log.info("Area %.1f km² -> using %s.", area, SOURCE_LABELS[DATA_SOURCE])

    if DATA_SOURCE == "hls":
        eo.authenticate()  # NASA login is only needed for HLS; Sentinel-2 is open, no login

    min_lon, min_lat, max_lon, max_lat = args.bbox
    # Local context for the public write-up (who depends on this water body).
    context = ""
    if args.location_id is not None:
        try:
            loc = storage.get_location(args.location_id) or {}
            OUTLINE = outline_utils.parse_outline(loc.get("outline"))
            log.info("Outline: %s", "mapped shoreline found - measuring this water body only" if OUTLINE
                     else "none saved - measuring all water in the area (run fix_outlines.py)")
            where = ", ".join(x for x in (loc.get("place"), loc.get("district"), loc.get("state"), loc.get("country")) if x)
            context = (f"Where: {where}\n" if where else "") \
                + (f"Type of water body: {loc['type']}\n" if loc.get("type") else "") \
                + (f"Used for: {loc['uses']}\n" if loc.get("uses") else "")
        except Exception as exc:  # noqa: BLE001 - context is nice to have, never fatal
            log.warning("Could not read location details: %s", storage.short_error(exc))

    goal_prompt = (
        f"Location: {args.location}\n"
        f"{context}"
        f"Bounding box: min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}\n\n"
        f"Task: {args.goal}\n\n"
    )

    fixed_dates = (args.run_date or "").lower() != "none"
    if fixed_dates:
        run_date = (datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date
                    # IST, not UTC: the workflow starts at 00:00 IST, which is still the
                    # previous day in UTC (on the 1st that would pick the wrong month).
                    else datetime.now(IST).date())
        if RUN_INDEX is not None and RUN_INDEX > 1:
            after_target, why = cycle_after_target(LOCATION_ID, RUN_INDEX, run_date)
            set_cycle_windows(after_target)
            log.info("Cycle %d: after target %s (%s).", RUN_INDEX, after_target, why)
        else:
            set_comparison_windows(run_date)
        base_windows = dict(WINDOWS)

    global _skip_reason
    try:
        # With fixed dates: if a month has no clear picture (cloud, monsoon), try
        # again with BOTH dates moved back one month - same month in both years,
        # so the seasons still match - up to MAX_MONTH_SHIFTS times.
        for shift in range(MAX_MONTH_SHIFTS + 1 if fixed_dates else 1):
            prompt = goal_prompt
            if fixed_dates:
                apply_month_shift(base_windows, shift)
                if shift:
                    log.warning("Attempt %d: %s - trying one month earlier.", shift + 1, _skip_reason)
                    cleanup_temp_folders()   # free the disk before downloading again
                    FOLDER_CLOUD.clear()
                    _skip_reason = None
                prompt += period_instructions(run_date, shift)
            else:
                prompt += (
                    "You have tools to fetch satellite scenes for date ranges you choose and to compute "
                    "water extent from them. Decide which date ranges to compare, fetch what you need, "
                    "compute the numbers, and only call finalize_report once confident in a conclusion."
                )
            run_agent(prompt, api_key, location=args.location)
            if _insight_saved or not _skip_reason:
                break   # saved, or failed for a reason an earlier month won't fix
    finally:
        cleanup_temp_folders()
        report_usage(args.location, args.location_id, _insight_saved)
        if args.location_id is not None:
            # Record that this location was attempted, whether or not it succeeded.
            # A failure here only logs a warning — it must not hide the real result.
            try:
                storage.mark_location_run(args.location_id)
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not update last_run_at for location id %s: %s",
                            args.location_id, storage.short_error(exc))

    # A run that finishes without saving a result is a failure, even if nothing
    # crashed (e.g. hit MAX_TURNS, a tool code error, or Supabase rejected the
    # save). Exiting non-zero makes GitHub Actions mark the run red and send the
    # failure email — otherwise these runs would silently show a green tick.
    if not _insight_saved and _skip_reason:
        message = f"Skipped {args.location}: {_skip_reason}."
        print(f"::warning::{message}")   # yellow note on the run summary; the job stays green
        log.warning(message)
        return
    if not _insight_saved:
        message = f"No insight was saved for {args.location} — see the log above for the reason."
        print(f"::error::{message}")  # shows as a red annotation on the GitHub run summary
        log.error(message)
        sys.exit(1)


if __name__ == "__main__":
    main()
