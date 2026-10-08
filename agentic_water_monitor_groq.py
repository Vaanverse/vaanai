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

import requests
from dotenv import load_dotenv

import extract_eo_data as eo
import compute_water_change as wc
import sentinel2_data as s2
import storage_utils as storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"  # confirmed available on this account; use --model to try openai/gpt-oss-120b for higher quality
AOI_BBOX = None
MEDIA_NEEDED = True  # set in main() from --media-needed; False = save the numbers only, no before/after images
LOCATION_ID = None   # set in main() from --location-id; saved as water_insights.location_id

# Month-vs-same-month-last-year comparison (set in main()). A run on, say,
# 1 October compares the PREVIOUS calendar month with the same month a year earlier:
#   "after"  = September this year  -> the LATEST clear scene in 1-30 Sep
#   "before" = September last year  -> the LATEST clear scene in 1-30 Sep last year
# "Latest" = closest to the month's last day. Satellites don't image a place every
# day (Sentinel-2 ~5 days, HLS ~2-3 days) and clouds hide some days, so the scene
# picked is the last CLEAR one in the month. fetch_scene only accepts date ranges
# inside these two months.
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
                        "description": "2-4 plain-English sentences: finding, magnitude, likely driver.",
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
    if WINDOWS:
        return (f"No usable {source} scene exists anywhere in {args['start_date']}..{args['end_date']}. "
                "The comparison months are fixed and cannot be widened, so do NOT retry this period. "
                "If you have a scene for the other period, stop and explain that no comparison is possible.")
    return f"No usable {source} scenes found for that range. Try a different or wider date range."


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
            return {"error": _no_scene_message("Sentinel-2", args)}
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


def tool_fetch_scene(args: dict) -> dict:
    snapped = _snap_to_window(args)
    if isinstance(snapped, dict):
        return snapped   # error: range outside both comparison windows
    start_date, end_date, target = snapped
    args = {**args, "start_date": start_date, "end_date": end_date}

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
            return {"error": _no_scene_message("HLS", args)}

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
        result = {
            "water_area_km2": round(wc.water_area_km2(ndwi, pixel_area_m2=res * res), 3 if res < 30 else 2),
            "cloud_snow_masked_pct": round(bad_pct * 100, 1),
        }
        if bad_pct > 0.10:
            result["warning"] = (
                f"{bad_pct*100:.0f}% of this scene is cloud/snow-covered and was excluded from the "
                "calculation. The remaining result may be unreliable — consider fetching a different, "
                "clearer date instead of trusting this value if the percentage is high."
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

        storage.generate_preview_png(final_args["before_folder"], before_png, bbox=AOI_BBOX)
        storage.generate_preview_png(final_args["after_folder"], after_png, bbox=AOI_BBOX)

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

    storage.save_insight(
        {
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
    )


def _handle_finalize(tool_args: dict, location: str) -> None:
    global _insight_saved
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
    candidates = {f: d for f, d in folder_dates.items() if f in computed and f not in rejected}
    if len(candidates) < 2:
        # Not enough clear scenes — fall back to every computed scene.
        candidates = {f: d for f, d in folder_dates.items() if f in computed}
        if len(candidates) < 2:
            return False   # still not enough to compare, even with cloudy ones

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
    _handle_finalize(
        {
            "summary": (
                f"Water surface area changed from {b:.2f} km² on {before_date} "
                f"to {a:.2f} km² on {after_date} ({pct:+.1f}%)."
            ),
            "confidence": "medium",
            "before_folder": before_folder, "after_folder": after_folder,
            "before_date": before_date, "after_date": after_date,
            "before_value": b, "after_value": a,
        },
        location,
    )
    return True


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
        help="Pretend the run happens on this date (default: today, UTC). The PREVIOUS month of this "
             "date is compared with the same month a year earlier, e.g. 2026-10-01 -> Sep 2026 vs Sep 2025. "
             "Use 'none' to let the model choose its own dates (old behaviour).",
    )
    parser.add_argument(
        "--location-id", type=int, default=None,
        help="id of the row in monitored_locations_water; when given, its last_run_at is updated after the run.",
    )
    return parser.parse_args()


def bbox_area_km2(bbox) -> float:
    min_lon, min_lat, max_lon, max_lat = bbox
    mid_lat = math.radians((min_lat + max_lat) / 2)
    return (max_lon - min_lon) * 111.32 * math.cos(mid_lat) * (max_lat - min_lat) * 110.57


def main() -> None:
    global MODEL, AOI_BBOX, DATA_SOURCE, DATA_RES, MEDIA_NEEDED, LOCATION_ID
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

    area = bbox_area_km2(AOI_BBOX)
    DATA_SOURCE = args.source if args.source != "auto" else ("s2" if area < SMALL_AOI_KM2 else "hls")
    DATA_RES = 10 if DATA_SOURCE == "s2" else 30
    log.info("Area %.1f km² -> using %s.", area, SOURCE_LABELS[DATA_SOURCE])

    if DATA_SOURCE == "hls":
        eo.authenticate()  # NASA login is only needed for HLS; Sentinel-2 is open, no login

    min_lon, min_lat, max_lon, max_lat = args.bbox
    goal_prompt = (
        f"Location: {args.location}\n"
        f"Bounding box: min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}\n\n"
        f"Task: {args.goal}\n\n"
    )

    if (args.run_date or "").lower() != "none":
        run_date = (datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date
                    else datetime.now(timezone.utc).date())
        set_comparison_windows(run_date)
        a, b = WINDOWS["after"], WINDOWS["before"]
        month = a[0].strftime("%B")
        log.info("Run date %s -> comparing %s %d (%s..%s) with %s %d (%s..%s), latest clear scene of each.",
                 run_date, month, a[0].year, a[0], a[1], month, b[0].year, b[0], b[1])
        goal_prompt += (
            f"COMPARISON PERIOD IS FIXED — compare {month} {a[0].year} with {month} {b[0].year}, "
            "using the LATEST clear scene of each month:\n"
            f"- AFTER ({month} {a[0].year}): call fetch_scene with start_date={a[0]}, end_date={a[1]}.\n"
            f"- BEFORE ({month} {b[0].year}): call fetch_scene with start_date={b[0]}, end_date={b[1]}.\n"
            "fetch_scene automatically picks the clear day closest to the end of the month and reports "
            "days_from_target. If a scene turns out too cloudy, fetch again with a narrower range "
            "INSIDE the same month that ends before that scene_date. Ranges outside these two months "
            "are rejected. Mention both scene dates and the month compared in your summary.\n\n"
            "Fetch both scenes, compute water extent for each, and only call finalize_report once "
            "confident in a conclusion."
        )
    else:
        goal_prompt += (
            "You have tools to fetch satellite scenes for date ranges you choose and to compute "
            "water extent from them. Decide which date ranges to compare, fetch what you need, "
            "compute the numbers, and only call finalize_report once confident in a conclusion."
        )

    try:
        run_agent(goal_prompt, api_key, location=args.location)
    finally:
        cleanup_temp_folders()
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
    if not _insight_saved:
        message = f"No insight was saved for {args.location} — see the log above for the reason."
        print(f"::error::{message}")  # shows as a red annotation on the GitHub run summary
        log.error(message)
        sys.exit(1)


if __name__ == "__main__":
    main()
