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
import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

import extract_eo_data as eo
import compute_water_change as wc
import storage_utils as storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"  # confirmed available on this account; use --model to try openai/gpt-oss-120b for higher quality
AOI_BBOX = None
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
                "Search NASA's HLS (Landsat) archive for the clearest satellite scene over a "
                "bounding box within a date range, download every tile needed to cover the box "
                "for that day, and return a local folder path plus scene_date (the actual day "
                "the picture was taken). Use this whenever you need imagery for a new date range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "short_name": {"type": "string", "description": "HLSL30 (Landsat) or HLSS30 (Sentinel-2)."},
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


def tool_fetch_scene(args: dict) -> dict:
    bbox = (args["min_lon"], args["min_lat"], args["max_lon"], args["max_lat"])
    short_name = "HLSL30"   # compute_water_extent assumes Landsat band codes (NIR = B05)
    try:
        granules = eo.search_granules(
            short_name=short_name, bbox=bbox,
            start_date=args["start_date"], end_date=args["end_date"],
            max_results=50, cloud_cover=20,
        )
        if not granules:
            return {"error": "No clear (<20% cloud) scenes found for that range. Try a different or wider date range."}

        # The bbox can span several tiles. Group tiles by the day they were
        # captured, then pick the day whose tiles together cover the bbox best,
        # and among those, the clearest. ALL tiles of that day are downloaded
        # and later stitched together onto one fixed grid.
        by_day: dict = {}
        for g in granules:
            by_day.setdefault(_acq_day(g), []).append(g)

        def day_cover(gs) -> float:
            return min(1.0, sum(_granule_overlap(g, bbox) for g in gs))

        def day_score(gs):
            return (day_cover(gs) >= 0.95, -max(_cloud_cover(g) for g in gs), day_cover(gs))

        best_day = max(by_day, key=lambda d: day_score(by_day[d]))
        chosen = by_day[best_day]

        folder = tempfile.mkdtemp(prefix="vaanai_scene_")
        _created_folders.append(folder)
        eo.download_granules(chosen, folder)

        return {
            "folder": folder,
            "scene_date": datetime.strptime(best_day, "%Y%j").date().isoformat(),
            "tile_count": len(chosen),
            "aoi_coverage_pct": round(day_cover(chosen) * 100),
            "tile_cloud_pct": max(_cloud_cover(g) for g in chosen),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc) or repr(exc) or "Unknown error during fetch (see logs above)."}


def tool_compute_water_extent(args: dict) -> dict:
    folder = args["folder"]
    green_band = args.get("green_band", "B03")
    nir_band = args.get("nir_band", "B05")
    try:
        bad_mask, bad_pct = wc.read_bad_pixel_mask(folder, bbox=AOI_BBOX)
        ndwi = wc.compute_ndwi(
            wc.find_band_files(folder, green_band),
            wc.find_band_files(folder, nir_band),
            bad_pixel_mask=bad_mask, bbox=AOI_BBOX,
        )
        result = {
            "water_area_km2": round(wc.water_area_km2(ndwi), 2),
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


def call_groq(messages: list, api_key: str, max_retries: int = 3) -> dict:
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
                "tool_choice": "auto",
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
    """Generates before/after preview images, uploads them to Supabase Storage,
    and inserts one row into the `insights` table — called once finalize_report
    is invoked. Keeps the model's own reported values as the source of truth
    for what to save, since it's the one that knows which folders/dates it used."""
    import tempfile as _tempfile

    before_dir = _tempfile.mkdtemp(prefix="vaanai_preview_")
    after_dir = _tempfile.mkdtemp(prefix="vaanai_preview_")
    _created_folders.extend([before_dir, after_dir])
    before_png = os.path.join(before_dir, "before.png")
    after_png = os.path.join(after_dir, "after.png")

    storage.generate_preview_png(final_args["before_folder"], before_png, bbox=AOI_BBOX)
    storage.generate_preview_png(final_args["after_folder"], after_png, bbox=AOI_BBOX)

    before_url = storage.upload_image(before_png)
    after_url = storage.upload_image(after_png)

    before_value = final_args["before_value"]
    after_value = final_args["after_value"]
    change_pct = ((after_value - before_value) / before_value * 100) if before_value else 0.0

    storage.save_insight(
        {
            "location": location,
            "before_date": final_args["before_date"],
            "after_date": final_args["after_date"],
            "before_value": before_value,
            "after_value": after_value,
            "change_pct": round(change_pct, 2),
            "summary": final_args["summary"],
            "confidence": final_args["confidence"],
            "before_image_url": before_url,
            "after_image_url": after_url,
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
        print("(Saved image pair + result to Supabase.)")
    except Exception as exc:  # noqa: BLE001 - don't let a save failure hide the report itself
        log.error("Failed to persist insight to Supabase: %s", exc)
        print(f"(Warning: could not save to Supabase — {exc})")

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

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": {"type": "function", "function": {"name": "finalize_report"}},
            "reasoning_format": "hidden",
            "reasoning_effort": "low",
        },
        timeout=60,
    )
    if not response.ok:
        log.error("Forced finalize call failed: %s", response.text)
        return False

    message = response.json()["choices"][0]["message"]
    for call in message.get("tool_calls") or []:
        if call["function"]["name"] == "finalize_report":
            tool_args = json.loads(call["function"]["arguments"])
            _handle_finalize(tool_args, location)
            return True
    return False


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
            recovered = _extract_report_from_text(message.get("content"), computed)
            if recovered:
                log.warning("Model wrote its report as text — using its own chosen scenes.")
                _handle_finalize(recovered, location)
                return
            if _force_finalize(messages, api_key, computed, folder_dates, location, rejected):
                return
            log.info("No tool call made, and not enough tracked evidence to force a finalize; ending.")
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
    return parser.parse_args()


def main() -> None:
    global MODEL, AOI_BBOX
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        log.error("GROQ_API_KEY not set in .env. Get a free key at https://console.groq.com")
        sys.exit(1)

    args = parse_args()
    MODEL = args.model
    AOI_BBOX = tuple(args.bbox)

    eo.authenticate()  # NASA credentials still needed for the data-fetching tool

    min_lon, min_lat, max_lon, max_lat = args.bbox
    goal_prompt = (
        f"Location: {args.location}\n"
        f"Bounding box: min_lon={min_lon}, min_lat={min_lat}, max_lon={max_lon}, max_lat={max_lat}\n\n"
        f"Task: {args.goal}\n\n"
        "You have tools to fetch satellite scenes for date ranges you choose and to compute "
        "water extent from them. Decide which date ranges to compare, fetch what you need, "
        "compute the numbers, and only call finalize_report once confident in a conclusion."
    )

    try:
        run_agent(goal_prompt, api_key, location=args.location)
    finally:
        cleanup_temp_folders()

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
