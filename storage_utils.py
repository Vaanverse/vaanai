"""
storage_utils.py

Two responsibilities that turn a raw analysis result into something a
website can actually display:

1. generate_preview_png(): converts a folder of raw HLS band files (which
   are not viewable images on their own — they're scientific data, 16-bit
   per pixel, not meant for a browser) into an ordinary true-color PNG,
   the kind of "satellite photo" a general audience recognizes.

2. upload_image() / save_insight(): push that PNG and the computed result
   to Supabase (Storage for the image, a table for the data), so both
   survive after the local/temporary folder is deleted — which matters
   especially on GitHub Actions, where the entire machine is destroyed at
   the end of every run.

The satellite libraries (numpy, rasterio via compute_water_change) are only
imported inside generate_preview_png(), so other agents that just need the
Supabase helpers (e.g. weekly_news_agent.py) don't have to load them.
"""

from __future__ import annotations   # lets type hints mention np.ndarray without importing numpy

import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone

from PIL import Image
from supabase import create_client

log = logging.getLogger(__name__)

BUCKET_NAME = "insight-images"

# Small areas (e.g. a 4 km box around a small dam) give tiny images at 30 m per
# pixel (~150 px). Previews whose longest side is below this are enlarged by a
# whole-number factor so they're comfortable to view. Large areas are unchanged.
MIN_PREVIEW_PX = 1000


def _stretch_to_uint8(band: np.ndarray, max_reflectance: float = 3000) -> np.ndarray:
    """Rescales a raw HLS reflectance band to 0-255 for display, using a FIXED
    scale (not adaptive to each image's own min/max). HLS stores reflectance
    as an integer scaled by 10000 (so 3000 = 0.3 reflectance, a reasonable
    ceiling for bright land surfaces). Using the same fixed scale for every
    image is important for before/after comparisons specifically — an
    adaptive per-image percentile stretch makes each image's color balance
    depend on that scene's own content, so two genuinely different scenes
    end up looking inconsistently colored even with no real change between
    them, which would mislead anyone comparing the two pictures."""
    import numpy as np

    valid_mask = band > -9000  # HLS uses -9999 as a "no data" fill value
    stretched = np.clip(band, 0, max_reflectance) / max_reflectance
    result = (stretched * 255).astype(np.uint8)
    result[~valid_mask] = 0  # render fill/no-data pixels as black
    return result

def generate_preview_png(folder: str, output_path: str, bbox=None) -> str:
    """Builds a true-color (Red/Green/Blue) PNG from an HLSL30 scene folder
    and saves it to output_path. Returns output_path for convenience.

    HLSL30 (Landsat) true color = B04 (Red), B03 (Green), B02 (Blue).
    If you're using HLSS30 (Sentinel-2) instead, the band numbers for
    true color are the same (B04/B03/B02), so this works for both.
    """
    import glob

    import numpy as np
    from compute_water_change import read_band as read_clipped

    def read_band(code: str) -> np.ndarray:
        matches = sorted(glob.glob(os.path.join(folder, f"*{code}*.tif")))
        if not matches:
            raise ValueError(f"Could not find band {code} in {folder} for preview image.")
        # With a bbox, all tiles are stitched onto the same fixed grid, so the
        # before and after images always come out exactly the same size.
        return read_clipped(matches if bbox is not None else matches[0], bbox).astype("float32")

    red = _stretch_to_uint8(read_band("B04"))
    green = _stretch_to_uint8(read_band("B03"))
    blue = _stretch_to_uint8(read_band("B02"))

    rgb = np.dstack([red, green, blue])
    img = Image.fromarray(rgb, mode="RGB")

    # Enlarge small previews. NEAREST + a whole-number factor turns each 30 m
    # satellite pixel into a crisp square block, instead of blurring it — the
    # picture gets bigger without pretending to have more detail than the data.
    # Before and after share the same grid size, so they get the same factor.
    longest = max(img.size)
    if longest < MIN_PREVIEW_PX:
        factor = -(-MIN_PREVIEW_PX // longest)  # ceiling division
        img = img.resize((img.width * factor, img.height * factor), Image.NEAREST)

    img.save(output_path)
    log.info("Saved preview image: %s (%dx%d px)", output_path, img.width, img.height)
    return output_path


def get_supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set in .env")
    return create_client(url, key)


RETRY_WAITS = (5, 15, 30)   # seconds to wait before attempts 2, 3 and 4


def short_error(exc: Exception, limit: int = 300) -> str:
    """One-line version of an error. Supabase/Cloudflare outages come back as a
    whole HTML error page, which otherwise floods the log with hundreds of lines."""
    text = " ".join(str(exc).split())
    if "<!DOCTYPE" in text or "<html" in text:
        title = text.split("<title>")[1].split("</title>")[0] if "<title>" in text else "HTML error page"
        status = text.split("'statusCode':")[1].split(",")[0].strip() if "'statusCode':" in text else "?"
        return f"HTTP {status} from Supabase: {title}"
    return text[:limit] + ("…" if len(text) > limit else "")


def with_retries(action: str, func):
    """Runs func(), retrying on failure. Supabase occasionally returns a
    temporary 5xx (e.g. Cloudflare 520) or drops the connection; a short wait and
    retry almost always succeeds, so one hiccup shouldn't throw away a finished
    multi-minute analysis. Re-raises the last error if every attempt fails."""
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001
            if attempt == len(RETRY_WAITS):
                raise RuntimeError(f"{action} failed after {attempt + 1} attempts: {short_error(exc)}") from None
            wait = RETRY_WAITS[attempt]
            log.warning("%s failed (attempt %d/%d): %s — retrying in %ds.",
                        action, attempt + 1, len(RETRY_WAITS) + 1, short_error(exc), wait)
            time.sleep(wait)


# Old name, kept so any script still calling storage._with_retries keeps working.
_with_retries = with_retries


def upload_image(local_path: str) -> str:
    """Uploads a local PNG to the Supabase Storage bucket and returns its
    public URL. Assumes the bucket is set to Public (see setup instructions)."""
    client = get_supabase_client()
    remote_name = f"{uuid.uuid4().hex}.png"

    def _upload():
        with open(local_path, "rb") as f:
            # upsert: if an attempt actually reached Supabase before the error
            # came back, the retry overwrites that file instead of failing
            # with "already exists".
            client.storage.from_(BUCKET_NAME).upload(
                remote_name, f, {"content-type": "image/png", "upsert": "true"}
            )

    with_retries(f"Uploading {os.path.basename(local_path)}", _upload)

    public_url = client.storage.from_(BUCKET_NAME).get_public_url(remote_name)
    log.info("Uploaded %s -> %s", local_path, public_url)
    return public_url


def save_insight(record: dict) -> None:
    """Saves one result to the `water_insights` table — ONE ROW PER LOCATION PER DAY.

    - First run of the day for a location  -> INSERT a new row (created_at is
      filled in by the database default; modified_at is set here).
    - Another run on the SAME created_at date (UTC) -> UPDATE that day's row
      in place and bump modified_at. created_at is left untouched.

    So every weekly run adds a new row, even when the agent picks the same
    two scene dates as last week (no new clear scene yet). The day is the UTC
    day, matching how Supabase stores created_at (the schedule runs 06:00 UTC).

    Requires the `modified_at` column, and the old UNIQUE constraint on
    (location, before_date, after_date) to be dropped — see the SQL in
    sql/2026-10-06_modified_at.sql. Otherwise the insert for a repeated scene
    pair would be rejected as a duplicate."""
    client = get_supabase_client()
    table = client.table("water_insights")

    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    row = {**record, "modified_at": now.isoformat()}
    row.pop("created_at", None)   # never overwrite the original creation time

    def _write():
        # The look-up runs again on every retry, so if an earlier attempt's
        # insert did reach the database, the retry updates that row instead of
        # adding a duplicate.
        todays_rows = (
            table.select("*")
            .eq("location", record["location"])
            .gte("created_at", day_start.isoformat())
            .lt("created_at", day_end.isoformat())
            .order("created_at", desc=True)
            .execute()
            .data
        )
        if todays_rows:
            existing = todays_rows[0]
            query = table.update(row)
            if "id" in existing:
                query = query.eq("id", existing["id"])
            else:   # no id column — fall back to matching the exact created_at
                query = query.eq("location", record["location"]).eq("created_at", existing["created_at"])
            return query.execute(), "Updated today's existing"
        return table.insert(row).execute(), "Inserted NEW"

    response, action = with_retries("Saving insight row", _write)

    if not response.data:
        # Supabase can "succeed" while writing nothing (e.g. a row-level-security
        # policy blocks it). Fail loudly so the run goes red instead of green.
        raise RuntimeError("Supabase returned no rows — nothing was written "
                           "(check the SUPABASE_KEY is the service_role key, or the table's RLS policies).")

    saved = response.data[0]
    log.info("%s insight row (id=%s, created_at=%s) for %s: %s -> %s.",
             action, saved.get("id", "n/a"), saved.get("created_at"), record["location"],
             record["before_date"], record["after_date"])
