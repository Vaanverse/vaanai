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
"""

import logging
import os
import uuid

import numpy as np
import rasterio
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


def upload_image(local_path: str) -> str:
    """Uploads a local PNG to the Supabase Storage bucket and returns its
    public URL. Assumes the bucket is set to Public (see setup instructions)."""
    client = get_supabase_client()
    remote_name = f"{uuid.uuid4().hex}.png"

    with open(local_path, "rb") as f:
        client.storage.from_(BUCKET_NAME).upload(
            remote_name, f, {"content-type": "image/png"}
        )

    public_url = client.storage.from_(BUCKET_NAME).get_public_url(remote_name)
    log.info("Uploaded %s -> %s", local_path, public_url)
    return public_url


def save_insight(record: dict) -> None:
    """Inserts one row into the `water_insights` table. `record` keys should match
    the table's columns (location, before_date, after_date, before_value,
    after_value, change_pct, summary, confidence, before_image_url,
    after_image_url) — see the SQL in the build plan for the exact schema."""
    client = get_supabase_client()
    # upsert (not insert) so re-running the same location/date-pair comparison
    # updates the existing row instead of creating a duplicate. Requires the
    # unique constraint on (location, before_date, after_date) — see build notes.
    client.table("water_insights").upsert(record, on_conflict="location,before_date,after_date").execute()
    log.info("Saved insight to database: %s", record.get("location"))
