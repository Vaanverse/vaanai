"""
compute_water_change.py

Compares water extent between two HLS (Harmonized Landsat Sentinel-2) scene
folders — e.g. one downloaded for 2015 and one for 2023 — using NDWI
(Normalized Difference Water Index), then asks Claude to write a plain-
English insight from the numbers.

NDWI = (Green - NIR) / (Green + NIR)
Water tends to have NDWI > 0; land/vegetation tends to be negative.
This is the standard, well-established index for surface water extent.

Usage
-----
    python compute_water_change.py --before ./data/lake_mead_2015 --after ./data/lake_mead_2023

Requires each folder to contain at least one HLSL30 granule's band files
(as downloaded by extract_eo_data.py) — specifically the Green (B03) and
NIR (B05) bands for HLSL30 (Landsat). If you used HLSS30 (Sentinel-2)
instead, the band codes differ (B03/B08) — pass --nir-band / --green-band
to override.
"""

import argparse
import glob
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import rasterio
import requests
from dotenv import load_dotenv
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PIXEL_AREA_M2 = 30 * 30  # HLS pixels are 30m x 30m


def _aoi_grid(bbox, res: int = 30):
    """A fixed 30 m grid covering the bbox, in the UTM zone of the bbox centre.
    Every scene is drawn onto this same grid, so before/after images and
    arrays always have exactly the same width and height — no matter which
    satellite tile(s) the data came from."""
    min_lon, min_lat, max_lon, max_lat = bbox
    zone = int(((min_lon + max_lon) / 2 + 180) // 6) + 1
    epsg = (32600 if (min_lat + max_lat) / 2 >= 0 else 32700) + zone
    left, bottom, right, top = transform_bounds("EPSG:4326", f"EPSG:{epsg}", *bbox)
    left, top = math.floor(left / res) * res, math.ceil(top / res) * res
    width, height = math.ceil((right - left) / res), math.ceil((top - bottom) / res)
    return f"EPSG:{epsg}", from_origin(left, top, res, res), width, height


def read_band(paths, bbox=None) -> np.ndarray:
    """Reads one band.
    - Without bbox: reads the first file as-is (whole tile).
    - With bbox: stitches EVERY tile file onto the fixed bbox grid (see
      _aoi_grid). Any part of the bbox that no tile covers is left as no-data
      (-9999 for reflectance bands, 255 for Fmask), which the Fmask check then
      counts as "bad" — so a coverage gap shows up as a warning instead of
      silently shrinking the measured lake."""
    if isinstance(paths, str):
        paths = [paths]
    if bbox is None:
        with rasterio.open(paths[0]) as src:
            return src.read(1)

    # Keep the data's own resolution: 30 m for NASA HLS, 10 m for Sentinel-2.
    crs, transform, width, height = _aoi_grid(bbox, band_resolution(paths[0]))
    out = None
    for p in paths:
        with rasterio.open(p) as src:
            nodata = src.nodata if src.nodata is not None else (255 if src.dtypes[0] == "uint8" else -9999)
            if out is None:
                out = np.full((height, width), nodata, dtype=src.dtypes[0])
            reproject(
                source=rasterio.band(src, 1), destination=out,
                src_transform=src.transform, src_crs=src.crs, src_nodata=nodata,
                dst_transform=transform, dst_crs=crs, dst_nodata=nodata,
                resampling=Resampling.nearest, init_dest_nodata=False,
            )
    return out


def band_resolution(path: str) -> int:
    """Pixel size in metres of a band file (30 for NASA HLS, 10 for Sentinel-2)."""
    with rasterio.open(path) as src:
        return int(round(abs(src.res[0])))


def find_band_files(folder: str, band_code: str) -> list[str]:
    """All files in `folder` for a band code — one per tile when a scene spans
    several tiles (common for areas near tile/UTM-zone edges, like Lake Mead)."""
    matches = sorted(glob.glob(os.path.join(folder, f"*{band_code}*.tif")))
    if not matches:
        raise ValueError(
            f"No file matches band code '{band_code}'. "
            f"Available band codes in this folder: {', '.join(list_band_codes(folder))}. "
            "HLS band codes are zero-padded (e.g. 'B03' for green, 'B05' for NIR)."
        )
    return matches

def list_band_codes(folder: str) -> list[str]:
    """Returns the sorted, deduplicated band codes present in a scene folder
    (e.g. ['B01', 'B02', ..., 'Fmask']), parsed from HLS's filename convention:
    HLS.L30.<tile>.<date>.v2.0.<BAND>.tif -- the band code is always the
    second-to-last dot-separated segment."""
    codes = set()
    for fname in os.listdir(folder):
        if fname.endswith(".tif"):
            parts = fname.split(".")
            if len(parts) >= 2:
                codes.add(parts[-2])
    return sorted(codes)


def find_band_file(folder: str, band_code: str) -> str:
    """Finds the single file in `folder` whose name contains `band_code`
    (e.g. 'B03'). Raises ValueError (not sys.exit) with a short, actionable
    message on failure -- concise on purpose, since this message may be fed
    back to an LLM agent as a tool result, and a 45-file dump wastes tokens
    and gives the model nothing useful to act on."""
    matches = glob.glob(os.path.join(folder, f"*{band_code}*.tif"))
    available = list_band_codes(folder)

    if not matches:
        raise ValueError(
            f"No file matches band code '{band_code}'. "
            f"Available band codes in this folder: {', '.join(available)}. "
            "HLS band codes are zero-padded (e.g. 'B03' for green, 'B05' for NIR)."
        )
    if len(matches) > 1:
        raise ValueError(
            f"Band code '{band_code}' matched {len(matches)} files -- this folder "
            "contains more than one tile/granule for the same date (common near "
            "UTM zone boundaries). Fetch a single granule at a time (max_results=1) "
            "to avoid this, or pick one specific file manually. "
            f"Available band codes here: {', '.join(available)}."
        )
    return matches[0]


# HLS Fmask is a bit-encoded quality band (per LP DAAC docs). Each bit flags
# a condition for that pixel; several of these commonly get misread as
# "water" by a naive NDWI threshold, so we exclude them before analysis.
_FMASK_CLOUD_BIT = 1
_FMASK_CLOUD_ADJACENT_BIT = 2
_FMASK_CLOUD_SHADOW_BIT = 3
_FMASK_SNOW_BIT = 4


def read_bad_pixel_mask(folder: str, bbox=None) -> tuple[np.ndarray, float]:
    """Reads the Fmask band and returns (bad_pixel_mask, bad_pixel_fraction).
    bad_pixel_mask is True wherever a pixel is cloud, cloud-adjacent, cloud
    shadow, or snow/ice -- all of which can look like "water" to a simple
    NDWI threshold otherwise. bad_pixel_fraction (0-1) tells the caller how
    much of the whole scene was excluded, which is a useful data-quality
    signal: a heavily clouded scene should be treated with lower confidence,
    or re-fetched for a clearer date, rather than trusted blindly."""
    fmask_files = find_band_files(folder, "Fmask") if bbox is not None else find_band_file(folder, "Fmask")
    fmask = read_band(fmask_files, bbox).astype("uint8")

    is_cloud = (fmask >> _FMASK_CLOUD_BIT) & 1
    is_cloud_adjacent = (fmask >> _FMASK_CLOUD_ADJACENT_BIT) & 1
    is_cloud_shadow = (fmask >> _FMASK_CLOUD_SHADOW_BIT) & 1
    is_snow = (fmask >> _FMASK_SNOW_BIT) & 1

    bad_mask = (is_cloud | is_cloud_adjacent | is_cloud_shadow | is_snow).astype(bool)
    bad_fraction = float(np.mean(bad_mask))
    return bad_mask, bad_fraction


def compute_ndwi(green_path, nir_path, bad_pixel_mask: np.ndarray | None = None, bbox=None) -> np.ndarray:
    """Reads the green and NIR bands (a single path, or a list of per-tile
    paths when bbox is given) and returns the NDWI array. If
    bad_pixel_mask is given (see read_bad_pixel_mask), those pixels are
    excluded (set to NaN) before the water-area calculation, so clouds/snow
    don't get miscounted as water."""
    green = read_band(green_path, bbox).astype("float32")
    nir = read_band(nir_path, bbox).astype("float32")

    if green.shape != nir.shape:
        log.error("Band shape mismatch: green=%s nir=%s", green.shape, nir.shape)
        sys.exit(1)

    denominator = green + nir
    # Avoid divide-by-zero on no-data/black-fill pixels
    ndwi = np.where(denominator == 0, np.nan, (green - nir) / denominator)

    if bad_pixel_mask is not None:
        if bad_pixel_mask.shape != ndwi.shape:
            log.error("Fmask shape %s does not match band shape %s", bad_pixel_mask.shape, ndwi.shape)
            sys.exit(1)
        ndwi = np.where(bad_pixel_mask, np.nan, ndwi)
    return ndwi


def water_area_km2(ndwi: np.ndarray, threshold: float = 0.0, pixel_area_m2: float = PIXEL_AREA_M2) -> float:
    """Counts pixels above the NDWI water threshold and converts to km2.
    pixel_area_m2 is 900 for 30 m HLS pixels, 100 for 10 m Sentinel-2 pixels."""
    water_pixels = np.nansum(ndwi > threshold)
    return float(water_pixels * pixel_area_m2 / 1_000_000)


def generate_insight(before_area: float, after_area: float, before_label: str, after_label: str) -> str:
    """Sends the computed stats to Claude and returns a plain-English insight."""
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        log.warning("ANTHROPIC_API_KEY not set — skipping AI summary, returning raw stats only.")
        return (
            f"Water extent changed from {before_area:.2f} km2 ({before_label}) "
            f"to {after_area:.2f} km2 ({after_label})."
        )

    change_pct = ((after_area - before_area) / before_area * 100) if before_area else 0.0

    prompt = (
        "You are a research analyst writing a short insight for a satellite "
        "monitoring report. Given these facts, write 2-3 plain-English sentences "
        "describing the change, its magnitude, and one plausible driver "
        "(e.g. drought, seasonal variation, water management) without overstating "
        "certainty.\n\n"
        f"Location period A ({before_label}): water extent = {before_area:.2f} sq km\n"
        f"Location period B ({after_label}): water extent = {after_area:.2f} sq km\n"
        f"Change: {change_pct:+.1f}%"
    )

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    response.raise_for_status()
    content_blocks = response.json()["content"]
    return "".join(block["text"] for block in content_blocks if block["type"] == "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True, help="Folder with the earlier-date HLS scene.")
    parser.add_argument("--after", required=True, help="Folder with the later-date HLS scene.")
    parser.add_argument("--green-band", default="B03", help="Green band code (default: B03, HLSL30/Landsat).")
    parser.add_argument("--nir-band", default="B05", help="NIR band code (default: B05, HLSL30/Landsat).")
    parser.add_argument("--threshold", type=float, default=0.0, help="NDWI threshold for classifying water.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for label, folder in [("before", args.before), ("after", args.after)]:
        if not Path(folder).is_dir():
            log.error("Folder does not exist: %s (--%s)", folder, label)
            sys.exit(1)

    try:
        log.info("Computing NDWI for 'before' scene: %s", args.before)
        before_mask, before_bad_pct = read_bad_pixel_mask(args.before)
        if before_bad_pct > 0.3:
            log.warning("'before' scene is %.0f%% cloud/snow-masked — result may be unreliable.", before_bad_pct * 100)
        before_ndwi = compute_ndwi(
            find_band_file(args.before, args.green_band),
            find_band_file(args.before, args.nir_band),
            bad_pixel_mask=before_mask,
        )
        before_area = water_area_km2(before_ndwi, args.threshold)

        log.info("Computing NDWI for 'after' scene: %s", args.after)
        after_mask, after_bad_pct = read_bad_pixel_mask(args.after)
        if after_bad_pct > 0.3:
            log.warning("'after' scene is %.0f%% cloud/snow-masked — result may be unreliable.", after_bad_pct * 100)
        after_ndwi = compute_ndwi(
            find_band_file(args.after, args.green_band),
            find_band_file(args.after, args.nir_band),
            bad_pixel_mask=after_mask,
        )
        after_area = water_area_km2(after_ndwi, args.threshold)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    change_pct = ((after_area - before_area) / before_area * 100) if before_area else 0.0

    print("\n--- Water Extent Change ---")
    print(f"Before ({args.before}): {before_area:.2f} sq km")
    print(f"After  ({args.after}):  {after_area:.2f} sq km")
    print(f"Change: {change_pct:+.1f}%\n")

    insight = generate_insight(before_area, after_area, args.before, args.after)
    print("--- AI Insight ---")
    print(insight)


if __name__ == "__main__":
    main()
