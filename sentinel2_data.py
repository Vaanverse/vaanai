"""
sentinel2_data.py

Sentinel-2 Level-2A (10 m) data source, used for SMALL water bodies (e.g.
village reservoirs and tanks) where NASA HLS's 30 m pixels are too coarse.
Large reservoirs keep using NASA HLS (see extract_eo_data.py) — the agent
picks the source automatically from the size of the bounding box.

Where the data comes from
-------------------------
Element 84's free "Earth Search" catalogue (a STAC API) indexes the public
Sentinel-2 archive on AWS. No account, login or API key is needed, and
Copernicus Sentinel data is free for commercial use — the only condition is
attribution ("Contains modified Copernicus Sentinel data <year>"), which the
website footer includes.

How it fits into the existing pipeline
--------------------------------------
Instead of downloading whole 100 km tiles, this reads ONLY the pixels inside
the bounding box straight from the cloud-optimised files (fast, a few MB),
and writes them out as small GeoTIFFs that use the SAME band codes the rest
of the code already expects:

    B02 = blue, B03 = green, B04 = red,
    B05 = near-infrared  (Sentinel-2's B08, renamed so NDWI code is unchanged)
    Fmask = cloud/shadow/snow mask in HLS Fmask bit layout (converted from
            Sentinel-2's Scene Classification Layer, "SCL")

Reflectance is stored like HLS (reflectance x 10000, -9999 = no data), so
compute_water_change.py and storage_utils.py work on these folders as-is.
"""

from __future__ import annotations

import logging
import math
import os
from datetime import datetime

import numpy as np
import rasterio
import requests
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from compute_water_change import _aoi_grid

log = logging.getLogger(__name__)

STAC_URL = "https://earth-search.aws.element84.com/v1/search"
# Newer reprocessed collection first; fall back to the original one.
COLLECTIONS = ["sentinel-2-c1-l2a", "sentinel-2-l2a"]

RES = 10          # metres per pixel for the output grid
NODATA = -9999    # same no-data value HLS uses for reflectance bands

# Our band code -> possible asset names in the catalogue (names differ slightly
# between collections, so we try each in turn).
BANDS = {
    "B02": ["blue", "B02"],
    "B03": ["green", "B03"],
    "B04": ["red", "B04"],
    "B05": ["nir", "B08"],   # NIR: Sentinel-2 B08, stored under HLS-L30's NIR code
}
SCL_KEYS = ["scl", "SCL"]

# Sentinel-2 Scene Classification (SCL) classes -> HLS Fmask bits
_SCL_NODATA = 0
_SCL_CLOUD = (1, 8, 9, 10)   # saturated/defective, cloud medium, cloud high, thin cirrus
_SCL_SHADOW = 3
_SCL_SNOW = 11
_SCL_WATER = 6
_FMASK_CLOUD, _FMASK_SHADOW, _FMASK_SNOW, _FMASK_WATER = 1 << 1, 1 << 3, 1 << 4, 1 << 5
_FMASK_BAD = _FMASK_CLOUD | _FMASK_SHADOW | _FMASK_SNOW

# Read public AWS files without credentials, and keep GDAL from listing
# whole bucket folders (slow) when opening a single file.
GDAL_ENV = {
    "AWS_NO_SIGN_REQUEST": "YES",
    "AWS_REGION": "us-west-2",
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "2",
}


# ---------------------------------------------------------------------------
# Searching the catalogue
# ---------------------------------------------------------------------------

def search_items(bbox, start_date: str, end_date: str, max_cloud: float = 60, limit: int = 100) -> list[dict]:
    """Returns Sentinel-2 L2A scenes (STAC items) over bbox in the date range.
    max_cloud is the WHOLE-TILE cloud %, kept loose on purpose: a tile can be
    60% cloudy and still be perfectly clear over a small reservoir. The real
    cloud check happens over the bbox only, in best_scene()."""
    body = {
        "bbox": list(bbox),
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
    }
    for collection in COLLECTIONS:
        try:
            r = requests.post(STAC_URL, json={**body, "collections": [collection]}, timeout=60)
        except requests.RequestException as exc:
            log.warning("Sentinel-2 search failed (%s): %s", collection, exc)
            continue
        if r.status_code != 200:
            log.warning("Sentinel-2 search returned HTTP %s for %s", r.status_code, collection)
            continue
        items = r.json().get("features", [])
        log.info("Sentinel-2 search (%s): %d scene(s).", collection, len(items))
        if items:
            return items
    return []


def item_day(item: dict) -> str:
    return item["properties"]["datetime"][:10]


def item_cloud(item: dict) -> float:
    return float(item["properties"].get("eo:cloud_cover", 100))


def item_overlap(item: dict, bbox) -> float:
    """Fraction (0-1) of bbox covered by the scene's bounding box."""
    ib = item.get("bbox")
    if not ib:
        return 0.0
    min_lon, min_lat, max_lon, max_lat = bbox
    w = max(0.0, min(max_lon, ib[2]) - max(min_lon, ib[0]))
    h = max(0.0, min(max_lat, ib[3]) - max(min_lat, ib[1]))
    return (w * h) / ((max_lon - min_lon) * (max_lat - min_lat))


# ---------------------------------------------------------------------------
# Reading only the bbox pixels from the cloud files
# ---------------------------------------------------------------------------

def _asset(item: dict, keys: list[str]) -> dict:
    for k in keys:
        if k in item.get("assets", {}):
            return item["assets"][k]
    raise ValueError(f"Scene {item.get('id')} has none of the assets {keys}.")


def _href(asset: dict) -> str:
    """https:// links are used as-is; s3:// links are read anonymously via GDAL."""
    href = asset["href"]
    if href.startswith("s3://"):
        alt = asset.get("alternate", {}).get("https", {}).get("href")
        return alt or "/vsis3/" + href[len("s3://"):]
    return href


def _dn_offset(item: dict, asset: dict) -> int:
    """Since 2022 (processing baseline 04.00) Sentinel-2 L2A values carry a
    +1000 offset that must be removed to get true reflectance x 10000. The
    catalogue states it either in the band metadata or in a flag."""
    rb = asset.get("raster:bands") or []
    if rb and rb[0].get("offset") is not None and rb[0].get("scale"):
        return int(round(rb[0]["offset"] / rb[0]["scale"]))     # e.g. -0.1 / 0.0001 = -1000
    props = item.get("properties", {})
    if props.get("earthsearch:boa_offset_applied") is False:
        try:
            if float(props.get("s2:processing_baseline", "0")) >= 4.0:
                return -1000
        except ValueError:
            pass
    return 0


def _warp_window_into(href, out, dst_crs, dst_transform, convert, nodata) -> bool:
    """Reads just the part of `href` that overlaps the output grid, converts it
    with `convert`, and reprojects it into `out` (only where there is data, so
    several scenes can be stitched together). Returns False if no overlap."""
    with rasterio.open(href) as src:
        h, w = out.shape
        left, top = dst_transform.c, dst_transform.f
        right, bottom = left + w * dst_transform.a, top + h * dst_transform.e
        sb = transform_bounds(dst_crs, src.crs, left, bottom, right, top)
        win = from_bounds(*sb, transform=src.transform)
        c0 = max(0, math.floor(win.col_off) - 2)
        r0 = max(0, math.floor(win.row_off) - 2)
        c1 = min(src.width, math.ceil(win.col_off + win.width) + 2)
        r1 = min(src.height, math.ceil(win.row_off + win.height) + 2)
        if c1 <= c0 or r1 <= r0:
            return False
        win = Window(c0, r0, c1 - c0, r1 - r0)
        data = convert(src.read(1, window=win))
        reproject(
            source=data, destination=out,
            src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=nodata,
            dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=nodata,
            resampling=Resampling.nearest, init_dest_nodata=False,
        )
        return True


def _reflectance_converter(offset: int):
    def convert(arr: np.ndarray) -> np.ndarray:
        a = arr.astype("int32")
        a = a + offset
        a[arr == 0] = NODATA                     # Sentinel-2 uses 0 for no data
        return np.clip(a, NODATA, 32767).astype("int16")
    return convert


def _scl_to_fmask(scl: np.ndarray) -> np.ndarray:
    fm = np.zeros(scl.shape, dtype="uint8")
    fm[np.isin(scl, _SCL_CLOUD)] |= _FMASK_CLOUD
    fm[scl == _SCL_SHADOW] |= _FMASK_SHADOW
    fm[scl == _SCL_SNOW] |= _FMASK_SNOW
    fm[scl == _SCL_WATER] |= _FMASK_WATER
    fm[scl == _SCL_NODATA] = 255                 # same fill value as HLS Fmask
    return fm


def _fmask_for(items: list[dict], bbox, res: int):
    crs, transform, w, h = _aoi_grid(bbox, res)
    fm = np.full((h, w), 255, dtype="uint8")
    for it in items:
        _warp_window_into(_href(_asset(it, SCL_KEYS)), fm, crs, transform, _scl_to_fmask, 255)
    return fm, crs, transform


def aoi_quality(items: list[dict], bbox) -> tuple[float, float]:
    """(bad_fraction, coverage_fraction) over the bbox only, from the scene
    classification layer at 20 m — a cheap check before downloading bands."""
    with rasterio.Env(**GDAL_ENV):
        fm, _, _ = _fmask_for(items, bbox, 20)
    covered = fm != 255
    bad = (~covered) | ((fm & _FMASK_BAD) != 0)
    return float(bad.mean()), float(covered.mean())


CLEAR_ENOUGH = 0.10   # with a target date: first day this clear (<10% cloudy over the bbox) wins


def best_scene(bbox, start_date: str, end_date: str, max_candidates: int = 6, target_date: str | None = None):
    """Finds the best day over the bbox. Groups scenes by acquisition day
    (a bbox can straddle two tiles), keeps the days that cover the bbox, then
    checks up to `max_candidates` days over the bbox itself and returns
    (day, items, bad_fraction, coverage_fraction) — or None.

    - Without target_date: checks the least cloudy days first; the clearest wins.
    - With target_date (YYYY-MM-DD): checks the days CLOSEST to that date first;
      the first one that is clear enough (CLEAR_ENOUGH) wins. If none is, the
      clearest one checked is returned."""
    items = search_items(bbox, start_date, end_date)
    by_day: dict = {}
    for it in items:
        by_day.setdefault(item_day(it), []).append(it)

    def cover(its):
        return min(1.0, sum(item_overlap(i, bbox) for i in its))

    days = [d for d in by_day if cover(by_day[d]) >= 0.95] or list(by_day)
    if target_date:
        target = datetime.strptime(target_date, "%Y-%m-%d").date()
        days.sort(key=lambda d: (abs((datetime.strptime(d, "%Y-%m-%d").date() - target).days),
                                 max(item_cloud(i) for i in by_day[d])))
        good_enough = CLEAR_ENOUGH
    else:
        days.sort(key=lambda d: max(item_cloud(i) for i in by_day[d]))
        good_enough = 0.02   # essentially clear — no need to check more days

    best = None
    for day in days[:max_candidates]:
        try:
            bad, covered = aoi_quality(by_day[day], bbox)
        except Exception as exc:  # noqa: BLE001 - one unreadable scene shouldn't stop the search
            log.warning("Skipping %s: could not read its cloud mask (%s)", day, exc)
            continue
        log.info("Sentinel-2 %s: %.0f%% cloudy/missing over the area.", day, bad * 100)
        if best is None or bad < best[2]:
            best = (day, by_day[day], bad, covered)
        if bad < good_enough:
            break
    return best


def download_scene(items: list[dict], bbox, folder: str) -> str:
    """Writes the bbox-only bands + Fmask for one day into `folder` on the
    fixed 10 m grid, using HLS-style file names and band codes."""
    os.makedirs(folder, exist_ok=True)
    crs, transform, w, h = _aoi_grid(bbox, RES)
    day = item_day(items[0]).replace("-", "")
    profile = dict(driver="GTiff", width=w, height=h, count=1, crs=crs, transform=transform, compress="deflate")

    def write(code, arr, nodata):
        path = os.path.join(folder, f"S2.L2A.AOI.{day}.v1.{code}.tif")
        with rasterio.open(path, "w", dtype=arr.dtype, nodata=nodata, **profile) as dst:
            dst.write(arr, 1)

    with rasterio.Env(**GDAL_ENV):
        for code, keys in BANDS.items():
            out = np.full((h, w), NODATA, dtype="int16")
            for it in items:
                asset = _asset(it, keys)
                _warp_window_into(_href(asset), out, crs, transform,
                                  _reflectance_converter(_dn_offset(it, asset)), NODATA)
            write(code, out, NODATA)
        fm, _, _ = _fmask_for(items, bbox, RES)
        write("Fmask", fm, 255)

    log.info("Saved Sentinel-2 %s (%dx%d px at %d m) to %s", item_day(items[0]), w, h, RES, folder)
    return folder
