"""
outline_utils.py

Helpers for a water body's OUTLINE (its shoreline polygon, stored as GeoJSON in
monitored_locations_water.outline by fix_outlines.py).

The outline is used for two things:
  1. Measuring: only water INSIDE the outline (plus a small margin, so a lake
     that has grown past its mapped shoreline is still counted fully) is
     counted, so nearby ponds, rivers and flooded paddy fields in the picture
     no longer inflate the number. Cloud cover is also judged over the water
     body itself rather than the whole picture.
  2. Showing: the outline is drawn as a border on the before/after pictures.

Everything works on the same fixed grid the rest of the pipeline uses
(compute_water_change._aoi_grid), so a mask from here lines up pixel for pixel
with the NDWI and cloud arrays.
"""

from __future__ import annotations

import json

import numpy as np
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from compute_water_change import _aoi_grid

OUTLINE_BUFFER_M = 100   # count water up to 100 m outside the mapped shoreline


def parse_outline(value) -> dict | None:
    """The outline column as a GeoJSON geometry dict (Polygon / MultiPolygon),
    or None. Accepts a dict, a JSON string, or a Feature / FeatureCollection."""
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, dict):
        return None
    if value.get("type") == "FeatureCollection":
        feats = value.get("features") or []
        value = feats[0] if feats else None
        if value is None:
            return None
    if value.get("type") == "Feature":
        value = value.get("geometry")
    if not value or value.get("type") not in ("Polygon", "MultiPolygon"):
        return None
    return value


def _grid_geometry(outline: dict, bbox, res: int, buffer_m: float = 0):
    """The outline in the grid's projected coordinates (metres), optionally
    widened by buffer_m. Returns (shapely geometry, crs, transform, width, height)."""
    crs, transform, width, height = _aoi_grid(bbox, res)
    geom = shape(transform_geom("EPSG:4326", crs, outline))
    if not geom.is_valid:
        geom = geom.buffer(0)    # repairs small self-crossings from hand-drawn outlines
    if buffer_m:
        geom = geom.buffer(buffer_m)
    return geom, crs, transform, width, height


def outline_mask(outline: dict, bbox, res: int, buffer_m: float = OUTLINE_BUFFER_M) -> np.ndarray:
    """True for every grid pixel inside the (widened) outline."""
    geom, _, transform, width, height = _grid_geometry(outline, bbox, res, buffer_m)
    return rasterize([(mapping(geom), 1)], out_shape=(height, width), transform=transform,
                     fill=0, all_touched=True, dtype="uint8").astype(bool)


def outline_rings_px(outline: dict, bbox, res: int) -> list[list[tuple[float, float]]]:
    """The outline's rings (outer shorelines and islands) as (x, y) pixel
    positions on the grid, for drawing a border on a preview picture."""
    geom, _, transform, _, _ = _grid_geometry(outline, bbox, res)
    inv = ~transform
    polys = getattr(geom, "geoms", [geom])
    rings = []
    for p in polys:
        for ring in [p.exterior, *p.interiors]:
            rings.append([inv * (x, y) for x, y in ring.coords])
    return rings
