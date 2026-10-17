"""
fix_outlines.py

Gives every water body in `monitored_locations_water` its REAL shape: looks up
its outline (shoreline polygon) in OpenStreetMap and saves it in the `outline`
column, and resets the bounding box to fit that outline with a margin, so the
pictures show the whole water body and the agent measures only this water body.

Why: the boxes in the table were squares around approximate centre points (the
OpenStreetMap lookup in seed_monitored_locations_water.py never succeeded), so
some pictures cut off part of the lake (e.g. Kolavai Lake) and some boxes
also caught other ponds and fields.

How a water body is matched
---------------------------
  1. By NAME: water / reservoir / wetland outlines within the search radius
     whose name matches (the osm_name_regex column if set, otherwise words
     taken from the name, e.g. "Mettur Dam (Stanley Reservoir)" -> Mettur|Stanley).
     Pieces of the same water body mapped separately are joined together.
  2. If nothing matches by name: the water outline that CONTAINS the centre of
     the current box (or the closest one within 1.5 km) - marked "REVIEW",
     because it may be a different pond.
  3. Not found at all: the row is left unchanged and listed, so you can draw it
     yourself (see "Drawing an outline by hand" below).

USAGE (from the earth_monitor folder, venv active, same .env as the agent)
-----
    pip install -r requirements.txt            # adds shapely

    # 1. Look everything up - writes review files, changes NOTHING in Supabase
    python fix_outlines.py

    #    -> open locations/outlines_review.geojson at https://geojson.io
    #       yellow = new outline, blue = new box, red dashed = old box
    #    -> check locations/outlines_review.csv (status column)

    # 2. Save the outlines + new boxes to Supabase
    python fix_outlines.py --upload             # only rows with status OK
    python fix_outlines.py --upload --include-review   # also the REVIEW ones you checked

    # Only some water bodies (by id) - the review files still list all of them:
    python fix_outlines.py --ids 17 23 41

    # Just rebuild the review files from the saved results (no searching):
    python fix_outlines.py --review-only

    # Use hand-drawn outlines + earlier results, never search (e.g. for the upload):
    python fix_outlines.py --no-search
    python fix_outlines.py --no-search --upload --include-review

Each lookup is saved in locations/outlines_cache.json as soon as it is done, so
you can stop the script (Ctrl+C) and run it again: it continues where it left
off, and --upload reuses the saved lookups instead of searching again. Only
finished matches (OK / REVIEW) are reused; NOT FOUND and LOOKUP FAILED are
searched again on every run. --refresh starts from scratch.

River stretches, anicuts/barrages, estuaries and salt pans are not looked up:
they have no single outline, so they keep their box (status KEEP BOX).

Drawing an outline by hand
--------------------------
For a water body OpenStreetMap doesn't have (or got wrong): at
https://geojson.io draw a polygon around it, click it, add a property
"id" = its id in monitored_locations_water, and save all of them as
locations/manual_outlines.geojson. Then run

    python fix_outlines.py --ids <those ids> --upload

Hand-drawn outlines always win over OpenStreetMap.

Needs: SUPABASE_URL / SUPABASE_KEY (service_role key) in .env, and the SQL in
sql/2026-10-15_outlines.sql run once in Supabase.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from shapely.geometry import LineString, Point, Polygon, mapping, shape
from shapely.ops import linemerge, polygonize, unary_union

HERE = Path(__file__).resolve().parent
TABLE = "monitored_locations_water"
REVIEW_CSV = HERE / "locations" / "outlines_review.csv"
REVIEW_GEOJSON = HERE / "locations" / "outlines_review.geojson"
MANUAL_GEOJSON = HERE / "locations" / "manual_outlines.geojson"
CACHE_JSON = HERE / "locations" / "outlines_cache.json"   # results of earlier lookups, so runs can resume

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
HEADERS = {"User-Agent": "VaanAI-outline-fixer/1.0 (UniverGen Technologies)"}
PAUSE_S = 1.0                # be polite to the free Overpass servers

MIN_SEARCH_KM = 8            # search at least this far from the current box centre
NEAREST_MAX_KM = 1.5         # fallback 2: an unnamed outline must be this close
MERGE_GAP_DEG = 0.005        # join same-named pieces less than ~500 m apart
SIMPLIFY_DEG = 0.0001        # ~11 m: keeps the outline accurate but small
MARGIN_FRACTION = 0.15       # new box = outline + 15% on each side ...
MIN_MARGIN_DEG = 0.004       # ... but at least ~450 m, so a grown lake still fits
MAX_OUTLINE_KM2 = 600        # bigger than any TN reservoir -> probably a wrong match
KM_PER_DEG = 111.32

# Words that are part of almost every name, so they can't identify a water body.
GENERIC = {"dam", "lake", "lakes", "reservoir", "tank", "big", "eri", "kulam", "cheruvu", "odai",
           "the", "of", "and", "upper", "lower", "new", "old", "river", "anicut", "barrage",
           "lagoon", "pond", "check", "sagar", "station", "project", "city", "reach", "town"}


# ----------------------------------------------------------------------------
# Small geometry helpers (degrees -> km near the given latitude)
# ----------------------------------------------------------------------------
def area_km2(geom) -> float:
    lat = geom.centroid.y
    return geom.area * KM_PER_DEG * KM_PER_DEG * math.cos(math.radians(lat))


def dist_km(geom, lon: float, lat: float) -> float:
    d = geom.distance(Point(lon, lat))          # degrees (rough, fine for ranking)
    return d * KM_PER_DEG


LANDSCAPE = 16 / 10          # picture shape (width / height) - same as the website's before/after viewer


def padded_bbox(geom) -> tuple[float, float, float, float]:
    """The outline's box + margin, then widened (or heightened) to a 16:10
    LANDSCAPE shape, so every picture has the same shape as the website's
    viewer and the whole water body is visible. The extra area is only land
    around the lake - the measurement counts water inside the outline only."""
    min_lon, min_lat, max_lon, max_lat = geom.bounds
    mx = max((max_lon - min_lon) * MARGIN_FRACTION, MIN_MARGIN_DEG)
    my = max((max_lat - min_lat) * MARGIN_FRACTION, MIN_MARGIN_DEG)
    min_lon, max_lon, min_lat, max_lat = min_lon - mx, max_lon + mx, min_lat - my, max_lat + my
    k = math.cos(math.radians((min_lat + max_lat) / 2))      # km per degree of longitude, relative
    w, h = (max_lon - min_lon) * k, (max_lat - min_lat)        # in "latitude degrees"
    if w / h < LANDSCAPE:                                      # too tall -> add land left and right
        extra = (h * LANDSCAPE - w) / k / 2
        min_lon, max_lon = min_lon - extra, max_lon + extra
    else:                                                      # too wide -> add land above and below
        extra = (w / LANDSCAPE - h) / 2
        min_lat, max_lat = min_lat - extra, max_lat + extra
    return (round(min_lon, 5), round(min_lat, 5), round(max_lon, 5), round(max_lat, 5))


def box_km(b) -> str:
    w = (b[2] - b[0]) * KM_PER_DEG * math.cos(math.radians((b[1] + b[3]) / 2))
    h = (b[3] - b[1]) * KM_PER_DEG
    return f"{w:.1f} x {h:.1f}"


def clean_geometry(geom) -> dict:
    """Simplified, rounded GeoJSON geometry, small enough to store in the table."""
    g = geom.simplify(SIMPLIFY_DEG, preserve_topology=True)
    if not g.is_valid:
        g = g.buffer(0)
    gj = mapping(g)

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], (list, tuple)) else [round(c[0], 6), round(c[1], 6)]
    return {"type": gj["type"], "coordinates": rnd(gj["coordinates"])}


# ----------------------------------------------------------------------------
# OpenStreetMap (Overpass) lookup
# ----------------------------------------------------------------------------
def name_regex(row: dict) -> str:
    """osm_name_regex from the table if set, else distinctive words of the name."""
    if (row.get("osm_name_regex") or "").strip():
        return row["osm_name_regex"].strip()
    words = re.findall(r"[A-Za-z][A-Za-z.\-']+", row["name"])
    keep = [w.strip(".-'") for w in words if w.lower().strip(".-'") not in GENERIC and len(w) > 2]
    return "|".join(dict.fromkeys(keep)) or re.escape(row["name"])


# Tag filters Overpass can look up from its index (fast). A key-pattern filter like
# [~"^(natural|landuse)$"~...] can't use the index and timed out on large areas.
WATER_FILTERS = ['["natural"="water"]["water"!~"river|canal|stream|ditch|drain|moat"]',
                 '["landuse"="reservoir"]',
                 '["natural"="wetland"]']
NAME_KEYS = "^(name|name:en|name:ta|alt_name|old_name)$"


def overpass_query(regex: str | None, lat: float, lon: float, radius_km: float) -> str:
    """ONE request per water body: lakes / reservoirs / wetlands within the radius
    whose name (in any of the usual name fields) matches - or, with no regex,
    every lake/reservoir outline near the point (fallback 2)."""
    around = f"(around:{int(radius_km * 1000)},{lat},{lon})"
    name = f'[~"{NAME_KEYS}"~"{regex.replace(chr(34), "")}",i]' if regex else ""
    filters = WATER_FILTERS if regex else WATER_FILTERS[:2]
    parts = "".join(f"{el}{around}{f}{name};" for f in filters for el in ("way", "relation"))
    return f"[out:json][timeout:120];({parts});out geom tags;"


def run_overpass(query: str) -> list[dict]:
    last_err = None
    for url in OVERPASS_URLS:
        host = url.split("/")[2]
        for attempt in range(2):
            try:
                resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=120)
                if resp.status_code in (429, 502, 503, 504):     # busy -> short wait, then retry / next server
                    last_err = f"HTTP {resp.status_code} from {host}"
                    wait = 10 * (attempt + 1)
                    print(f"      {host} is busy ({resp.status_code}) - waiting {wait}s ...", flush=True)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                remark = data.get("remark") or ""
                if "error" in remark.lower() or "timed out" in remark.lower():
                    # Overpass answers "200 OK" with NO results when a search runs out of
                    # time or memory - that must not be read as "lake not found".
                    last_err = f"{host}: {remark[:80]}"
                    print(f"      {host} could not finish the search ({remark[:60]}) - trying again ...", flush=True)
                    time.sleep(5)
                    continue
                return data.get("elements", [])
            except (requests.RequestException, ValueError) as exc:
                last_err = exc
                print(f"      {host} failed ({type(exc).__name__}) - trying again ...", flush=True)
                time.sleep(3)
    raise RuntimeError(f"All Overpass servers failed: {last_err}")


def element_geometry(el: dict):
    """A shapely (Multi)Polygon for one OSM way or multipolygon relation, or None."""
    if el["type"] == "way":
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", [])]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            poly = Polygon(pts)
            return poly if poly.is_valid else poly.buffer(0)
        return None
    outers, inners = [], []
    for m in el.get("members", []):
        if m.get("type") != "way" or not m.get("geometry"):
            continue
        line = LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
        (inners if m.get("role") == "inner" else outers).append(line)
    if not outers:
        return None
    outer = unary_union(list(polygonize(linemerge(outers))))
    if outer.is_empty:
        return None
    if inners:
        holes = unary_union(list(polygonize(linemerge(inners))))
        if not holes.is_empty:
            outer = outer.difference(holes)
    return outer if outer.is_valid else outer.buffer(0)


def candidates(elements: list[dict]) -> list[tuple]:
    out = []
    for el in elements:
        try:
            g = element_geometry(el)
        except Exception:  # noqa: BLE001 - one broken outline shouldn't stop the run
            g = None
        if g is not None and not g.is_empty and g.area > 0:
            tags = el.get("tags", {})
            out.append((g, tags.get("name:en") or tags.get("name", ""), f"{el['type']}/{el['id']}"))
    return out


def match_by_name(cands, lon, lat):
    """Largest name-matched outline (preferring ones near the centre), joined with
    other same-name pieces close to it."""
    if not cands:
        return None
    cands.sort(key=lambda c: (dist_km(c[0], lon, lat) < 3, area_km2(c[0])), reverse=True)
    main, name, ref = cands[0]
    near = main.buffer(MERGE_GAP_DEG)
    parts = [c for c in cands if c[0].intersects(near)]
    geom = unary_union([c[0] for c in parts])
    refs = ",".join(sorted({c[2] for c in parts}))
    return geom, name, refs


def match_by_position(cands, lon, lat):
    """The outline containing the point, else the closest one within NEAREST_MAX_KM."""
    pt = Point(lon, lat)
    inside = [c for c in cands if c[0].contains(pt)]
    if inside:
        return max(inside, key=lambda c: area_km2(c[0]))
    near = [c for c in cands if dist_km(c[0], lon, lat) <= NEAREST_MAX_KM and area_km2(c[0]) >= 0.02]
    return min(near, key=lambda c: dist_km(c[0], lon, lat)) if near else None


def _manual_geometry(geom: dict | None) -> dict | None:
    """A drawn shape as a Polygon/MultiPolygon. A line traced around the shore
    (geojson.io's line tool) is accepted too, and closed automatically."""
    if not geom:
        return None
    if geom.get("type") in ("Polygon", "MultiPolygon"):
        return geom
    if geom.get("type") == "LineString" and len(geom.get("coordinates", [])) >= 4:
        pts = [tuple(p[:2]) for p in geom["coordinates"]]
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return mapping(poly) if not poly.is_empty else None
    return None


def load_manual() -> dict:
    """{id or lower-case name: geometry} from locations/manual_outlines.geojson.
    Shapes without an "id" (or "name") property are ignored, with a note."""
    if not MANUAL_GEOJSON.exists():
        return {}
    data = json.loads(MANUAL_GEOJSON.read_text(encoding="utf-8"))
    out, skipped = {}, 0
    for f in data.get("features", []):
        props, geom = f.get("properties") or {}, _manual_geometry(f.get("geometry"))
        if geom is None:
            continue          # points, search markers, empty circles ...
        key = props.get("id")
        if key is not None and str(key).strip().isdigit():
            out[int(key)] = geom
        elif props.get("name"):
            out[str(props["name"]).strip().lower()] = geom
        else:
            skipped += 1
    if skipped:
        print(f"Note: {skipped} drawn shape(s) in {MANUAL_GEOJSON.name} have no \"id\" property and were ignored.")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def load_cache() -> dict:
    try:
        return {int(k): v for k, v in json.loads(CACHE_JSON.read_text(encoding="utf-8")).items()}
    except (OSError, ValueError):
        return {}


def save_cache(cache: dict) -> None:
    CACHE_JSON.parent.mkdir(exist_ok=True)
    CACHE_JSON.write_text(json.dumps(cache), encoding="utf-8")


def to_cache(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("status", "source", "osm_name", "regex", "area_km2", "outline", "new_bbox")}


def from_cache(row: dict, hit: dict) -> dict:
    old = (row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"])
    rec = {"id": row["id"], "name": row["name"], "district": row.get("district") or "", "old_bbox": old,
           "source": "", "osm_name": "", **hit,
           "geom": shape(hit["outline"]) if hit.get("outline") else None}
    if rec["geom"] is not None:
        rec["new_bbox"] = padded_bbox(rec["geom"])     # always the current box rule (16:10 landscape)
    elif rec.get("new_bbox"):
        rec["new_bbox"] = tuple(rec["new_bbox"])
    return rec


# River stretches, anicuts/barrages, estuaries and salt pans have no single
# outline in OpenStreetMap; a search by position only finds some nearby pond or
# temple tank (e.g. "Cauvery River - Srirangam reach" matched a temple tank).
# They keep their box and the agent keeps measuring all water inside it.
KEEP_BOX = re.compile(r"\b(river|anicut|barrage|estuary|salt|swamp|reach|mukkombu|kallanai)\b", re.I)
KEEP_BOX_STATUS = "KEEP BOX - river stretch / anicut / salt pans: no single outline, box kept"


def resolve(row: dict, manual: dict) -> dict:
    old = (row["min_lon"], row["min_lat"], row["max_lon"], row["max_lat"])
    lon, lat = (old[0] + old[2]) / 2, (old[1] + old[3]) / 2
    rec = {"id": row["id"], "name": row["name"], "district": row.get("district") or "",
           "old_bbox": old, "status": "", "source": "", "osm_name": "", "geom": None}

    man = manual.get(row["id"]) or manual.get(row["name"].strip().lower())
    if man:
        rec.update(geom=shape(man), source="manual", status="OK (hand-drawn)")
    elif KEEP_BOX.search(row["name"]):
        rec["status"] = KEEP_BOX_STATUS
    else:
        half_diag_km = math.hypot((old[2] - old[0]) * KM_PER_DEG, (old[3] - old[1]) * KM_PER_DEG) / 2
        radius = max(MIN_SEARCH_KM, half_diag_km + 3)
        regex = name_regex(row)
        try:
            m = match_by_name(candidates(run_overpass(overpass_query(regex, lat, lon, radius))), lon, lat)
            if m:
                rec.update(geom=m[0], osm_name=m[1], source=f"osm:{m[2]}", status="OK (name match)")
            else:
                time.sleep(PAUSE_S)
                m = match_by_position(candidates(run_overpass(overpass_query(None, lat, lon, NEAREST_MAX_KM + 1))), lon, lat)
                if m:
                    rec.update(geom=m[0], osm_name=m[1], source=f"osm:{m[2]}",
                               status="REVIEW: matched by position, not by name - check it is the right water body")
                else:
                    rec["status"] = "NOT FOUND - draw it by hand (see top of this script)"
        except RuntimeError as exc:
            rec["status"] = f"LOOKUP FAILED - run again later ({exc})"
        rec["regex"] = regex
        time.sleep(PAUSE_S)

    if rec["geom"] is not None:
        a = area_km2(rec["geom"])
        rec["area_km2"] = round(a, 3)
        if a > MAX_OUTLINE_KM2 and rec["status"].startswith("OK") and rec["source"] != "manual":
            rec["status"] = f"REVIEW: outline is very large ({a:.0f} km²) - probably the wrong match"
        rec["new_bbox"] = padded_bbox(rec["geom"])
        rec["outline"] = clean_geometry(rec["geom"])
    return rec


def write_review(recs: list[dict]) -> None:
    REVIEW_CSV.parent.mkdir(exist_ok=True)
    with open(REVIEW_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "district", "status", "source", "osm_name", "searched_for",
                    "water_area_km2", "old_box_km", "new_box_km"])
        for r in recs:
            w.writerow([r["id"], r["name"], r["district"], r["status"], r["source"], r["osm_name"],
                        r.get("regex", ""), r.get("area_km2", ""), box_km(r["old_bbox"]),
                        box_km(r["new_bbox"]) if r.get("new_bbox") else ""])

    def box_feature(b, props):
        x0, y0, x1, y1 = b
        return {"type": "Feature", "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}}

    feats = []
    for r in recs:
        base = {"id": r["id"], "name": r["name"], "status": r["status"]}
        feats.append(box_feature(r["old_bbox"], {**base, "what": "old box", "stroke": "#d32f2f",
                                                 "stroke-width": 1, "fill-opacity": 0}))
        if r.get("outline"):
            feats.append(box_feature(r["new_bbox"], {**base, "what": "new box", "stroke": "#1565c0",
                                                     "stroke-width": 2, "fill-opacity": 0}))
            feats.append({"type": "Feature", "geometry": r["outline"],
                          "properties": {**base, "what": "outline", "area_km2": r.get("area_km2"),
                                         "stroke": "#f9a825", "stroke-width": 2,
                                         "fill": "#fdd835", "fill-opacity": 0.25}})
    REVIEW_GEOJSON.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


def upload(client, recs: list[dict], include_review: bool) -> int:
    n = 0
    for r in recs:
        if not r.get("outline"):
            continue
        if not (r["status"].startswith("OK") or (include_review and r["status"].startswith("REVIEW"))):
            continue
        b = r["new_bbox"]
        resp = (client.table(TABLE)
                .update({"outline": r["outline"], "outline_source": r["source"],
                         "min_lon": b[0], "min_lat": b[1], "max_lon": b[2], "max_lat": b[3]})
                .eq("id", r["id"]).execute())
        if not resp.data:
            raise RuntimeError(f"Supabase wrote nothing for id {r['id']} - is SUPABASE_KEY the service_role "
                               "key, and has sql/2026-10-15_outlines.sql been run?")
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upload", action="store_true", help="save outlines + new boxes to Supabase")
    ap.add_argument("--include-review", action="store_true", help="with --upload: also save REVIEW rows")
    ap.add_argument("--ids", type=int, nargs="*",
                    help="look up only these monitored_locations_water ids (the review files still list all)")
    ap.add_argument("--refresh", action="store_true",
                    help="ignore results saved by earlier runs and look everything up again")
    ap.add_argument("--no-search", action="store_true",
                    help="never search OpenStreetMap: use hand-drawn outlines and results found earlier only")
    ap.add_argument("--review-only", action="store_true",
                    help="only rebuild the review files from saved results - no searching, nothing uploaded")
    args = ap.parse_args()
    load_dotenv(HERE / ".env")

    sys.path.insert(0, str(HERE))
    from storage_utils import get_supabase_client   # same credentials as the agent
    client = get_supabase_client()
    # All ACTIVE water bodies (inactive ones, e.g. replaced river stretches, are skipped).
    rows = client.table(TABLE).select("*").eq("active", True).order("id").execute().data or []
    selected = set(args.ids) if args.ids else {r["id"] for r in rows}
    if args.review_only:
        selected = set()
    manual = load_manual()
    todo = sum(r["id"] in selected for r in rows)
    print(f"{len(rows)} active water bodies; working on {todo}"
          + (f" ({len(manual)} hand-drawn outline(s) found)" if manual else "") + " ...\n")

    # Earlier lookups are reused (so a stopped run continues where it left off,
    # and --upload doesn't look everything up again). Failed lookups are retried.
    # Rows not selected with --ids are taken from the saved results as they are,
    # so the review files always show ALL water bodies.
    cache = {} if args.refresh else load_cache()
    recs = []
    for i, row in enumerate(rows, 1):
        hit = cache.get(row["id"])
        if row["id"] not in selected:
            if KEEP_BOX.search(row["name"]):
                hit = {"status": KEEP_BOX_STATUS}
            r = from_cache(row, hit) if hit else from_cache(row, {"status": "NOT LOOKED UP YET"})
            recs.append(r)
            continue
        reuse = (hit and hit["status"].startswith(("OK", "REVIEW"))
                 and not KEEP_BOX.search(row["name"])
                 and not (manual.get(row["id"]) or manual.get(row["name"].strip().lower())))
        is_manual = bool(manual.get(row["id"]) or manual.get(row["name"].strip().lower()))
        if reuse:
            r, note = from_cache(row, hit), "  (saved earlier)"
        elif args.no_search and not is_manual and not KEEP_BOX.search(row["name"]):
            # --no-search: keep whatever was found before (NOT FOUND etc.) without searching
            r = from_cache(row, hit) if hit else from_cache(row, {"status": "NOT LOOKED UP YET"})
            recs.append(r)
            continue
        else:
            if not is_manual:
                print(f"{i:3d}/{len(rows)} #{row['id']:<4} {row['name'][:42]} - looking up ...", flush=True)
            r, note = resolve(row, manual), ""
            cache[row["id"]] = to_cache(r)
            save_cache(cache)
        recs.append(r)
        print(f"{i:3d}/{len(rows)} #{r['id']:<4} {r['name'][:42]:42s} "
              f"{str(r.get('area_km2', '-')):>9} km²  {r['status'][:60]}{note}", flush=True)

    write_review(recs)
    ok = sum(r["status"].startswith("OK") for r in recs)
    review = [r for r in recs if r["status"].startswith("REVIEW")]
    missing = [r for r in recs if not r.get("outline") and not r["status"].startswith("KEEP BOX")]
    kept = [r for r in recs if r["status"].startswith("KEEP BOX")]
    print(f"\nOK: {ok}   Needs a look (REVIEW): {len(review)}   No outline: {len(missing)}   "
          f"Box kept (river / anicut / salt pans): {len(kept)}")
    for r in review + missing:
        print(f"   #{r['id']:<4} {r['name']}: {r['status']}")
    print(f"\nReview table: {REVIEW_CSV}\nMap:          {REVIEW_GEOJSON}  (open at https://geojson.io)")

    if args.upload and not args.review_only:
        n = upload(client, [r for r in recs if r["id"] in selected], args.include_review)
        print(f"\nSaved {n} outline(s) and box(es) to Supabase.")
    else:
        print("\nNothing was changed in Supabase. When the map looks right, run again with --upload.")


if __name__ == "__main__":
    main()
