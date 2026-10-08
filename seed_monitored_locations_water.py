"""
Seed the Supabase table `monitored_locations_water` from a CSV list of water bodies.

The CSV (default: locations/tn_water_bodies.csv) holds, for each water body, a
name, district, an approximate ANCHOR point and a name pattern. This script
turns that into an accurate bounding box:

  mode = osm    Looks the water body up in OpenStreetMap (Overpass API) by name,
                within `search_km` of the anchor, and uses the outline's
                bounding box (+ a small margin). Used for dams, lakes, tanks.
  mode = fixed  A square box of +/- `box_km` around the anchor. Used for river
                reaches, anicuts/barrages and salt pans, which have no single
                named outline in OpenStreetMap.

If an `osm` lookup finds nothing, the row is still saved with a box around the
anchor, but with active = false so the weekly run skips it until you have
checked it (fix the bbox in Supabase, then set active = true).

USAGE (from the earth_monitor folder, with the venv active):

    # 1. Look everything up and write a review file — touches nothing in Supabase
    python seed_monitored_locations_water.py

    #    -> check locations/tn_water_bodies_resolved.csv
    #    -> optional: drag locations/tn_water_bodies_resolved.geojson onto
    #       https://geojson.io to see every box on a map

    # 2. When happy, write the rows to Supabase (insert new / update existing)
    python seed_monitored_locations_water.py --upload

Re-running is safe: rows are matched on (name, state, country) and updated.
Note: --upload overwrites the bbox/sensor/active of rows that already exist, so
if you hand-edit a row in Supabase later, remove it from the CSV or don't re-upload.

Needs SUPABASE_URL and SUPABASE_KEY (service_role key) in .env, like the agent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE / "locations" / "tn_water_bodies.csv"
TABLE = "monitored_locations_water"

# Public Overpass servers, tried in order if one is busy.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
HEADERS = {"User-Agent": "VaanAI-location-seeder/1.0 (UniverGen Technologies)"}
PAUSE_S = 2.0            # be polite to the free Overpass servers between lookups

MARGIN_FRACTION = 0.10   # add 10% of the outline's size on each side ...
MIN_MARGIN_DEG = 0.003   # ... but at least ~300 m, so the shoreline is never clipped
SUSPICIOUS_FACTOR = 6    # matched box much bigger than expected -> flag for review

KM_PER_DEG_LAT = 111.32


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def box_around(lat: float, lon: float, half_km: float) -> tuple[float, float, float, float]:
    """Square box of +/- half_km around a point -> (min_lon, min_lat, max_lon, max_lat)."""
    dlat = half_km / KM_PER_DEG_LAT
    dlon = half_km / (KM_PER_DEG_LAT * math.cos(math.radians(lat)))
    return (round(lon - dlon, 5), round(lat - dlat, 5), round(lon + dlon, 5), round(lat + dlat, 5))


def pad_box(b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    min_lon, min_lat, max_lon, max_lat = b
    mx = max((max_lon - min_lon) * MARGIN_FRACTION, MIN_MARGIN_DEG)
    my = max((max_lat - min_lat) * MARGIN_FRACTION, MIN_MARGIN_DEG)
    return (round(min_lon - mx, 5), round(min_lat - my, 5), round(max_lon + mx, 5), round(max_lat + my, 5))


def box_size_km(b) -> tuple[float, float]:
    min_lon, min_lat, max_lon, max_lat = b
    mid_lat = (min_lat + max_lat) / 2
    w = (max_lon - min_lon) * KM_PER_DEG_LAT * math.cos(math.radians(mid_lat))
    h = (max_lat - min_lat) * KM_PER_DEG_LAT
    return w, h


# ----------------------------------------------------------------------------
# OpenStreetMap lookup
# ----------------------------------------------------------------------------
def overpass_query(name_regex: str, lat: float, lon: float, search_km: float) -> str:
    r = int(search_km * 1000)
    rx = name_regex.replace('"', "")
    around = f"(around:{r},{lat},{lon})"
    parts = []
    # Water polygons (lakes, reservoirs, tanks, lagoons) and wetlands, by name or English name.
    # Rivers/canals/streams are excluded on purpose: their polygons can run for
    # hundreds of km and would give a useless box.
    for key in ("name", "name:en"):
        for el in ("way", "relation"):
            parts.append(f'{el}{around}["natural"="water"]["water"!~"river|canal|stream|ditch|drain"]["{key}"~"{rx}",i];')
            parts.append(f'{el}{around}["landuse"="reservoir"]["{key}"~"{rx}",i];')
            parts.append(f'{el}{around}["natural"="wetland"]["{key}"~"{rx}",i];')
    return "[out:json][timeout:90];(" + "".join(parts) + ");out bb tags;"


def run_overpass(query: str) -> list[dict]:
    last_err = None
    for url in OVERPASS_URLS:
        for attempt in range(2):
            try:
                resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=120)
                if resp.status_code in (429, 504):        # busy -> wait, then retry / next server
                    time.sleep(10 * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json().get("elements", [])
            except (requests.RequestException, ValueError) as exc:
                last_err = exc
                time.sleep(5)
    raise RuntimeError(f"All Overpass servers failed: {last_err}")


def best_match(elements: list[dict], lat: float, lon: float):
    """Pick the matched outline to use: the largest one (ties broken by whether
    it contains the anchor). Returns (bbox, osm_name, osm_ref) or None."""
    cands = []
    for e in elements:
        b = e.get("bounds")
        if not b:
            continue
        box = (b["minlon"], b["minlat"], b["maxlon"], b["maxlat"])
        w, h = box_size_km(box)
        contains = box[0] <= lon <= box[2] and box[1] <= lat <= box[3]
        tags = e.get("tags", {})
        cands.append((contains, w * h, box, tags.get("name:en") or tags.get("name", ""), f"{e['type']}/{e['id']}"))
    if not cands:
        return None
    # Largest outline wins (a reservoir is usually mapped as one big polygon plus
    # small named bits like islands/bays); containing the anchor only breaks ties.
    cands.sort(key=lambda c: (c[1], c[0]), reverse=True)
    _, _, box, name, ref = cands[0]
    return box, name, ref


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def resolve(rows: list[dict]) -> list[dict]:
    out = []
    for i, row in enumerate(rows, 1):
        lat, lon = float(row["anchor_lat"]), float(row["anchor_lon"])
        box_km = float(row["box_km"])
        rec = {k: row[k] for k in ("name", "place", "district", "state", "country")}
        rec.update(sensor="auto", media_needed=True, active=True,
                   kind=row["kind"], uses=row["uses"], osm_name="", osm_ref="")

        if row["mode"] == "fixed":
            bbox = box_around(lat, lon, box_km)
            rec["status"] = "fixed box (river reach / anicut / salt pans)"
        else:
            try:
                elements = run_overpass(overpass_query(row["osm_name_regex"], lat, lon, float(row["search_km"])))
            except RuntimeError as exc:
                elements = []
                print(f"   ! lookup error: {exc}")
            m = best_match(elements, lat, lon)
            if m:
                bbox = pad_box(m[0])
                rec["osm_name"], rec["osm_ref"] = m[1], m[2]
                w, h = box_size_km(bbox)
                if max(w, h) > SUSPICIOUS_FACTOR * 2 * box_km:
                    rec["active"] = False
                    rec["status"] = f"REVIEW: OSM match is unusually large ({w:.0f} x {h:.0f} km)"
                else:
                    rec["status"] = "OSM outline"
            else:
                bbox = box_around(lat, lon, box_km)
                rec["active"] = False
                rec["status"] = "REVIEW: not found in OpenStreetMap - box around approximate anchor"
            time.sleep(PAUSE_S)

        rec["min_lon"], rec["min_lat"], rec["max_lon"], rec["max_lat"] = bbox
        w, h = box_size_km(bbox)
        rec["size_km"] = f"{w:.1f} x {h:.1f}"
        flag = "" if rec["active"] else "   <-- inactive, review"
        print(f"{i:3d}/{len(rows)} {rec['name'][:48]:48s} {rec['size_km']:>12s} km  {rec['status'][:40]}{flag}")
        out.append(rec)
    return out


def write_review_files(recs: list[dict], csv_path: Path) -> tuple[Path, Path]:
    review_csv = csv_path.with_name(csv_path.stem + "_resolved.csv")
    cols = ["name", "place", "district", "state", "country", "kind", "uses",
            "min_lon", "min_lat", "max_lon", "max_lat", "size_km",
            "sensor", "media_needed", "active", "status", "osm_name", "osm_ref"]
    with open(review_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)

    geojson = csv_path.with_name(csv_path.stem + "_resolved.geojson")
    feats = []
    for r in recs:
        x0, y0, x1, y1 = r["min_lon"], r["min_lat"], r["max_lon"], r["max_lat"]
        feats.append({
            "type": "Feature",
            "properties": {"name": r["name"], "district": r["district"], "status": r["status"],
                           "active": r["active"],
                           "stroke": "#1565c0" if r["active"] else "#d32f2f", "fill-opacity": 0.1},
            "geometry": {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]},
        })
    geojson.write_text(json.dumps({"type": "FeatureCollection", "features": feats}, indent=1), encoding="utf-8")
    return review_csv, geojson


def upload(recs: list[dict]) -> None:
    sys.path.insert(0, str(HERE))
    from storage_utils import get_supabase_client   # same client/credentials the agent uses

    table_cols = ["name", "place", "district", "state", "country",
                  "min_lon", "min_lat", "max_lon", "max_lat", "sensor", "media_needed", "active"]
    payload = [{k: r[k] for k in table_cols} for r in recs]
    client = get_supabase_client()
    resp = client.table(TABLE).upsert(payload, on_conflict="name,state,country").execute()
    if not resp.data:
        raise RuntimeError("Supabase wrote nothing - check SUPABASE_KEY is the service_role key "
                           "and that sql/2026-10-08_monitored_locations_water.sql has been run.")
    print(f"\nUpserted {len(resp.data)} rows into {TABLE}.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="input list (default: %(default)s)")
    ap.add_argument("--upload", action="store_true", help="also write the rows to Supabase")
    args = ap.parse_args()
    load_dotenv(HERE / ".env")

    with open(args.csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Resolving {len(rows)} water bodies from {args.csv.name} ...\n")

    recs = resolve(rows)
    review_csv, geojson = write_review_files(recs, args.csv)
    inactive = [r["name"] for r in recs if not r["active"]]
    print(f"\nReview file : {review_csv}\nMap preview : {geojson}  (open at https://geojson.io)")
    print(f"Active: {len(recs) - len(inactive)}   Needs review (saved as inactive): {len(inactive)}")
    for n in inactive:
        print(f"   - {n}")

    if args.upload:
        upload(recs)
    else:
        print("\nDry run - nothing written to Supabase. Re-run with --upload when the review file looks right.")


if __name__ == "__main__":
    main()
