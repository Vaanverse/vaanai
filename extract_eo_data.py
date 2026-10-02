"""
extract_eo_data.py

Searches and downloads Earth Observation data from NASA's Common Metadata
Repository (CMR) — the catalog behind every NASA data archive (Earthdata,
LP DAAC, ASF, etc.) — using the official `earthaccess` client.

Usage examples
--------------
List datasets matching a keyword (useful for finding the right short_name):
    python extract_eo_data.py --list-datasets "surface reflectance"

Search + download Harmonized Landsat Sentinel-2 (HLS) scenes over an AOI:
    python extract_eo_data.py \
        --short-name HLSL30 \
        --bbox 77.55 12.90 77.65 13.00 \
        --start-date 2025-01-01 \
        --end-date 2025-03-01 \
        --max-results 5 \
        --output-dir ./data/hls

Notes
-----
- bbox order is: min_lon min_lat max_lon max_lat (a bounding box, not a
  single point). The example above is roughly central Bengaluru.
- Auth: put your NASA Earthdata Login credentials in a .env file
  (copy .env.example -> .env) before running. earthaccess also supports
  interactive login and a ~/.netrc file if you prefer those instead.
- This script only touches NASA's own CMR/Earthdata infrastructure. It does
  NOT touch ISRO/Bhuvan/MOSDAC data, which has commercial-use restrictions —
  keep those pipelines separate if you build them later.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import earthaccess
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def authenticate() -> earthaccess.Auth:
    """
    Logs in to NASA Earthdata. Tries, in order:
      1. EARTHDATA_USERNAME / EARTHDATA_PASSWORD from the environment (.env)
      2. A ~/.netrc file, if one exists
      3. An interactive prompt, as a last resort

    Raises SystemExit with a clear message if authentication fails, so a
    scheduled/unattended run fails loudly instead of hanging on a prompt.
    """
    load_dotenv()  # reads .env into the process environment, if present

    username = os.getenv("EARTHDATA_USERNAME")
    password = os.getenv("EARTHDATA_PASSWORD")

    if username and password:
        os.environ["EARTHDATA_USERNAME"] = username
        os.environ["EARTHDATA_PASSWORD"] = password
        strategy = "environment"
    else:
        strategy = "netrc"  # falls back further to interactive if netrc is missing

    try:
        auth = earthaccess.login(strategy=strategy)
    except Exception as exc:  # noqa: BLE001 - we want to surface any auth failure clearly
        log.error("Earthdata authentication failed: %s", exc)
        sys.exit(1)

    if not auth.authenticated:
        log.error(
            "Could not authenticate with NASA Earthdata. "
            "Check EARTHDATA_USERNAME/EARTHDATA_PASSWORD in your .env file."
        )
        sys.exit(1)

    log.info("Authenticated with NASA Earthdata as '%s'.", username or "netrc user")
    return auth


def list_matching_datasets(keyword: str, limit: int = 15) -> None:
    """Prints dataset short_names matching a keyword — use this to discover
    the right --short-name before running a real search/download."""
    log.info("Searching NASA CMR for datasets matching: '%s'", keyword)
    results = earthaccess.search_datasets(keyword=keyword, count=limit)

    if not results:
        log.info("No datasets found for that keyword. Try a broader term.")
        return

    for ds in results:
        short_name = ds.get("umm", {}).get("ShortName", "UNKNOWN")
        title = ds.get("umm", {}).get("EntryTitle", "Untitled")
        print(f"  {short_name:<20}  {title}")


def search_granules(
    short_name: str,
    bbox: tuple[float, float, float, float] | None,
    start_date: str | None,
    end_date: str | None,
    max_results: int,
    cloud_cover: float | None = None,
) -> list:
    """
    Searches CMR for individual data files ("granules") for a given dataset.

    A granule is one scene/tile at one point in time — this is the thing
    you actually download, as opposed to the dataset (collection) itself.
    If cloud_cover is given (0-100), only tiles at or below that cloud % are returned.
    """
    kwargs: dict = {"short_name": short_name, "count": max_results}

    if bbox:
        kwargs["bounding_box"] = bbox  # (min_lon, min_lat, max_lon, max_lat)
    if start_date or end_date:
        kwargs["temporal"] = (start_date, end_date)
    if cloud_cover is not None:
        kwargs["cloud_cover"] = (0, cloud_cover)   # tile-level cloud % filter

    log.info("Searching for granules: %s", kwargs)
    granules = earthaccess.search_data(**kwargs)
    log.info("Found %d granule(s).", len(granules))
    return granules


def download_granules(granules: list, output_dir: str) -> list[str]:
    """Downloads a list of granules to output_dir, creating it if needed."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not granules:
        log.info("Nothing to download.")
        return []

    log.info("Downloading %d granule(s) to %s ...", len(granules), output_dir)
    downloaded_paths = earthaccess.download(granules, local_path=output_dir)
    log.info("Download complete.")
    return downloaded_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--list-datasets",
        metavar="KEYWORD",
        help="Search dataset names/titles for a keyword and exit (no download).",
    )
    parser.add_argument("--short-name", help="CMR dataset short_name, e.g. HLSL30, MOD13Q1.")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box for the area of interest.",
    )
    parser.add_argument("--start-date", help="YYYY-MM-DD")
    parser.add_argument("--end-date", help="YYYY-MM-DD")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--output-dir", default="./data", help="Where downloaded files are saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authenticate()

    if args.list_datasets:
        list_matching_datasets(args.list_datasets)
        return

    if not args.short_name:
        log.error("You must pass --short-name (or use --list-datasets to find one first).")
        sys.exit(1)

    bbox = tuple(args.bbox) if args.bbox else None
    granules = search_granules(
        short_name=args.short_name,
        bbox=bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        max_results=args.max_results,
    )
    download_granules(granules, args.output_dir)


if __name__ == "__main__":
    main()
