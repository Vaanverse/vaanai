"""
list_monitored_locations.py

Reads the ACTIVE rows of the Supabase table `monitored_locations_water` and
prints them as the job list ("matrix") for the GitHub workflow
.github/workflows/weekly-run.yml.

On GitHub Actions it writes two outputs for the next job:
    locations = JSON list of {id, name, bbox, sensor, media_needed}
    count     = number of locations

Run it locally to see what the next workflow run would do:
    python list_monitored_locations.py

Only needs `supabase` and `python-dotenv` (not the heavy satellite libraries),
so the workflow's first job installs just those two.
"""

import json
import os
import sys

from dotenv import load_dotenv
from supabase import create_client

TABLE = "monitored_locations_water"
GITHUB_MATRIX_LIMIT = 256   # GitHub refuses a matrix with more jobs than this


def main() -> None:
    load_dotenv()
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_KEY not set")

    rows = (
        create_client(url, key)
        .table(TABLE)
        .select("id, name, min_lon, min_lat, max_lon, max_lat, sensor, media_needed")
        .eq("active", True)
        .order("id")
        .execute()
        .data
    )

    locations = [
        {
            "id": r["id"],
            "name": r["name"],
            "bbox": f'{r["min_lon"]} {r["min_lat"]} {r["max_lon"]} {r["max_lat"]}',
            "sensor": r["sensor"] or "auto",
            "media_needed": bool(r["media_needed"]),
        }
        for r in rows
    ]

    if len(locations) > GITHUB_MATRIX_LIMIT:
        print(f"::warning::{len(locations)} active locations, but GitHub allows only "
              f"{GITHUB_MATRIX_LIMIT} jobs per run - running the first {GITHUB_MATRIX_LIMIT} (lowest id).")
        locations = locations[:GITHUB_MATRIX_LIMIT]

    print(f"{len(locations)} active location(s):")
    for loc in locations:
        media = "images" if loc["media_needed"] else "no images"
        print(f'  #{loc["id"]:<4} {loc["name"]:<50} sensor={loc["sensor"]:<4} {media}')

    gh_output = os.getenv("GITHUB_OUTPUT")
    if gh_output:   # running on GitHub Actions -> hand the list to the next job
        with open(gh_output, "a", encoding="utf-8") as f:
            f.write(f"locations={json.dumps(locations)}\n")
            f.write(f"count={len(locations)}\n")


if __name__ == "__main__":
    main()
