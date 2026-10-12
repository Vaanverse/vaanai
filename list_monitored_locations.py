"""
list_monitored_locations.py

Picks TODAY'S BATCH of water bodies from the Supabase table
`monitored_locations_water` and prints it as the job list ("matrix") for the
GitHub workflow .github/workflows/weekly-run.yml.

Daily batches and yearly cycles (run_index)
-------------------------------------------
Every day the workflow runs only BATCH_SIZE (10) active water bodies, in
ascending id order. The column run_index records which yearly cycle each row
has reached:

  - rows still below the current cycle are picked first (lowest id first) and
    moved up to that cycle;
  - once every active row has reached the current cycle, the next run starts
    the next cycle from the lowest id again;
  - after MAX_CYCLES (5) cycles nothing more is picked, and the workflow
    simply has nothing to run.

With 100 water bodies that is one full cycle every 10 days. Cycle 1 compares
last month with the same month a year earlier; each later cycle steps one more
year back (see agentic_water_monitor_groq.py).

The picking and the run_index update happen in ONE database call, the
function public.claim_next_water_batch (sql/2026-10-12_run_index.sql).

Usage
-----
    python list_monitored_locations.py            # PREVIEW only: shows the next batch, changes nothing
    python list_monitored_locations.py --claim    # marks the batch in the table (the workflow does this)

On GitHub Actions it writes two outputs for the next job:
    locations = JSON list of {id, name, bbox, sensor, media_needed, run_index}
    count     = number of locations

Only needs `supabase` and `python-dotenv` (not the heavy satellite libraries),
so the workflow's first job installs just those two.
"""

import argparse
import json
import os
import sys

from dotenv import load_dotenv
from supabase import create_client

TABLE = "monitored_locations_water"
BATCH_SIZE = 10    # water bodies per daily run
MAX_CYCLES = 5     # yearly cycles: 5 -> five years of comparisons per water body
COLUMNS = "id, name, min_lon, min_lat, max_lon, max_lat, sensor, media_needed, run_index"


def preview_batch(client) -> list[dict]:
    """Same rule as claim_next_water_batch, but read-only: what the next
    --claim WOULD pick, with the run_index it WOULD get."""
    rows = client.table(TABLE).select(COLUMNS).eq("active", True).order("id").execute().data
    cycle = max((r["run_index"] or 0 for r in rows), default=0)
    behind = [r for r in rows if (r["run_index"] or 0) < cycle]
    if not behind:
        if cycle >= MAX_CYCLES:
            return []
        cycle += 1
        behind = [r for r in rows if (r["run_index"] or 0) < cycle]
    return [{**r, "run_index": cycle} for r in behind[:BATCH_SIZE]]


def claim_batch(client) -> list[dict]:
    """Marks the next batch in the table and returns it (one atomic call)."""
    rows = client.rpc("claim_next_water_batch",
                      {"batch_size": BATCH_SIZE, "max_cycle": MAX_CYCLES}).execute().data or []
    return sorted(rows, key=lambda r: r["id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick today's batch of water bodies.")
    parser.add_argument("--claim", action="store_true",
                        help="Update run_index in the table for the picked rows (used by the workflow). "
                             "Without it, only shows what would be picked.")
    args = parser.parse_args()

    load_dotenv()
    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
    if not url or not key:
        sys.exit("SUPABASE_URL / SUPABASE_KEY not set")
    client = create_client(url, key)

    rows = claim_batch(client) if args.claim else preview_batch(client)

    locations = [
        {
            "id": r["id"],
            "name": r["name"],
            "bbox": f'{r["min_lon"]} {r["min_lat"]} {r["max_lon"]} {r["max_lat"]}',
            "sensor": r["sensor"] or "auto",
            "media_needed": bool(r["media_needed"]),
            "run_index": r["run_index"],
        }
        for r in rows
    ]

    if not locations:
        print(f"Nothing to run: every active location has finished all {MAX_CYCLES} cycles "
              "(or there are no active locations).")
    else:
        verb = "Claimed" if args.claim else "PREVIEW (nothing changed) - next batch would be"
        print(f"{verb} {len(locations)} location(s), cycle {locations[0]['run_index']}:")
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
