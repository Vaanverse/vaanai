"""
cleanup_images.py

Deletes satellite pictures from the Supabase Storage bucket `insight-images`
that NO row in `water_insights` points to any more - e.g. after you deleted old
results with SQL. (Supabase doesn't allow deleting storage files with an SQL
query; it must go through the Storage API, which is what this script uses.)

Pictures still used by a result (before_image_url / after_image_url) are never
touched. The news pictures live in a different bucket (news-images) and are
not affected.

USAGE (from the earth_monitor folder, venv active, same .env as the agent)
-----
    python cleanup_images.py            # PREVIEW: lists what would be deleted, deletes nothing
    python cleanup_images.py --delete   # really delete those files (cannot be undone)
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
BUCKET = "insight-images"
PAGE = 1000          # files listed per request
DELETE_BATCH = 100   # files removed per request


def list_bucket(client) -> list[dict]:
    """Every file at the top level of the bucket (where the agent saves them)."""
    files, offset = [], 0
    while True:
        page = client.storage.from_(BUCKET).list("", {"limit": PAGE, "offset": offset,
                                                     "sortBy": {"column": "name", "order": "asc"}})
        page = [f for f in page or [] if f.get("id")]       # entries without id are folders
        files += page
        if len(page) < PAGE:
            return files
        offset += PAGE


def used_file_names(client) -> set[str]:
    """File names that some water_insights row still points to."""
    used, start = set(), 0
    while True:
        rows = (client.table("water_insights").select("before_image_url, after_image_url")
                .range(start, start + 999).execute().data or [])
        for r in rows:
            for url in (r.get("before_image_url"), r.get("after_image_url")):
                if url:
                    used.add(url.split("?")[0].rstrip("/").split(f"/{BUCKET}/")[-1])
        if len(rows) < 1000:
            return used
        start += 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--delete", action="store_true", help="really delete the unused files")
    args = ap.parse_args()
    load_dotenv(HERE / ".env")
    sys.path.insert(0, str(HERE))
    from storage_utils import get_supabase_client, with_retries
    client = get_supabase_client()

    files = list_bucket(client)
    used = used_file_names(client)
    unused = [f for f in files if f["name"] not in used]
    size_mb = sum((f.get("metadata") or {}).get("size", 0) for f in unused) / 1_000_000

    print(f"{len(files)} picture(s) in '{BUCKET}', {len(files) - len(unused)} still used by a result, "
          f"{len(unused)} unused ({size_mb:.1f} MB).")
    for f in unused[:20]:
        print(f"   {f['name']}   {f.get('created_at', '')[:10]}")
    if len(unused) > 20:
        print(f"   ... and {len(unused) - 20} more")

    if not unused:
        return
    if not args.delete:
        print("\nPreview only - nothing was deleted. Run again with --delete to remove them.")
        return

    names = [f["name"] for f in unused]
    for i in range(0, len(names), DELETE_BATCH):
        batch = names[i:i + DELETE_BATCH]
        with_retries(f"Deleting pictures {i + 1}-{i + len(batch)}",
                     lambda b=batch: client.storage.from_(BUCKET).remove(b))
    print(f"\nDeleted {len(names)} unused picture(s).")


if __name__ == "__main__":
    main()
