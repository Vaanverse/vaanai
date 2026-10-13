"""
usage_report.py

Adds up the Groq token usage of one night's run. Each agent job saves its own
groq_usage.json (see report_usage() in agentic_water_monitor_groq.py); the
workflow's last job downloads them all into one folder and runs:

    python usage_report.py usage/

It prints, and on GitHub Actions also writes to the run's Summary page:
  - tokens per water body,
  - the total for the night,
  - tokens per minute (UTC) and the busiest 60 seconds across the whole run,
  - the latest Groq limit figures seen.
"""

import glob
import json
import os
import sys
from datetime import datetime, timedelta


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    runs = []
    for path in sorted(glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)):
        try:
            with open(path, encoding="utf-8") as f:
                runs.append(json.load(f))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
    if not runs:
        print("No usage files found - nothing to report.")
        return
    runs.sort(key=lambda r: (r.get("location_id") is None, r.get("location_id") or 0))

    calls = [c for r in runs for c in r.get("calls", [])]
    total = sum(c["total_tokens"] for c in calls)

    per_minute: dict = {}
    for c in calls:
        per_minute[c["time"][:16]] = per_minute.get(c["time"][:16], 0) + c["total_tokens"]

    times = sorted((datetime.fromisoformat(c["time"]), c["total_tokens"]) for c in calls)
    peak, peak_at = 0, None
    for t0, _ in times:
        n = sum(k for t, k in times if t0 <= t < t0 + timedelta(seconds=60))
        if n > peak:
            peak, peak_at = n, t0

    limits = next((r["limits"] for r in reversed(runs) if r.get("limits")), {})

    md = ["## Groq token usage", "",
          "| # | Water body | Cycle | Result | Calls | Rejected | Tokens in | Tokens out | Total |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        cs = r.get("calls", [])
        md.append(f"| {r.get('location_id') or '-'} | {r.get('location')} | {r.get('run_index') or '-'} | "
                  f"{'saved' if r.get('saved') else 'not saved'} | {len(cs)} | {r.get('failed_calls', 0)} | "
                  f"{sum(c['prompt_tokens'] for c in cs):,} | {sum(c['completion_tokens'] for c in cs):,} | "
                  f"{sum(c['total_tokens'] for c in cs):,} |")
    md += ["", f"**Total for {len(runs)} water bodies: {total:,} tokens** in {len(calls)} answered calls "
               f"({sum(r.get('failed_calls', 0) for r in runs)} rejected calls not measured).", "",
           f"**Busiest 60 seconds:** {peak:,} tokens"
           + (f" (from {peak_at:%H:%M:%S} UTC)" if peak_at else ""), ""]
    if limits:
        md += ["**Groq limits (latest reply):** " + ", ".join(
            f"{k.replace('x-ratelimit-', '')} = {v}" for k, v in sorted(limits.items())), ""]
    md += ["### Tokens per minute (UTC)", "", "| Minute | Tokens |", "|---|---|"]
    md += [f"| {m[:10]} {m[11:]} | {n:,} |" for m, n in sorted(per_minute.items())]

    text = "\n".join(md)
    print(text)
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
