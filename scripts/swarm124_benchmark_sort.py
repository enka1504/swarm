#!/usr/bin/env python3
"""
Fetch Swarm124 public benchmark table JSON and print rows sorted by **screening** (ascending).

The website https://swarm124.com/benchmark loads data from a JSON API (see browser
**Network** tab → XHR/fetch). This script does not hard-code a private URL (it may
change). Pass the JSON endpoint you see, or a saved JSON file.

Examples::

    # After copying the Request URL from DevTools (example shape only — replace with yours):
    python3 scripts/swarm124_benchmark_sort.py \\
      --fetch-url 'https://swarm124.com/api/...'

    python3 scripts/swarm124_benchmark_sort.py --json-file ./downloads/benchmark.json

    # Prefer screening column named like the site (screening_score, screening, etc.)
    python3 scripts/swarm124_benchmark_sort.py --json-file data.json --screening-key screening
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen


def _load_json_from_url(url: str, timeout: float) -> Any:
    req = Request(url, headers={"User-Agent": "swarm124_benchmark_sort/1.0"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ("rows", "items", "models", "data", "results", "leaderboard"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        return [payload]
    return []


def _find_screening_key(rows: Iterable[dict[str, Any]], hint: str | None) -> str | None:
    if hint:
        return hint if any(hint in r for r in rows) else None
    candidates = (
        "screening",
        "screening_score",
        "screeningScore",
        "screening_pass_score",
        "screening_result",
    )
    sample = list(rows)[:50]
    for c in candidates:
        if any(c in r for r in sample):
            return c
    for r in sample:
        for k in r:
            if "screen" in k.lower():
                return k
    return None


def _to_float(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            pass
    return float("nan")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sort Swarm124 benchmark rows by screening (asc).")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch-url", type=str, help="HTTPS URL returning benchmark JSON")
    g.add_argument("--json-file", type=Path, help="Saved JSON (same shape as API)")
    p.add_argument("--screening-key", type=str, default=None, help="Field name for screening score")
    p.add_argument("--timeout", type=float, default=60.0)
    args = p.parse_args(argv)

    try:
        payload = (
            _load_json_from_url(args.fetch_url, args.timeout)
            if args.fetch_url
            else _load_json_file(args.json_file)
        )
    except Exception as e:
        print(f"Failed to load JSON: {e}", file=sys.stderr)
        return 1

    rows = _iter_rows(payload)
    if not rows:
        print("No row objects found. Expected a list or dict with rows/items/models/data.", file=sys.stderr)
        return 1

    sk = _find_screening_key(rows, args.screening_key)
    if not sk:
        print(
            "Could not guess screening field. Keys seen (first row): "
            + ", ".join(sorted(rows[0].keys())),
            file=sys.stderr,
        )
        return 1

    def sort_key(r: dict[str, Any]) -> tuple[float, str]:
        v = _to_float(r.get(sk))
        label = str(r.get("model", r.get("name", r.get("uid", ""))))
        return (v, label)

    sorted_rows = sorted(rows, key=sort_key)

    print(f"Sorted by {sk!r} ascending ({len(sorted_rows)} rows)\n")
    print("rank\t" + sk + "\tuid/model\tgithub_url?")
    for i, r in enumerate(sorted_rows, start=1):
        gh = r.get("github_url") or r.get("github") or r.get("repo_url") or ""
        uid = r.get("uid", r.get("model_uid", ""))
        name = r.get("model", r.get("name", ""))
        ident = f"{uid} {name}".strip() or json.dumps(r, default=str)[:80]
        print(f"{i}\t{r.get(sk)}\t{ident}\t{gh}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
