#!/usr/bin/env python3
"""
Fetch subnet neurons' on-chain commitment strings (typically https://github.com/owner/repo).

Swarm miners publish their model repo with ``neurons/miner.py --github_url ...``.
Validators read the same metadata via chain storage (see ``subtensor.get_commitment``).

**Implementation note:** Some chain + bittensor versions fail to decode
``NeuronInfoRuntimeApi`` / ``neuron_for_uid_lite`` (``RemainingScaleBytesNotEmptyException``).
This script avoids neuron runtime APIs entirely: it reads **``SubtensorModule::Keys``**
``(netuid, uid) -> hotkey`` via ``query_subtensor``, then **``Commitments::CommitmentOf``**
via ``get_metadata`` — the same logical path validators use, without Scale-decoding
full neuron structs.

Examples::

    source miner_env/bin/activate
    python3 scripts/fetch_onchain_github_urls.py --netuid 124 --subtensor.network finney

    python3 scripts/fetch_onchain_github_urls.py --netuid 124 --uid 42
    python3 scripts/fetch_onchain_github_urls.py --netuid 124 \\
      --uids 47,10,155,48,174,50,198,138,110,149 --only-github
    python3 scripts/fetch_onchain_github_urls.py --netuid 124 --json-out /tmp/commits.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import bittensor as bt

try:
    from bittensor.extrinsics.serving import get_metadata  # bittensor < 8
except ImportError:
    get_metadata = None  # bittensor >= 8: use Subtensor.get_commitment directly

try:
    from scalecodec.utils.ss58 import ss58_encode
except ImportError:
    ss58_encode = None


def _subtensor(network: str):
    cls = bt.Subtensor if hasattr(bt, "Subtensor") else bt.subtensor
    return cls(network=network)


def _decode_commitment_metadata(metadata: Any) -> Optional[str]:
    """Same decoding as ``Subtensor.get_commitment`` (extrinsic metadata layout)."""
    if metadata is None:
        return None
    try:
        commitment = metadata["info"]["fields"][0]
        hex_data = commitment[list(commitment.keys())[0]][2:]
        return bytes.fromhex(hex_data).decode()
    except (KeyError, TypeError, ValueError, IndexError):
        return None


def _commitment_for_hotkey(sub: Any, netuid: int, hotkey: str, block: Optional[int] = None) -> str:
    if get_metadata is not None:
        raw = get_metadata(sub, netuid, hotkey, block)
        out = _decode_commitment_metadata(raw)
    else:
        out = None
    if out is None:
        raise ValueError("No on-chain commitment (or unknown metadata layout) for this hotkey")
    return out


def _commitment_for_uid(sub: Any, netuid: int, uid: int, block: Optional[int] = None) -> str:
    """Use Subtensor.get_commitment (bittensor >= 8) which takes uid directly."""
    out = sub.get_commitment(netuid, uid, block)
    if not out:
        raise ValueError("No on-chain commitment for this uid")
    return out


def _account_value_to_ss58(val: Any) -> Optional[str]:
    """Normalize ``Keys`` storage value to an ss58 hotkey string."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        # Typical ss58 account strings on finney (length varies by format)
        if len(s) >= 40:
            return s
    raw: Any = val
    if isinstance(val, dict):
        for k in ("Id", "id", "AccountId"):
            if k in val:
                raw = val[k]
                break
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return ss58_encode(bytes(raw), bt.__ss58_format__)
    if isinstance(raw, str) and raw.startswith("0x") and len(raw) >= 66:
        return ss58_encode(bytes.fromhex(raw[2:]), bt.__ss58_format__)
    if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], int):
        return ss58_encode(bytes(raw), bt.__ss58_format__)
    return str(raw) if raw is not None else None


def _hotkey_ss58_for_uid(
    sub: Any, netuid: int, uid: int, block: Optional[int] = None
) -> Optional[str]:
    """Resolve hotkey from ``SubtensorModule::Keys`` — no NeuronInfo decode."""
    try:
        result = sub.query_subtensor("Keys", block, [netuid, uid])
    except Exception:
        return None
    val = getattr(result, "value", None)
    return _account_value_to_ss58(val)


def _parse_uid_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _subnet_neuron_count(sub: Any, netuid: int, block: Optional[int] = None) -> int:
    n = sub.subnetwork_n(netuid, block=block)
    if n is not None:
        return int(n)
    # Fallback: full metagraph (slower; avoids bulk lite API when subnetwork_n missing)
    mg = sub.metagraph(netuid=netuid, lite=False, block=block)
    return int(mg.n.item()) if hasattr(mg.n, "item") else int(mg.n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Print on-chain commitment per UID (GitHub URL for Swarm miners)."
    )
    p.add_argument("--netuid", type=int, default=124, help="Subnet netuid (Swarm default: 124)")
    p.add_argument(
        "--subtensor.network",
        type=str,
        default="finney",
        dest="network",
        help="Chain endpoint name (finney, test, local, …)",
    )
    p.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Only query this single UID (default: all UIDs 0..N-1)",
    )
    p.add_argument(
        "--uids",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated UIDs to query (e.g. from leaderboard). Do not use with --uid.",
    )
    p.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        metavar="SEC",
        help="Pause between UIDs to reduce public RPC load (e.g. 0.2)",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Write JSON array of {uid, hotkey, commitment, error} to this path",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress table; use with --json-out")
    p.add_argument(
        "--only-github",
        action="store_true",
        help="Only print rows whose commitment looks like https://github.com/…",
    )
    args = p.parse_args(argv)

    if args.uid is not None and args.uids is not None:
        print("Use either --uid or --uids, not both.", file=sys.stderr)
        return 2

    try:
        sub = _subtensor(args.network)
    except Exception as e:
        print(f"Failed to connect subtensor ({args.network}): {e}", file=sys.stderr)
        return 1

    try:
        n = _subnet_neuron_count(sub, args.netuid)
    except Exception as e:
        print(
            f"Failed to get neuron count for netuid={args.netuid}: {e}\n"
            "Try upgrading bittensor or use a local subtensor node.",
            file=sys.stderr,
        )
        return 1

    if args.uids is not None:
        uids = _parse_uid_list(args.uids)
    elif args.uid is not None:
        uids = [args.uid]
    else:
        uids = list(range(n))

    rows: list[dict[str, Any]] = []
    for i, uid in enumerate(uids):
        if i and args.sleep_sec > 0:
            time.sleep(args.sleep_sec)
        if uid < 0 or uid >= n:
            rows.append(
                {
                    "uid": uid,
                    "hotkey": None,
                    "commitment": None,
                    "error": f"uid out of range (subnetwork_n={n})",
                }
            )
            continue
        # bittensor >= 8 has get_commitment(netuid, uid) — no hotkey lookup needed
        use_direct = get_metadata is None and hasattr(sub, "get_commitment")
        hotkey = None
        if not use_direct:
            hotkey = _hotkey_ss58_for_uid(sub, args.netuid, uid, None)
            if not hotkey:
                rows.append(
                    {
                        "uid": uid,
                        "hotkey": None,
                        "commitment": None,
                        "error": "no hotkey in Keys storage (empty slot or query failed)",
                    }
                )
                continue
        try:
            if use_direct:
                data = _commitment_for_uid(sub, args.netuid, uid, None)
            else:
                data = _commitment_for_hotkey(sub, args.netuid, hotkey, None)
            rows.append({"uid": uid, "hotkey": hotkey, "commitment": data, "error": None})
        except Exception as e:
            rows.append(
                {
                    "uid": uid,
                    "hotkey": hotkey,
                    "commitment": None,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    to_print = (
        [
            r
            for r in rows
            if r.get("commitment")
            and str(r["commitment"]).startswith("https://github.com/")
        ]
        if args.only_github
        else rows
    )

    if not args.quiet:
        print(f"netuid={args.netuid} network={args.network} subnetwork_n={n}")
        print("uid\thotkey\tcommitment_or_error")
        for r in to_print:
            hk = (r["hotkey"] or "")[:16] + "…" if r["hotkey"] and len(r["hotkey"]) > 20 else r["hotkey"]
            if r["commitment"]:
                print(f"{r['uid']}\t{hk}\t{r['commitment']}")
            else:
                print(f"{r['uid']}\t{hk}\t<{r['error']}>")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        if not args.quiet:
            print(f"Wrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
