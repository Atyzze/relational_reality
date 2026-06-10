#!/usr/bin/env python3
"""
reverify — re-judge equilibration verdicts from STORED traces, no recompute
===========================================================================

Every cell's meta sidecar persists its thermalisation trace
(sweep, k_avg, Σd²). When the verification statistics improve (as they did
when the fixed-τ window test was replaced by the τ-aware drift-slope +
impact test in core.graph_builder._drift_verdict), the verdicts already on
disk can be recomputed from those traces in seconds instead of re-running
days of builds. This tool does exactly that:

  • therm_verified is rewritten as True / False / None (unresolved),
  • the previous verdict is preserved as therm_verified_v1,
  • the test's evidence (τ, z, impact per observable) is stored under
    therm_verify_stats, and therm_verify_method records which trace
    region the verdict came from:
        "trace-post"  — post-phase-1 verification samples (new-style metas)
        "trace-tail"  — tail-of-phase-1 + final post-production sample
                        (pre-upgrade metas; verdicts are mostly None unless
                        the move across the production window is flagrant)

Usage
-----
    python main.py reverify              # rewrites verdicts under ./output
    python main.py reverify --dir DIR    # another output directory
    python main.py reverify --dry-run    # show the table, write nothing

Downstream gates (flow_modes, flow_convergence) reject only explicit
False, so a None verdict keeps the cell usable while recording that the
trace could not resolve its slowest mode within the stored span.
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

from core.graph_builder import _drift_verdict


def _series(meta):
    tr = meta.get("therm_trace")
    if not tr or not tr.get("rows"):
        return None
    rows = sorted(tr["rows"])
    N = float(meta.get("N") or 0)
    if N <= 0:
        return None
    sw = np.array([r[0] for r in rows], dtype=float)
    k = np.array([r[1] for r in rows], dtype=float)
    q = np.array([r[2] for r in rows], dtype=float) / N
    return sw, k, q


def reverify_meta(meta):
    """Return (verdict, stats, method) or None when no trace exists."""
    ser = _series(meta)
    if ser is None:
        return None
    sw, k, q = ser
    s0 = float(meta.get("therm_sweeps") or 0)
    post = sw > s0 + 0.5
    if post.sum() >= 8:
        method = "trace-post"
        swp, kp, qp = sw[post], k[post], q[post]
        # impact horizon = the cell's own thermalisation scale ("as long
        # again"), matching the live verification's semantics
        horizon = max(s0, float(swp[-1] - swp[0]), 400.0)
        vk, sk = _drift_verdict(swp, kp, horizon, abs_floor=5e-4)
        vq, sq = _drift_verdict(swp, qp, horizon,
                                abs_floor=0.005 * max(abs(np.median(qp)), 1e-12))
        if vk is False or vq is False:
            v = False
        elif vk is None or vq is None:
            v = None
        else:
            v = True
        stats = {"k": sk, "q": sq}
        if v is False and post.sum() >= 16:
            # Mirror the live protocol: when the full stretch flags drift,
            # the LATER half alone gets the final word (drift may have been
            # the decaying tail of relaxation; this also suppresses the
            # slow-τ chance-slope false-fail quadratically).
            h = post.sum() // 2
            swh, kh, qh = swp[h:], kp[h:], qp[h:]
            hor2 = max(s0, float(swh[-1] - swh[0]), 400.0)
            vk2, sk2 = _drift_verdict(swh, kh, hor2, abs_floor=5e-4)
            vq2, sq2 = _drift_verdict(swh, qh, hor2,
                                      abs_floor=0.005 * max(abs(np.median(qh)), 1e-12))
            v = False if (vk2 is False or vq2 is False) else \
                (None if (vk2 is None or vq2 is None) else True)
            stats = {"k": sk2, "q": sq2, "full_stretch": {"k": sk, "q": sq}}
        return v, stats, method
    # Pre-upgrade metas: sparse phase-1 samples + ONE final post-production
    # point. Too little for a slope test — verdict None unless the jump
    # across the production window is flagrant relative to the late-phase-1
    # fluctuation band (and to a 2% relative floor).
    method = "trace-tail"
    tail = sw <= s0 + 0.5
    if tail.sum() < 4 or post.sum() < 1:
        return None, {"reason": "trace too sparse"}, method
    q_tail = q[tail][-6:]
    q_fin = float(q[post][-1])
    sig = float(np.std(q_tail, ddof=1)) if q_tail.size > 1 else 0.0
    jump = abs(q_fin - float(np.mean(q_tail)))
    floor = max(3.0 * sig, 0.02 * abs(np.mean(q_tail)))
    stats = {"q": {"jump": round(jump, 4), "floor": round(floor, 4),
                   "n_tail": int(q_tail.size)}}
    return (False if jump > floor else None), stats, method


def main(argv=None):
    ap = argparse.ArgumentParser(prog="main.py reverify",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", default="output",
                    help="output directory holding flow/meta_*.json "
                         "(default: ./output; '.' when already inside it)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the verdict table, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="also re-judge metas whose verdict came from the "
                         "current live test (therm_verify_method=live-v2); "
                         "by default those are authoritative and skipped")
    args = ap.parse_args(argv)

    pats = [os.path.join(args.dir, "flow", "meta_*.json"),
            os.path.join(args.dir, "meta_*.json")]
    paths = sorted({p for pat in pats for p in glob.glob(pat)})
    if not paths:
        print(f"no meta sidecars under {args.dir}", file=sys.stderr)
        return 1

    agg = defaultdict(lambda: defaultdict(int))
    n_seen = n_traceless = n_changed = 0
    for p in paths:
        try:
            with open(p) as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        if meta.get("kind") not in ("cell", "torus"):
            continue
        n_seen += 1
        if meta.get("therm_verify_method") == "live-v2" and not args.force:
            # already judged by the current live test at full per-sweep
            # resolution — the decimated-trace re-judgement here is an
            # approximation of that and must not overrule it
            agg[(meta.get("T", "-"), meta.get("lb", "-"),
                 meta.get("N", "-"))]["live-v2 (kept)"] += 1
            continue
        res = reverify_meta(meta)
        if res is None:
            n_traceless += 1
            continue
        v, stats, method = res
        old = meta.get("therm_verified", "absent")
        key = (meta.get("T", "-"), meta.get("lb", "-"), meta.get("N", "-"))
        agg[key][{True: "pass", False: "drift", None: "unresolved"}[v]] += 1
        if old != "absent" and (old is not v):
            n_changed += 1
        if args.dry_run:
            continue
        if "therm_verified_v1" not in meta and old != "absent":
            meta["therm_verified_v1"] = old
        meta["therm_verified"] = v
        meta["therm_verify_stats"] = stats
        meta["therm_verify_method"] = method
        tmp = f"{p}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w") as fh:
                json.dump(meta, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    print(f"{n_seen} cell/torus metas · {n_traceless} without a usable trace"
          f" · {n_changed} verdicts changed"
          + ("  [dry run — nothing written]" if args.dry_run else ""))
    print(f"{'T':<8}{'lb':<8}{'N':>9}   {'pass':>5} {'drift':>6} "
          f"{'unresolved':>11} {'live-v2':>8}")
    for (T, lb, N) in sorted(agg, key=lambda x: (str(x[0]), str(x[1]),
                                                 str(x[2]))):
        a = agg[(T, lb, N)]
        print(f"{T!s:<8}{lb!s:<8}{N!s:>9}   {a['pass']:>5} {a['drift']:>6} "
              f"{a['unresolved']:>11} {a['live-v2 (kept)']:>8}")
    if not args.dry_run:
        print("\nverdicts rewritten in place — regenerate the figures "
              "(`python main.py figures`) to see the gates with the new "
              "verdicts.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(
        os.path.abspath(__file__))))
    main()
