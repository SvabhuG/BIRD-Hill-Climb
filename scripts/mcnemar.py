"""McNemar test + 95% CI on the EX-delta between two paired prediction sets.

Usage:
    python scripts/mcnemar.py <ref.json> <new.json>

Each input is the evaluator's saved JSON ({"results": [{question_id, status, ...}]}).
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path


def _load_correct_map(path: Path) -> dict[int, bool]:
    payload = json.loads(path.read_text())
    out: dict[int, bool] = {}
    for r in payload["results"]:
        qid = int(r["question_id"])
        out[qid] = r["status"] == "correct"
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: mcnemar.py <ref.json> <new.json>", file=sys.stderr)
        sys.exit(2)
    ref = _load_correct_map(Path(sys.argv[1]))
    new = _load_correct_map(Path(sys.argv[2]))
    overlap = ref.keys() & new.keys()
    print(f"n_ref={len(ref)} n_new={len(new)} n_overlap={len(overlap)}")

    b11 = b10 = b01 = b00 = 0
    for q in overlap:
        r = ref[q]
        n = new[q]
        if r and n:
            b11 += 1
        elif r and not n:
            b10 += 1
        elif not r and n:
            b01 += 1
        else:
            b00 += 1

    n = len(overlap)
    ref_ex = (b11 + b10) / n
    new_ex = (b11 + b01) / n
    delta = new_ex - ref_ex

    # McNemar with continuity correction.
    if b10 + b01 == 0:
        chi2 = 0.0
    else:
        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    # Two-sided p-value approx via chi2 with 1 dof
    # p = erfc(sqrt(chi2/2)) for chi2 with 1 dof
    p = math.erfc(math.sqrt(chi2 / 2)) if chi2 > 0 else 1.0

    # 95% CI on (b01-b10)/n via normal approx
    var = (b01 + b10) / (n * n)  # variance of (b01-b10)/n under the difference
    ci_halfwidth_pp = 1.96 * math.sqrt(var) * 100

    print(f"\nContingency (new=rows, ref=cols):")
    print(f"  b11 both correct   = {b11}")
    print(f"  b01 new only       = {b01}  (new fixed ref-wrong)")
    print(f"  b10 ref only       = {b10}  (new broke ref-correct)")
    print(f"  b00 both wrong     = {b00}")
    print(f"  net new vs ref     = {b01 - b10:+d}")
    print(f"\n  ref EX  = {ref_ex:.4f}")
    print(f"  new EX  = {new_ex:.4f}")
    print(f"  delta   = {delta * 100:+.2f}pp")
    print(f"  chi2    = {chi2:.3f}")
    print(f"  p (2-sided, McNemar cc) = {p:.4g}")
    print(f"  95% CI half-width (pp)  = {ci_halfwidth_pp:.2f}")
    print(f"  CI on delta = [{(delta * 100) - ci_halfwidth_pp:+.2f}, "
          f"{(delta * 100) + ci_halfwidth_pp:+.2f}]pp")


if __name__ == "__main__":
    main()
