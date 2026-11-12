"""Plan v16 step 33: translate each arm's earliest recovery time into the
response classes it makes observable -- reporting ONLY boundaries for which a
source is held.

Source held: Wagenaar, DeMarse, Pine & Potter (2002), Proceedings of
Mathematics in Molecular Biology (Santa Fe) -- a CONFERENCE ABSTRACT, not a
peer-reviewed paper: direct responses at 1.5-20 ms (jitter 0.05-0.15 ms),
synaptic responses from ~5 ms (jitter 1-5 ms). No sub-millisecond window is
reported: none is sourced (the earliest recovery time is given in ms alone).

    python runs/aim2_latency_classes.py [results/runs/aim2_budget/summary.json]
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "runs" / "aim2_budget" / "summary.json"
SOURCE = "Wagenaar et al. 2002, conference abstract"
CLASSES = [("direct, precisely timed responses (1.5-20 ms)", 1.5, 20.0),
           ("synaptic responses (from ~5 ms)", 5.0, None)]
s = json.loads(path.read_text())
conds = s.get("by_condition") or s.get("by_range")
out = {"source": SOURCE, "classes": [{"name": c[0], "from_ms": c[1], "to_ms": c[2]} for c in CLASSES],
       "by_condition": {}}
for cond, d in conds.items():
    print(f"\n{cond}")
    out["by_condition"][cond] = {}
    for m, byb in d["methods"].items():
        budgets = [k for k in byb if k.isdigit()]
        e = byb[max(budgets, key=int)]["earliest_median_ms"]
        if e is None:
            vis = "no delay bin up to 2.5 ms meets the criteria"
        else:
            whole = [c[0] for c in CLASSES if c[1] >= e]
            vis = (f"recovers from {e} ms; covers the whole of: " + "; ".join(whole)) if whole else f"recovers from {e} ms"
        out["by_condition"][cond][m] = {"earliest_ms": e, "observable": vis}
        print(f"  {m:34s} {vis}")
print(f"\nwindows from {SOURCE}; no sub-millisecond class is reported (no source).")
(path.parent / "latency_classes.json").write_text(json.dumps(out, indent=2))
