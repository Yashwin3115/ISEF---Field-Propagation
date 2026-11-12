"""Step 32 / section 5.7: translate each method's earliest recovery time into
the classes of evoked response it makes observable.

Windows and their sources (VERIFY against primary sources before submission,
as the plan requires -- the first is derived here, not taken from the
literature):
  direct activation of axons near the pad, seen at nearby electrodes:
      0 - 1.5 ms. DERIVED: latency = distance / conduction velocity; the
      model's 1 um axon conducts at 1.06 m/s (tests/test_neuron.py), so an
      electrode 100-1000 um away sees the spike 0.1-1 ms after it starts.
  direct responses with precise timing: 1.5 - 20 ms (jitter 0.05-0.15 ms).
      Wagenaar, DeMarse, Pine & Potter (2002), Proc. Mathematics in
      Molecular Biology, Santa Fe -- a conference abstract; their lower
      bound 1.5 ms was set by their own artifact.
  synaptically mediated responses: from ~5 ms (jitter 1-5 ms). Same source.

    python runs/aim2_latency_classes.py [results/runs/aim2_budget/summary.json]
"""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "runs" / "aim2_budget" / "summary.json"
CLASSES = [
    ("direct activation near the pad (derived)", 0.0, 1.5, "derived: distance / conduction velocity"),
    ("direct, precisely timed responses", 1.5, 20.0, "Wagenaar et al. 2002 (abstract)"),
    ("synaptic responses", 5.0, None, "Wagenaar et al. 2002 (abstract)"),
]
s = json.loads(path.read_text())
out = {"classes": [{"name": c[0], "from_ms": c[1], "to_ms": c[2], "source": c[3]} for c in CLASSES],
       "by_condition": {}}
for cond, d in s["by_range"].items():
    print(f"\n{cond}")
    out["by_condition"][cond] = {}
    for m, byb in d["methods"].items():
        budgets = [k for k in byb if k.isdigit()]
        e = byb[max(budgets, key=int)]["earliest_median_ms"]
        if e is None:
            vis = "no delay bin up to 2.5 ms meets the criteria"
            full = []
        else:
            full = [c[0] for c in CLASSES if c[1] >= e]
            partial = [c[0] for c in CLASSES if c[1] < e < (c[2] or 1e9)]
            vis = "; ".join(full) + (f"; partly: {', '.join(partial)}" if partial else "")
        out["by_condition"][cond][m] = {"earliest_ms": e, "observable": vis}
        print(f"  {m:32s} earliest {str(e):>5s} ms -> {vis}")
(path.parent / "latency_classes.json").write_text(json.dumps(out, indent=2))
