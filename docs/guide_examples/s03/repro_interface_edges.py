"""Section 3.N, note 9: three small interface edges of fit.py, each raising before any fitting.
Synthetic data only (scenario S1, 4 domains)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]

from helpers import small_panel, spec_for
from pdx.engine.fit import fit_model, fit_pair, null_to, start_values

panel, info = small_panel("S1", n=80, seed=1, d=4)
spec = spec_for(panel, info)
x_null = start_values(panel, spec, "null")
cases = [("fit_model with no start", lambda: fit_model(panel, spec, "null", use_default_start=False)),
         ("fit_pair with x0 in **kw", lambda: fit_pair(panel, spec, x0=x_null)),
         ("null_to(..., 'null')", lambda: null_to(x_null, spec, "null"))]
for label, call in cases:
    try:
        call()
        print(f"{label}: no error")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
