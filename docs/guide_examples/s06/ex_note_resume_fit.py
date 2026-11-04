"""Guide §6.N, note on resuming: after a restart run_gate refits the speed-only model from the saved
parameters (gate.py:351-353). The refit is not bit-identical to the uninterrupted fit (synthetic only).

    OMP_NUM_THREADS=1 ../ppmi-causal/.venv/bin/python docs/guide_examples/s06/ex_note_resume_fit.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from pdx.engine.fit import fit_model  # noqa: E402
from pdx.qualification import spec_for  # noqa: E402
from pdx.sim import scenarios  # noqa: E402

panel, info = scenarios.generate("S1", 120, np.random.default_rng(5), d=4)
spec = spec_for(panel, info)
f0 = fit_model(panel, spec, "null")                               # the uninterrupted run
saved = json.loads(json.dumps(f0.x.tolist()))                     # what the checkpoint keeps (null_fit_x)
f0r = fit_model(panel, spec, "null", x0=np.array(saved))           # the resumed run
print(f"uninterrupted: loglik {f0.loglik:.10f} after {f0.n_eval} evaluations ({f0.message})")
print(f"resumed:       loglik {f0r.loglik:.10f} after {f0r.n_eval} evaluations ({f0r.message})")
print(f"largest parameter change {np.abs(f0r.x - f0.x).max():.1e}; log-likelihood change {f0r.loglik - f0.loglik:.1e}")
