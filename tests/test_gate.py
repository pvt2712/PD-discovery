"""The engine gate's checkpoint: finished work is kept, and a run with other settings is refused."""
import tempfile
from pathlib import Path

import numpy as np

from pdx import gate


def test_checkpoint_keeps_finished_sections_and_refuses_other_settings():
    path = Path(tempfile.mkdtemp()) / "gate.partial.json"
    settings = {"n": 10, "sim_kw": {"target_counts": (3, 2), "visit_probs": {2: 1}}}
    calls = []

    def work():
        calls.append(1)
        return {"x": np.float64(1.5), "v": np.arange(3), "ok": np.bool_(True)}

    said = []
    ck = gate.Checkpoint(path, settings, said.append)
    assert ck.section("a", work) == {"x": 1.5, "v": [0, 1, 2], "ok": True}
    again = gate.Checkpoint(path, settings, said.append)          # a restart with the same settings
    assert again.section("a", work) == {"x": 1.5, "v": [0, 1, 2], "ok": True} and len(calls) == 1
    assert said[0].startswith("a: done (") and said[1] == "a: done earlier, skipped"
    try:
        gate.Checkpoint(path, {**settings, "n": 11})
        raise AssertionError("other settings were accepted")
    except RuntimeError:
        pass
    assert gate.Checkpoint(None, settings).section("b", lambda: 2) == 2   # no path: nothing saved


def test_dispersed_starts_resumed_midway_give_the_same_fits():
    from pdx.qualification import spec_for
    from pdx.sim import scenarios
    panel, info = scenarios.generate("S1", 80, np.random.default_rng(6), d=4)
    spec = spec_for(panel, info)
    seen = []
    full = gate.check_dispersed_starts(panel, spec, np.random.default_rng(1), n_starts=1,
                                       on_fit=lambda done: seen.append({k: list(v) for k, v in done.items()}))
    resumed = gate.check_dispersed_starts(panel, spec, np.random.default_rng(1), n_starts=1, done=seen[1])
    assert len(seen) == 4 and seen[1] == {"null": full["null"]["logliks"]} and resumed == full


def test_quadrature_check_reports_the_statistic_at_each_node_count():
    from pdx.engine.fit import fit_model
    from pdx.qualification import spec_for
    from pdx.sim import scenarios
    panel, info = scenarios.generate("S1", 120, np.random.default_rng(5), d=4)
    f0 = fit_model(panel, spec_for(panel, info), "null")
    q = gate.check_quadrature_lr(f0, panel)
    assert set(q["lr"]) == {"12", "20", "30"} and min(q["lr"].values()) > -1e-6
    assert q["lr_spread"] == max(q["lr"].values()) - min(q["lr"].values()) and "pass" in q
