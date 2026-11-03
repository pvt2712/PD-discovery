from pdx import qualification as Q


def test_configuration_keys_and_seeds():
    a = Q.make_tasks("S1", 5, 1400, "A", d=4, step1=True, fit_kw={"maxiter": 3000})
    b = Q.make_tasks("S1", 5, 1400, "A", d=4, step1=False, fit_kw={"maxiter": 50})
    c = Q.make_tasks("S2", 5, 1400, "A", d=4, step1=True, fit_kw={"maxiter": 3000})
    assert Q.config_key(a[0]) != Q.config_key(b[0])            # different settings never pool
    assert Q.config_key(a[0]) == Q.config_key(c[0])            # a power scenario finds its null reference
    seeds = [t["seed"] for t in a + c] + [t["seed"] for t in Q.make_tasks("S1", 5, 1400, "B")]
    assert len(set(seeds)) == len(seeds)


def test_failed_replicates_enter_the_pool_as_infinite():
    assert Q._rank_quantile([1.0, 2.0, None, 3.0], 0.5) == 2.0
    assert Q._rank_quantile([1.0, None], 0.95) == float("inf")
    rows = [{"family": "A", "scenario": "S1", "config": "x", "lr": 5.0, "lr_star": v, "df": 6, "converged": True,
             "replicate_ok": v is not None, "replicate_converged": True, "seconds": 1.0}
            for v in [1.0] * 18 + [None, None]]
    cal = Q.calibration_summary(rows)
    v = cal["A:S1:x"]
    assert v["critical_value"] == float("inf") and v["rejection_rate"] == 0.0 and v["replicates_failed"] == 2
