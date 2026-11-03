"""Named comparators of protocol v0.5, §8.4: a descriptive benchmark, not part of the decision.

They estimate yearly-scale differences, which are truly non-zero under speed alone, so the
benchmark measures how often their output would be read as a mechanism difference.
`REGISTRY` records each comparator's status in this repository. Those marked "needs install"
wait for the investigator's approval to install and pin their software (D23).
"""

REGISTRY = {
    "C1": {"name": "Multigroup RI-CLPM (chi-square difference, per-edge score tests)", "software": "lavaan (R)",
           "status": "needs install (D23)"},
    "C2": {"name": "Multigroup panel GVAR; IVPP global test and partial pruning",
           "software": "psychonetrics, IVPP (R)", "status": "needs install (D23)"},
    "C3": {"name": "NCT-style permutation test on temporal networks", "software": "this package",
           "status": "implemented: pdx.comparators.run.c3_nct"},
    "C4": {"name": "Naive multigroup continuous-time model (additive phenotype effect per drift entry)",
           "software": "ctsem; DynNet (R)", "status": "needs install (D23)"},
    "C5": {"name": "tsLDiffPC and tsDCI on two-slice tables", "software": "re-implementation of pyCIPHOD",
           "status": "implemented: pdx.comparators.run.c5_tsldiffpc (pyCIPHOD cross-check needs install)"},
    "C6": {"name": "DCI, lag-to-current candidate pairs", "software": "causaldag", "status": "needs install (D23)"},
    "C7": {"name": "CD-NOD with time tiers", "software": "causal-learn",
           "status": "available in the existing environment; wrapper to be written in WP3"},
    "C8": {"name": "Sparse mechanism shift", "software": "sparse_shift", "status": "needs install (D23)"},
    "C9": {"name": "Data-shared lasso; multi-VAR", "software": "glmnet; multivar (R)", "status": "needs install (D23)"},
    "C10": {"name": "Joint causal inference; J-PCMCI+ context edges", "software": "causal-learn; tigramite",
            "status": "available in the existing environment; wrapper to be written in WP3"},
    "C11": {"name": "Separate graphs read naively; v0.3 yearly link-wise interaction tests",
            "software": "this package", "status": "implemented: pdx.comparators.run.c11_naive"},
    "C12": {"name": "DYNOTEARS, one graph per phenotype", "software": "re-implementation of causalnex",
            "status": "implemented: pdx.comparators.run.c12_dynotears"},
}
