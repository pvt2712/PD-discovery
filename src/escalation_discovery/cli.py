"""escd — command line of the discovery experiment (PROTOCOL.md §10 lists the run order).

    uv run escd methods                 # which adapters can run here
    uv run escd truth                   # build and cache the Arm A truths (synthetic source)
    uv run escd arm-a --tune            # tune on synthetic data (frozen configurations)
    uv run escd arm-a                   # Arm A
    uv run escd tau                     # stability threshold for Arm B, calibrated on Arm A data
    uv run escd arm-b --data synthetic  # Arm B pipeline on a simulated cohort
    uv run escd handover --data synthetic
  Gated until the protocol is approved (PROTOCOL.md §11):
    uv run escd build | trials | audit | core-map | fit-generator | arm-a --source fitted | arm-b --data ppmi
"""
import sys


def _methods(argv):
    from .methods import registry
    ok, missing = registry.available(registry.panel())
    for e in registry.panel():
        state = "ok" if e.name not in missing else f"unavailable: {missing[e.name]}"
        scope = "R only" if e.realistic_only else "all"
        print(f"{e.name:22s} {e.method.family:11s} {e.method.output:5s} knowledge={str(e.knowledge):5s} {e.role:9s} "
              f"settings={scope:6s} {state}")


def _truth(argv):
    from . import arm_a, expert as E
    gens = arm_a.load_generators("synthetic")
    for s in arm_a.settings(E.load()):
        tr = arm_a.get_truth("synthetic", gens[s.generator], s.hidden)
        print(s.name, tr.name, "scored pairs", len(tr.scored()), "true", len([p for p in tr.mag.adjacencies() if p in tr.scored()]))


def _tau(argv):
    from . import arm_b
    print(arm_b.calibrate_tau("synthetic" if "--fitted" not in argv else "fitted")[0])


def _fit_generator(argv):
    from . import fit
    fit.main(argv)


COMMANDS = {
    "methods": _methods,
    "truth": _truth,
    "arm-a": lambda a: __import__("escalation_discovery.arm_a", fromlist=["main"]).main(a),
    "tau": _tau,
    "arm-b": lambda a: __import__("escalation_discovery.arm_b", fromlist=["main"]).main(a),
    "handover": lambda a: __import__("escalation_discovery.handover", fromlist=["main"]).main(a),
    "build": lambda a: __import__("escalation_discovery.build", fromlist=["main"]).main(a),
    "trials": lambda a: __import__("escalation_discovery.trials", fromlist=["main"]).main(a),
    "audit": lambda a: __import__("escalation_discovery.audit", fromlist=["main"]).main(a),
    "core-map": lambda a: __import__("escalation_discovery.falsify", fromlist=["core_map_main"]).core_map_main(a),
    "fit-generator": _fit_generator,
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] not in COMMANDS:
        print(__doc__)
        return 1
    return COMMANDS[args[0]](args[1:])
