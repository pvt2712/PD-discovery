"""Plain test runner (pytest is not installed): runs every test_* function in tests/test_*.py.

    python tests/run_tests.py            # all tests
    python tests/run_tests.py engine     # files whose name contains "engine"
    PDX_SLOW=1 python tests/run_tests.py # include the slow tests
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))


def main(argv: list[str]) -> int:
    pattern = argv[1] if len(argv) > 1 else ""
    files = sorted(p for p in HERE.glob("test_*.py") if pattern in p.stem)
    failed, passed, skipped = [], 0, 0
    for path in files:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            if getattr(fn, "slow", False) and not os.environ.get("PDX_SLOW"):
                skipped += 1
                print(f"  skip  {path.stem}.{name} (slow; set PDX_SLOW=1)")
                continue
            t0 = time.time()
            try:
                fn()
                passed += 1
                print(f"  ok    {path.stem}.{name}  ({time.time() - t0:.1f} s)")
            except Exception:
                failed.append(f"{path.stem}.{name}")
                print(f"  FAIL  {path.stem}.{name}  ({time.time() - t0:.1f} s)")
                traceback.print_exc()
    print(f"\n{passed} passed, {len(failed)} failed, {skipped} skipped")
    for f in failed:
        print("  failed:", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
