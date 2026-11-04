"""Shared setup for the Section 1 examples. Synthetic data only.

- Puts src/, tests/ and scripts/ on the import path.
- Makes a fresh temporary folder for each run (under $TMPDIR) and removes it at exit.
- Guards pdx.ppmi.read, so that no example can read a table outside that folder. Even a call
  that falls back on the default argument raw=ppmi.RAW (the real download) is refused.
- Offers the repository's PPMI-shaped mock (tests/ppmi_mock.py, 13 made-up participants) and a
  writer for small hand-made tables in the PPMI folder layout.
"""
import atexit
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "scripts")]

import numpy as np                      # noqa: E402
import pandas as pd                     # noqa: E402

from pdx import ppmi                    # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="pdx_s01_")).resolve()
atexit.register(shutil.rmtree, TMP, ignore_errors=True)
_real_read = ppmi.read


def _guarded_read(raw, folder, name, usecols=None):
    if TMP not in Path(raw).resolve().parents:
        raise RuntimeError(f"refusing to read outside the synthetic folder: {raw}")
    return _real_read(raw, folder, name, usecols)


ppmi.read = _guarded_read


class _MaskTmp:
    """Print the temporary folder as <tmp>, so that outputs do not depend on the machine."""

    def __init__(self, stream):
        self.stream = stream

    def write(self, s):
        return self.stream.write(s.replace(str(TMP), "<tmp>"))

    def flush(self):
        self.stream.flush()


sys.stdout = _MaskTmp(sys.stdout)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
np.set_printoptions(precision=4, suppress=True, linewidth=120)


def mock_download(name: str = "mock") -> Path:
    """The repository's mock download, written to a fresh synthetic folder."""
    import ppmi_mock
    root = TMP / name
    ppmi_mock.make(root)
    return root


def write_table(root: Path, folder: str, name: str, df: pd.DataFrame) -> None:
    """One synthetic table in the PPMI layout: <root>/<folder>/<name>_01Jan2026.csv."""
    (root / folder).mkdir(parents=True, exist_ok=True)
    df.to_csv(root / folder / f"{name}_01Jan2026.csv", index=False)


def protect_script_paths():
    """Point the run scripts' data/ and results/ folders (scripts/_common.py) at the synthetic
    folder, so that a script's main() can be run without touching the repository's folders."""
    import _common
    _common.DATA = TMP / "data"
    _common.RESULTS = TMP / "results"
    return _common
