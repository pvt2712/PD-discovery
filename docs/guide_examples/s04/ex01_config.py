"""Section 4.2: the three Step 1 configurations the code uses. No data."""
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from pdx.step1 import Step1Config  # noqa: E402

print("default (= fast):     ", Step1Config())
with open(ROOT / "configs" / "protocol_v0.5.toml", "rb") as f:
    s1 = tomllib.load(f)["step1"]
fast = Step1Config(alpha=s1["alpha"], depth=s1["depth_fast"], test=s1["test_fast"])
print("run_hp_real.py:       ", fast)
print("reported_map default: ", Step1Config(test="cluster_fisherz", depth=4))
print("type of alpha read from the TOML file:", type(s1["alpha"]).__name__)
