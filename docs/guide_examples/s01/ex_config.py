"""Section 1.1: config.DEFAULT and config.load. No data at all."""
from _setup import ROOT, TMP

from pdx import config

print("DEFAULT =", config.DEFAULT.relative_to(ROOT))
cfg = config.load()
print("tables:", list(cfg))
print("engine:", cfg["engine"])
print("hp alpha, n_max, h:", cfg["hp"]["alpha"], cfg["hp"]["n_max"], cfg["hp"]["h"])
print("types: eps", type(cfg["localisation"]["eps"]).__name__, "| n_quad", type(cfg["engine"]["n_quad"]).__name__)

# another file can be passed explicitly (for example an amended copy)
alt = TMP / "amended.toml"
alt.write_text('[engine]\nn_quad = 20\ntreatment = "linear"\n')
print("amended copy:", config.load(alt))
try:
    config.load(TMP / "missing.toml")
except FileNotFoundError as e:
    print("missing file ->", type(e).__name__)
