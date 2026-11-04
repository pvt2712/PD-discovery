"""Sections 1.10-1.13: the module constants of ppmi.py that encode rules. No data."""
import re

import numpy as np
from _setup import ROOT

from pdx import ppmi

print("== 1.10 ==")
print("RAW = <folder holding the repository>/" + str(ppmi.RAW.relative_to(ROOT.parent)))
print("NOT_GRID:", ppmi.NOT_GRID)
for code in ("SC", "U01", "U02", "R08", "V04", "BL"):
    print(f"  {code}: excluded from the grid = {code in ppmi.NOT_GRID}")

print("\n== 1.11 domain item lists and the 80% rule ==")
for name, items in [("TREMOR_III", ppmi.TREMOR_III), ("BRADY_RIGID", ppmi.BRADY_RIGID), ("AXIAL", ppmi.AXIAL),
                    ("FUNCTION_II", ppmi.FUNCTION_II)]:
    print(f"  {name:12s} {len(items):2d} items, needs {int(np.ceil(0.8 * len(items))):2d}, range 0-{4 * len(items)}")
allitems = list(dict.fromkeys(ppmi.TREMOR_III + ppmi.BRADY_RIGID + ppmi.AXIAL))
print("  Part III items in the three motor domains:", len(allitems), "| shared items:",
      len(ppmi.TREMOR_III) + len(ppmi.BRADY_RIGID) + len(ppmi.AXIAL) - len(allitems),
      "| an examination needs >=", 0.8 * len(allitems))
print("  ceil(0.8 n) equals the exact integer ceiling for n = 1..199:",
      all(int(np.ceil(0.8 * n)) == -(-4 * n // 5) for n in range(1, 200)))

print("\n== 1.12 Stebbins items ==")
print("  tremor:", len(ppmi.STEBBINS_TREMOR), "items; PIGD:", len(ppmi.STEBBINS_PIGD), "items; cut-offs",
      ppmi.TD_CUT, ppmi.PIGD_CUT)
print("  PIGD items that are also axial-domain items:", sorted(set(ppmi.STEBBINS_PIGD) & set(ppmi.AXIAL)))

print("\n== 1.13 questionnaires, flags, the levodopa pattern ==")
print("  GDS: yes-items", len(ppmi.GDS_YES), "no-items", len(ppmi.GDS_NO), "| RBDSQ items", len(ppmi.RBDSQ_ITEMS),
      "+ question 10 from", len(ppmi.RBDSQ_NEURO), "flags | genetic flags", len(ppmi.GENETIC_FLAGS))
print("  DOMAINS:", len(ppmi.DOMAINS), ppmi.DOMAINS)
names = ["Sinemet CR", "Carbidopa/Levodopa", "Levadopa", "levodopa/benserazide", "Madopar HBS", "Stalevo", "Rytary",
         "Duodopa", "Inbrija", "Carbidopa", "L-Dopa", "Duopa", "Co-careldopa", "Co-beneldopa", "Dhivy",
         "Entacapone", "Pramipexole", "Rasagiline", "Amantadine"]
for nm in names:
    print(f"  {nm:22s} levodopa = {bool(re.search(ppmi.LEVODOPA, nm.lower()))}")
