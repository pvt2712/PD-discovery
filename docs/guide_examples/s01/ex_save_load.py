"""Section 1.6: save_panel and load_panel (flat arrays plus offsets, no pickling).
Made-up participants; the file goes to a temporary folder."""
import numpy as np
from _setup import TMP

from pdx.data import Panel, load_panel, save_panel

p = Panel(y=[np.array([[0.1, 0.2], [0.3, np.nan]]), np.array([[1.0, 1.1]]), np.array([[2.0, 2.1], [2.2, 2.3], [2.4, 2.5]])],
          t=[[1.0, 2.0], [1.0], [0.9, 2.0, 3.1]], group=[0, 1, 0], z=[[0.1], [-0.2], [0.1]],
          dose=[[0, 1], [2], [0, 0, 3]], ids=np.array([11, 12, 13]), domains=["tremor", "axial"],
          meta={"source": "synthetic"}, year=[[1, 2], [1], [1, 2, 3]])
path = TMP / "panel.npz"
save_panel(p, path)
with np.load(path) as f:
    print("stored arrays:", {k: f[k].shape for k in f.files})
    print("lens:", f["lens"], "-> cut points:", np.cumsum(f["lens"])[:-1])
q = load_panel(path)
same = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(p.y, q.y))
print("y identical:", same, "| t identical:", all(np.array_equal(a, b) for a, b in zip(p.t, q.t)))
print("dose[2]:", q.dose[2], "| year[2]:", q.year[2], "| domains:", q.domains)
print("ids after loading:", q.ids, "| meta after loading:", q.meta)
