"""Section 2.8: treatment terms h on the observed scores, Emax and linear forms."""
import numpy as np
import torch

import _setup                                    # noqa: F401
from pdx.engine.model import Layout, ModelSpec, treat_effect

for form in ("emax", "linear"):
    spec = ModelSpec(d=3, support=np.zeros((3, 3)), treat_mask=[True, True, False], treatment=form)
    lay = Layout(spec, "null")
    x = torch.zeros(lay.size)
    x[lay.slices["kappa"][0]] = torch.tensor([0.5, 0.4, 0.6, 0.2])        # (G, nt): TD then PIGD
    x[lay.slices["kappa_ldr"][0]] = torch.tensor([0.25, 0.2, 0.3, 0.1])
    dose = torch.tensor([[0.0, 1.0, 3.0, 6.0, 12.0]])                    # one PIGD participant, 100 mg LEDD
    on = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0]])                       # 1 = examined in the ON state
    expo = torch.tensor([[0.0, 0.5, 2.0, 4.0, 8.0]])
    h = treat_effect(x, lay, dose, expo, on, torch.tensor([1]))
    print(f"{form}: kappa[PIGD] = {lay.get(x, 'kappa')[1].numpy()}, kappa_ldr[PIGD] = {lay.get(x, 'kappa_ldr')[1].numpy()}")
    print(" visit  dose  on  expo   h (domains 0, 1, 2)")
    for k in range(5):
        print(f"  {k}   {dose[0, k]:5.1f} {on[0, k]:3.0f} {expo[0, k]:5.1f}  ", h[0, k].numpy().round(4))
fd, fe = 3.0 / (3.0 + 3.0), 2.0 / (3.0 + 2.0)
print(f"by hand, Emax, visit 2, domain 0: -(0.6 * {fd:.3f} * 1 + 0.3 * {fe:.3f}) = {-(0.6 * fd + 0.3 * fe):.4f}")
