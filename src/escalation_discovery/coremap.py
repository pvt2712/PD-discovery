# Ported from escalation-study-v2/src/escalation_v2/coremap.py (29 October 2026). Changes are marked 'escalation-discovery:'.
"""coremap — the guideline-based core causal map (docs/core_map.yaml) and the analysis blocks it defines.

    uv run esc2 map            # checks the map against a trial table and draws docs/core_map.png (and .pdf)

The map has 13 nodes: 11 measured (three guideline triggers, four guideline brakes, the dose in force, the decision
and two 12-month outcomes) and 2 latent (disease progression; the guideline-named reasons PPMI does not record).
Every arrow into the decision carries the guideline locator it was derived from. The six analysis blocks of the
confounding questions are the three guideline blocks plus three measured blocks no guideline names.
"""
import pathlib
from functools import lru_cache

import yaml

from . import config as C

MAP_FILE = C.docs_dir() / "core_map.yaml"
GRADE_STYLE = {"G1": "solid", "G2": "solid", "E": "solid", "R": "solid", "M": "dashed", "N": "dotted", "H": "dashed"}


@lru_cache(maxsize=1)
def load():
    with open(MAP_FILE) as f:
        return yaml.safe_load(f)


def nodes():
    return {n["id"]: n for n in load()["nodes"]}


def arrows():
    return load()["arrows"]


def decision_arrows():
    """The guideline-graded reasons into the decision whose cause is measured: (node, arrow) pairs the audit tests."""
    nd = nodes()
    return [(nd[a["from"]], a) for a in arrows() if a["to"] == "escalate" and nd[a["from"]]["measured"]]


def blocks():
    """The six analysis blocks, in order: id -> dict(label, columns, guideline)."""
    m, nd = load(), nodes()
    out = {}
    for bid, b in m["analysis_blocks"].items():
        cols = sum((nd[n]["columns"] for n in b["nodes"]), []) if "nodes" in b else m["off_guideline_blocks"][b["off_guideline"]]["columns"]
        out[bid] = dict(label=b["label"], columns=list(cols), guideline=bool(b["guideline"]))
    return out


def adjustment_columns():
    """Every pre-decision covariate of the adjustment set (the union of the six blocks)."""
    return list(dict.fromkeys(c for b in blocks().values() for c in b["columns"]))


def check(d):
    """Columns of the map missing from a trial table (an empty list when the table carries them all)."""
    want = adjustment_columns() + ["E", "y12_adl", "y12_cx"]
    return [c for c in want if c not in d.columns]


def _edge(c, half, toward):
    """Point on a box boundary (centre c, half-sizes) along the line to `toward`."""
    dx, dy = toward[0] - c[0], toward[1] - c[1]
    s = min(half[0] / abs(dx) if dx else float("inf"), half[1] / abs(dy) if dy else float("inf"))
    return c[0] + dx * s, c[1] + dy * s


def _wrap(label, width=20):
    if len(label) <= width:
        return label
    words, best = label.split(" "), None
    for k in range(1, len(label.split(" "))):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if best is None or max(len(a), len(b)) < max(len(best[0]), len(best[1])):
            best = (a, b)
    return best[0] + "\n" + best[1]


def _rounded(points, r=0.22):
    """A path through the points, its corners rounded with radius r (data units)."""
    import numpy as np
    from matplotlib.path import Path
    p = [np.asarray(q, float) for q in points]
    verts, codes = [p[0]], [Path.MOVETO]
    for a, b, c in zip(p, p[1:], p[2:]):
        u, v = (b - a) / np.linalg.norm(b - a), (c - b) / np.linalg.norm(c - b)
        verts += [b - r * u, b, b + r * v]
        codes += [Path.LINETO, Path.CURVE3, Path.CURVE3]
    return Path(verts + [p[-1]], codes + [Path.LINETO])


# arrows drawn as arcs (rad) so they do not pass through a box; the rest are straight
ROUTE = {("wearing_off", "motor_control"): -0.55, ("wearing_off", "complications_12m"): -0.32, ("unrecorded_reasons", "daily_function_12m"): 0.18}
# disease progression reaches the outcomes over the top of the map (complications down its right side), so the paths of
# confounding by indication pass behind no box
VIA = {("progression", "daily_function_12m"): "top", ("progression", "complications_12m"): "right"}
TOP, RIGHT = 8.2, 12.1


def draw(path=None):
    """The core map left to right: the guideline reasons in one column (triggers blue, the dose in force grey, brakes
    red), the decision in the centre, the 12-month outcomes on the right, the two latent nodes dashed. Arrow colour
    follows the role of the reason; dashed arrows leave a latent node; dotted arrows are natural-history carry-over."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    nd = nodes()
    fig, ax = plt.subplots(figsize=(12.5, 8.4))
    ax.set_xlim(-0.6, 12.45); ax.set_ylim(-0.85, 8.45); ax.axis("off")
    col = {"trigger": "#2a78d6", "brake": "#c0392b", "context": "#5b6b7b", "decision": "#0b2545", "outcome": "#0b2545", "latent": "#8a99ad"}
    size = {}
    for n in nd.values():
        lab = _wrap(n["label"])
        two = "\n" in lab
        w, h = (2.25 if n["role"] != "decision" else 2.5), (0.80 if two else 0.64)
        size[n["id"]] = (w / 2, h / 2)
        latent = not n["measured"]
        fc = "#0b2545" if n["role"] == "decision" else ("#f4f6f9" if latent else "white")
        ax.add_patch(FancyBboxPatch((n["x"] - w / 2, n["y"] - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.6,
                                    edgecolor=col[n["role"]], facecolor=fc, linestyle="--" if latent else "-", zorder=3))
        sub = {"trigger": "trigger: pushes the dose up", "brake": "brake: holds the dose back", "context": "context", "decision": "≥ +100 mg/day within 3 months",
               "outcome": "outcome", "latent": "latent: not recorded"}[n["role"]]
        ax.text(n["x"], n["y"] + (0.10 if not two else 0.13), lab, ha="center", va="center", fontsize=9.3, fontweight="bold", zorder=4,
                color="white" if n["role"] == "decision" else "#0b2545", style="italic" if latent else "normal", linespacing=1.05)
        ax.text(n["x"], n["y"] - (0.17 if not two else 0.27), sub, ha="center", va="center", fontsize=7.3, zorder=4,
                color="#c9d6e6" if n["role"] == "decision" else "#5b6b7b")
    for a in arrows():
        src, dst = nd[a["from"]], nd[a["to"]]
        c0, c1 = (src["x"], src["y"]), (dst["x"], dst["y"])
        if a["grade"] == "N":
            c, lw, ls = "#aab4bf", 1.0, ":"
        elif a["grade"] in ("M", "H"):
            c, lw, ls = "#8a99ad", 1.1, "--"
        elif src["role"] in ("trigger", "brake") and dst["role"] == "decision":
            c, lw, ls = col[src["role"]], 1.9, "-"
        else:
            c, lw, ls = "#5b6b7b", 1.4, "-"
        rad = ROUTE.get((a["from"], a["to"]), 0.0)
        (w0, h0), (w1, h1) = size[src["id"]], size[dst["id"]]
        via = VIA.get((a["from"], a["to"]))
        if via:
            pts = [(c0[0], c0[1] + h0), (c0[0], TOP)]
            pts += [(c1[0], TOP), (c1[0], c1[1] + h1 + 0.04)] if via == "top" else [(RIGHT, TOP), (RIGHT, c1[1]), (c1[0] + w1 + 0.05, c1[1])]
            ax.add_patch(FancyArrowPatch(path=_rounded(pts), arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=c, linestyle=ls, zorder=2))
            continue
        column = src["role"] in ("trigger", "brake", "context")
        if column and dst["role"] in ("decision", "outcome"):        # leave a reason from its right edge, so no arrow crosses a box
            p0 = (c0[0] + w0, c0[1])
            p1 = (c1[0] - w1, c1[1] + 0.06 * (c0[1] - c1[1])) if dst["role"] == "decision" else _edge(c1, (w1, h1), p0)
        elif src["role"] == "latent" and dst["role"] in ("trigger", "brake", "context"):
            p0, p1 = _edge(c0, (w0, h0), (c1[0] - w1, c1[1])), (c1[0] - w1, c1[1])
        elif (a["from"], a["to"]) == ("wearing_off", "motor_control"):
            p0, p1 = (c0[0] - w0, c0[1]), (c1[0] - w1, c1[1])
        else:
            p0, p1 = _edge(c0, (w0, h0), c1), _edge(c1, (w1, h1), c0)
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, linewidth=lw, color=c, linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=3, zorder=2))
    # the recorded blocks no guideline names: adjusted for, not drawn as nodes; Q3 sizes their share of the confounding
    ax.add_patch(FancyBboxPatch((-0.45, 0.02), 3.0, 1.8, boxstyle="round,pad=0.02,rounding_size=0.08", linewidth=1.0,
                                edgecolor="#b8c2cc", facecolor="#fafbfc", linestyle=":", zorder=1))
    ax.text(-0.32, 1.70, "Recorded, named by no guideline", fontsize=8.4, fontweight="bold", color="#0b2545", va="top")
    ax.text(-0.32, 1.40, "adjusted for in every estimate; Q3 sizes\ntheir share of the confounding", fontsize=7.3, color="#5b6b7b",
            va="top", linespacing=1.15)
    ax.text(-0.32, 0.86, "\n".join("•  " + b["label"] for b in load()["off_guideline_blocks"].values()), fontsize=7.6, color="#0b2545",
            va="top", linespacing=1.45)
    key = [Line2D([0], [0], color="#5b6b7b", lw=1.6, ls="-", label="guideline reason (G1, G2, E) or randomised evidence (R)"),
           Line2D([0], [0], color="#8a99ad", lw=1.2, ls="--", label="from a latent node: disease course (M) or hidden-confounding hypothesis (H)"),
           Line2D([0], [0], color="#aab4bf", lw=1.2, ls=":", label="natural-history carry-over (N)")]
    ax.legend(handles=key, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.8, handlelength=2.6,
              columnspacing=2.2, labelcolor="#5b6b7b")
    fig.tight_layout()
    path = pathlib.Path(path or (C.docs_dir() / "core_map.png"))
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def main(argv=None):
    C.require_ppmi_approval("the core-map check")  # escalation-discovery: PROTOCOL.md §11 gate
    import pandas as pd
    d = pd.read_csv(C.derived_dir() / "trials_dt_ppmi1.csv", nrows=5)
    missing = check(d)
    print(f"core map: {len(nodes())} nodes ({sum(n['measured'] for n in nodes().values())} measured), {len(arrows())} arrows, "
          f"{len(decision_arrows())} measured guideline reasons into the decision; {len(adjustment_columns())} adjustment columns in "
          f"{len(blocks())} blocks; missing from the trial table: {missing or 'none'}")
    print("figure:", draw())
