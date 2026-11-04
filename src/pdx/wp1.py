"""WP1: the expert map from the panel's ratings (protocol v0.5, §6.4).

Standard library only, so it also runs under the system Python that reads the returned Word
questionnaires (scripts/wp1_consensus.py).

Each rater's answer to a link (Section E1 or E1b of docs/clinical_expert_questionnaire.md) is
"in" (kept, strength 1 or 2), "out" (not kept, or strength 0), "unsure" (? anywhere) or
"missing" (no answer).
- Round 1: a link enters the map when every rater's answer is "in", and stays out when every
  rater's answer is "out". Every other link, and every link a rater adds in E1b, returns in round 2.
- Round 2: the round-2 raters decide by majority; a rater who misses round 2 leaves the counts.
  Without a majority either way the investigator decides, and the link is recorded as disputed.
- With fewer than five raters the map is described as investigator-led (D25).
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

DISPLAY = {
    "tremor": "Tremor", "brady_rigidity": "Bradykinesia and rigidity", "axial": "Axial and gait",
    "function": "Daily function", "independence": "Independence", "complications": "Motor complications",
    "cognition": "Cognition", "mood": "Mood", "autonomic": "Autonomic", "hallucinations": "Hallucinations",
    "sleepiness": "Daytime sleepiness", "rbd": "RBD symptoms", "treatment": "Treatment",
}
LABELS = {v.lower(): k for k, v in DISPLAY.items()}
ARROW = re.compile(r"\s*(?:→|->|=>|⟶)\s*")
MINUS = re.compile(r"[-−–]")
MIN_RATERS = 5


def domain(label: str) -> str:
    """The code's name for a questionnaire label ("Treatment (a higher dose)" is "treatment")."""
    key = re.sub(r"\s*\(.*?\)\s*", " ", label.replace("*", "")).strip().lower()
    key = re.sub(r"\s+", " ", key)
    if key in LABELS:
        return LABELS[key]
    if key in DISPLAY:                              # already a code name
        return key
    raise ValueError(f"unknown domain {label!r}")


def parse_link(text: str) -> tuple[str, str] | None:
    """("from", "to") for "From → To"; None for a row that is not a link (a section label, a blank)."""
    parts = ARROW.split(text.strip())
    if len(parts) != 2 or not all(p.strip() for p in parts):
        return None
    return domain(parts[0]), domain(parts[1])


def answer(keep: str, strength: str, confidence: str = "") -> dict:
    """One rater's answer to one link, read leniently (Y/yes, 2-, 2 −, ?, blank)."""
    k = keep.strip().upper()[:1]
    s = strength.strip()
    sign = -1 if MINUS.search(s) else 1
    digits = re.findall(r"[012]", s)
    level = int(digits[0]) if digits else None
    if k == "N" or (k in ("", "Y") and level == 0):
        verdict = "out"
    elif k == "?" or "?" in s:
        verdict = "unsure"
    elif level in (1, 2):
        verdict = "in"
    elif k == "Y":
        verdict = "unsure"                          # kept, but no strength given
    else:
        verdict = "missing"
    conf = confidence.strip().upper()[:1]
    return {"verdict": verdict, "strength": level, "sign": sign if verdict == "in" else None,
            "confidence": conf if conf in ("H", "M", "L") else None}


def candidates(questionnaire: str | Path) -> list:
    """The numbered candidate links of Section E1, read from the questionnaire's markdown."""
    out = []
    for line in Path(questionnaire).read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and cells[0].isdigit():
            link = parse_link(cells[1])
            if link:
                out.append((int(cells[0]), *link))
    return out


def read_table(rows: list) -> dict:
    """A rater's answers from one table (rows of cell texts) whose first row names its columns:
    Link, Keep?, Strength and Confidence (E1, and round 2), or Link, Strength and Confidence (E1b,
    where an added link counts as kept). Any other table (Section F's links, E2) gives nothing."""
    if not rows:
        return {}
    low = [c.strip().lower() for c in rows[0]]
    col = {name: next((i for i, c in enumerate(low) if c.startswith(name)), None)
           for name in ("link", "keep", "strength", "confidence")}
    if col["link"] is None or col["strength"] is None or col["confidence"] is None:
        return {}
    answers = {}
    for cells in rows[1:]:
        link = parse_link(cells[col["link"]]) if col["link"] < len(cells) else None
        if link is None:
            continue

        def cell(name):
            i = col[name]
            return cells[i] if i is not None and i < len(cells) else ""
        keep = cell("keep") if col["keep"] is not None else ("Y" if cell("strength").strip() else "")
        a = answer(keep, cell("strength"), cell("confidence"))
        a["added"] = col["keep"] is None
        answers[link] = a
    return answers


def read_csv(path: str | Path) -> dict:
    """A rater's answers from a CSV whose header is link, keep, strength, confidence (an added link:
    keep left blank, strength given)."""
    with open(path, newline="", encoding="utf-8") as f:
        return read_table(list(csv.reader(f)))


def read_docx(path: str | Path) -> dict:
    """A rater's answers from a returned Word questionnaire (needs python-docx), table by table."""
    from docx import Document
    answers = {}
    for t in Document(str(path)).tables:
        answers.update(read_table([[c.text for c in r.cells] for r in t.rows]))
    return answers


def read_rater(path: str | Path) -> dict:
    return read_docx(path) if str(path).lower().endswith(".docx") else read_csv(path)


def tally(answers: list) -> dict:
    """Anonymised counts of the raters' answers to one link."""
    c = Counter(a["verdict"] for a in answers)
    strengths = Counter(a["strength"] for a in answers if a["verdict"] == "in")
    signs = Counter(a["sign"] for a in answers if a["verdict"] == "in")
    conf = Counter(a["confidence"] for a in answers if a["confidence"])
    return {"in": c["in"], "out": c["out"], "unsure": c["unsure"], "missing": c["missing"],
            "strong": strengths[2], "weak": strengths[1], "improving": signs[-1],
            "confidence": {k: conf[k] for k in ("H", "M", "L")}}


def round1(raters: dict, listed: list) -> dict:
    """Round 1: each link in, out or back to round 2. `raters` maps a rater's code to its answers;
    `listed` are the candidate links (from, to). Links a rater adds are rated by that rater only."""
    links = list(dict.fromkeys([tuple(x[-2:]) for x in listed] + [l for a in raters.values() for l in a]))
    out = {}
    for link in links:
        ans = [a.get(link, {"verdict": "missing", "strength": None, "sign": None, "confidence": None})
               for a in raters.values()]
        t = tally(ans)
        n = len(ans)
        verdict = "in" if t["in"] == n else "out" if t["out"] == n else "round 2"
        out[link] = {"verdict": verdict, "tally": t,
                     "added": any(a.get(link, {}).get("added") for a in raters.values())}
    return {"raters": len(raters), "links": out}


def round2(raters: dict, returning: list, decisions: dict | None = None) -> dict:
    """Round 2: the majority of the round-2 raters decides each returning link; without a majority
    the investigator's decision ("in"/"out", keyed "from->to") settles it and it is disputed."""
    decisions = decisions or {}
    n = len(raters)
    out = {}
    for link in returning:
        ans = [a.get(link, {"verdict": "missing", "strength": None, "sign": None, "confidence": None})
               for a in raters.values()]
        t = tally(ans)
        if 2 * t["in"] > n:
            verdict, disputed = "in", False
        elif 2 * t["out"] > n:
            verdict, disputed = "out", False
        else:
            verdict, disputed = decisions.get(key(link), "investigator to decide"), True
        out[link] = {"verdict": verdict, "disputed": disputed, "tally": t}
    return {"raters": n, "links": out}


def key(link: tuple) -> str:
    return f"{link[0]}->{link[1]}"


def build_map(r1: dict, r2: dict | None, source: str) -> dict:
    """The expert map in the format the H-P run reads (configs/expert_map.json): domain links,
    treatment links, the disputed ones, the modal sign of each link, and the panel's size."""
    settled = {l: v for l, v in r1["links"].items() if v["verdict"] != "round 2"}
    if r2:
        settled.update(r2["links"])
    pending = [key(l) for l, v in r1["links"].items() if v["verdict"] == "round 2" and l not in settled]
    pending += [key(l) for l, v in settled.items() if v["verdict"] not in ("in", "out")]
    kept = [l for l, v in settled.items() if v["verdict"] == "in"]
    sizes = [r1["raters"]] + ([r2["raters"]] if r2 else [])
    return {
        "status": "draft: frozen only when the investigator records it in the protocol (§17)",
        "source": source,
        "raters": {"round_1": r1["raters"], "round_2": r2["raters"] if r2 else None},
        "investigator_led": min(sizes) < MIN_RATERS,
        "links": [list(l) for l in kept if "treatment" not in l],
        "treatment_links": [list(l) for l in kept if "treatment" in l],
        "disputed": [list(l) for l, v in settled.items() if v.get("disputed") and v["verdict"] == "in"],
        "disputed_left_out": [list(l) for l, v in settled.items() if v.get("disputed") and v["verdict"] == "out"],
        "improving": [list(l) for l in kept if (v := settled[l]["tally"])["improving"] > v["in"] / 2],
        "not_settled": pending,
    }


def round2_markdown(r1: dict) -> str:
    """Section E of the round-2 questionnaire: each returning link with the group's anonymised
    round-1 answers, and columns for the new rating."""
    n = r1["raters"]
    lines = [f"**E (round 2).** These links did not reach agreement in round 1 ({n} raters). For each,",
             "the group's round-1 answers are shown without names. Please rate it again, as in round 1.", "",
             "| Link (this year → next year) | Round 1: strong / weak / none / unsure | Keep? | Strength | Confidence |",
             "|---|---|---|---|---|"]
    for link, v in r1["links"].items():
        if v["verdict"] != "round 2":
            continue
        t = v["tally"]
        label = f"{DISPLAY[link[0]]} → {DISPLAY[link[1]]}"
        note = f"{t['strong']} / {t['weak']} / {t['out']} / {t['unsure'] + t['missing']}"
        if v["added"]:
            note += " (added by a rater)"
        lines.append(f"| {label} | {note} | | | |")
    return "\n".join(lines) + "\n"
