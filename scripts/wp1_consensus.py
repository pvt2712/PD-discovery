"""WP1: compile the panel's ratings into the expert map (protocol v0.5, §6.4). No PPMI data.

Returned questionnaires (Word files, or CSV answer sheets) stay in data/wp1/ (git-ignored: they
identify the raters). Only anonymised counts go to results/wp1/, with raters numbered in file order.

    /usr/bin/python3 scripts/wp1_consensus.py sheet data/wp1/answer_sheet.csv     # a blank CSV answer sheet
    /usr/bin/python3 scripts/wp1_consensus.py round1 data/wp1/round1/*.docx
    /usr/bin/python3 scripts/wp1_consensus.py round2 data/wp1/round2/*.docx [--decisions data/wp1/decisions.json]

The system Python reads Word files (python-docx); CSV sheets work with either Python.
- Round 1 writes results/wp1/round1.json and the round-2 table, results/wp1/round2_section_E.md.
- Round 2 writes results/wp1/round2.json and the draft map, configs/expert_map_draft.json. It
  becomes configs/expert_map.json once the investigator records it in the protocol (§17).
- --decisions settles links without a round-2 majority: {"from->to": "in" or "out"}; they are
  recorded as disputed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import _common
from pdx import wp1

OUT = _common.RESULTS / "wp1"
QUESTIONNAIRE = _common.ROOT / "docs" / "clinical_expert_questionnaire.md"


def save(r: dict, path: Path) -> None:
    path.write_text(json.dumps({"raters": r["raters"], "links": {wp1.key(l): v for l, v in r["links"].items()}},
                               indent=1))


def load(path: Path) -> dict:
    r = json.loads(path.read_text(encoding="utf-8"))
    return {"raters": r["raters"], "links": {tuple(k.split("->")): v for k, v in r["links"].items()}}


def read_raters(files: list, listed: list) -> dict:
    """Answers by rater number, with a check of how many listed links each rater answered."""
    raters = {}
    for i, f in enumerate(sorted(files), 1):
        ans = wp1.read_rater(f)
        answered = sum(1 for _, a, b in listed if ans.get((a, b), {}).get("verdict", "missing") != "missing")
        print(f"rater {i}: {len(ans)} links read, {answered} of the {len(listed)} listed answered")
        if not ans:
            sys.exit(f"No ratings found in rater {i}'s file: check it is the returned questionnaire.")
        raters[f"rater {i}"] = ans
    return raters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("sheet", "round1", "round2"))
    ap.add_argument("files", nargs="+")
    ap.add_argument("--decisions", default=None, help="JSON {'from->to': 'in' or 'out'} for links without a majority")
    a = ap.parse_args()
    files = [_common.require_inside(Path(f), _common.DATA, "the questionnaires and answer sheets") for f in a.files]
    listed = wp1.candidates(QUESTIONNAIRE)
    if a.step == "sheet":
        files[0].parent.mkdir(parents=True, exist_ok=True)
        with open(files[0], "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["link", "keep", "strength", "confidence"])
            w.writerows([[f"{wp1.DISPLAY[x]} → {wp1.DISPLAY[y]}", "", "", ""] for _, x, y in listed])
        print("wrote", files[0])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if a.step == "round1":
        r1 = wp1.round1(read_raters(files, listed), listed)
        save(r1, OUT / "round1.json")
        (OUT / "round2_section_E.md").write_text(wp1.round2_markdown(r1), encoding="utf-8")
        c = Counter(v["verdict"] for v in r1["links"].values())
        print(f"{r1['raters']} raters: {c['in']} links in, {c['out']} out, {c['round 2']} back in round 2. "
              "Wrote results/wp1/round1.json and round2_section_E.md.")
        return
    r1 = load(OUT / "round1.json")
    returning = [l for l, v in r1["links"].items() if v["verdict"] == "round 2"]
    decisions = json.loads(Path(a.decisions).read_text(encoding="utf-8")) if a.decisions else {}
    r2 = wp1.round2(read_raters(files, [(0, *l) for l in returning]), returning, decisions)
    save(r2, OUT / "round2.json")
    m = wp1.build_map(r1, r2, f"{QUESTIONNAIRE.name}, rounds 1 and 2")
    path = _common.ROOT / "configs" / "expert_map_draft.json"
    path.write_text(json.dumps(m, indent=1))
    print(f"Map: {len(m['links'])} domain links and {len(m['treatment_links'])} treatment links; "
          f"{len(m['disputed'])} disputed; investigator-led: {m['investigator_led']}. Wrote {path.name}.")
    if m["not_settled"]:
        print("Still to settle (no round-2 majority): " + ", ".join(m["not_settled"])
              + ". Add them to --decisions and rerun.")


if __name__ == "__main__":
    main()
