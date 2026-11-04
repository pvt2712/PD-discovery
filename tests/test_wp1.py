"""WP1 consensus rules (§6.4) on made-up ratings."""
import csv
import tempfile
from pathlib import Path

from pdx import ppmi, wp1

ROOT = Path(__file__).resolve().parent.parent
QUESTIONNAIRE = ROOT / "docs" / "clinical_expert_questionnaire.md"


def test_candidates_read_from_the_questionnaire_use_known_domains():
    cands = wp1.candidates(QUESTIONNAIRE)
    assert [c[0] for c in cands] == list(range(1, 32))
    names = {x for c in cands for x in c[1:]}
    assert names <= set(ppmi.DOMAINS) | {"treatment"}
    assert set(wp1.DISPLAY) == set(ppmi.DOMAINS) | {"treatment"}
    assert (24, "tremor", "treatment") in cands and (31, "treatment", "sleepiness") in cands


def test_answers_are_read_leniently():
    assert wp1.answer("Y", "2", "H") == {"verdict": "in", "strength": 2, "sign": 1, "confidence": "H"}
    assert wp1.answer("yes", "2 −", "medium")["sign"] == -1
    assert wp1.answer("Y", "0", "")["verdict"] == "out"
    assert wp1.answer("N", "", "")["verdict"] == "out"
    assert wp1.answer("?", "1", "L")["verdict"] == "unsure"
    assert wp1.answer("Y", "?", "")["verdict"] == "unsure"
    assert wp1.answer("Y", "", "")["verdict"] == "unsure"
    assert wp1.answer("", "", "")["verdict"] == "missing"


def _rater(rows, added=()):
    """Answers as read from an E1 table and an E1b table."""
    e1 = [["#", "Link (this year → next year)", "Keep?", "Strength", "Confidence"]]
    e1 += [[str(i), f"{a} → {b}", k, s, "H"] for i, (a, b, k, s) in enumerate(rows, 1)]
    e1b = [["Link (this year → next year)", "Strength", "Confidence"]] + [[f"{a} → {b}", s, "M"] for a, b, s in added]
    out = wp1.read_table(e1)
    out.update(wp1.read_table(e1b))
    return out


LISTED = [(1, "tremor", "function"), (2, "axial", "independence"), (3, "mood", "cognition")]


def test_round_one_needs_every_rater_and_sends_the_rest_to_round_two():
    raters = {}
    for r in range(5):
        rows = [("Tremor", "Daily function", "Y", "2" if r else "1"),            # all 1 or 2: in
                ("Axial and gait", "Independence", "N" if r < 4 else "Y", "0"),   # all 0 or N: out
                ("Mood", "Cognition", "Y", "1" if r < 3 else "?")]              # mixed: round 2
        added = [("RBD symptoms", "Hallucinations", "1")] if r == 0 else []
        raters[f"r{r}"] = _rater(rows, added)
    r1 = wp1.round1(raters, LISTED)
    v = {wp1.key(l): x["verdict"] for l, x in r1["links"].items()}
    assert v == {"tremor->function": "in", "axial->independence": "out", "mood->cognition": "round 2",
                 "rbd->hallucinations": "round 2"}
    assert r1["links"][("rbd", "hallucinations")]["added"]
    t = r1["links"][("tremor", "function")]["tally"]
    assert (t["strong"], t["weak"], t["in"]) == (4, 1, 5)
    md = wp1.round2_markdown(r1)
    assert "Mood → Cognition | 0 / 3 / 0 / 2 |" in md and "RBD symptoms → Hallucinations" in md
    assert "Tremor → Daily function" not in md


def test_round_two_majority_ties_and_the_map():
    raters = {f"r{r}": _rater([("Tremor", "Daily function", "Y", "2"), ("Treatment", "Axial and gait", "Y", "2-")])
              for r in range(5)}
    r1 = wp1.round1(raters, [(1, "tremor", "function"), (2, "treatment", "axial")])
    assert all(x["verdict"] == "in" for x in r1["links"].values())
    back = [("mood", "cognition"), ("rbd", "hallucinations")]
    two = {"r0": {back[0]: wp1.answer("Y", "1"), back[1]: wp1.answer("Y", "2")},
           "r1": {back[0]: wp1.answer("Y", "2"), back[1]: wp1.answer("N", "")},
           "r2": {back[0]: wp1.answer("Y", "1"), back[1]: wp1.answer("?", "")},
           "r3": {back[0]: wp1.answer("N", ""), back[1]: wp1.answer("N", "")}}      # one rater left
    r2 = wp1.round2(two, back)
    assert r2["links"][back[0]]["verdict"] == "in" and not r2["links"][back[0]]["disputed"]
    assert r2["links"][back[1]]["verdict"] == "investigator to decide" and r2["links"][back[1]]["disputed"]
    m = wp1.build_map(r1, r2, "test")
    assert m["not_settled"] == ["rbd->hallucinations"] and m["investigator_led"]       # 4 raters in round 2
    r2 = wp1.round2(two, back, {"rbd->hallucinations": "in"})
    m = wp1.build_map(r1, r2, "test")
    assert sorted(m["links"]) == [["mood", "cognition"], ["rbd", "hallucinations"], ["tremor", "function"]]
    assert m["treatment_links"] == [["treatment", "axial"]] and m["improving"] == [["treatment", "axial"]]
    assert m["disputed"] == [["rbd", "hallucinations"]] and m["not_settled"] == []


def test_other_tables_and_csv():
    f_table = [["#", "Link (this year → next year)", "Expected (P/T/0/?)", "Plausible (1–5)"],
               ["F1.1", "Axial and gait → independence", "P", "4"]]
    assert wp1.read_table(f_table) == {}                                  # Section F is not a rating table
    path = Path(tempfile.mkdtemp()) / "r1.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["link", "keep", "strength", "confidence"], ["Tremor -> Daily function", "Y", "2", "H"],
                                 ["Mood => Cognition", "", "1", "L"]])
    a = wp1.read_csv(path)
    assert a[("tremor", "function")]["verdict"] == "in" and a[("mood", "cognition")]["verdict"] == "in"
