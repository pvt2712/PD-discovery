"""Sections 1.38-1.47: the WP1 consensus rules (wp1.py) on made-up ratings."""
from _setup import ROOT

from pdx import wp1

print("== 1.38 constants ==")
print("MIN_RATERS:", wp1.MIN_RATERS, "| domains:", len(wp1.DISPLAY), "| arrows split:",
      [wp1.ARROW.split(s) for s in ("A → B", "A->B", "A => B")])

print("\n== 1.39 domain and parse_link ==")
for label in ("Treatment (a higher dose)", "Axial and gait*", "  daily   FUNCTION ", "rbd"):
    print(f"  domain({label!r}) = {wp1.domain(label)!r}")
for text in ("Tremor → Daily function", "**Motor signs and daily life**", "Mood -> ", "Tremor → Daily → Mood"):
    print(f"  parse_link({text!r}) = {wp1.parse_link(text)}")
try:
    wp1.parse_link("Fatigue → Mood")
except ValueError as e:
    print("  parse_link('Fatigue → Mood') -> ValueError:", e)

print("\n== 1.40 answer ==")
for keep, strength, conf in [("Y", "2", "H"), ("yes", "1 −", "medium"), ("N", "2", ""), ("Y", "0", ""), ("", "0", ""),
                             ("?", "1", "L"), ("Y", "2?", ""), ("Y", "0?", ""), ("Y", "", ""), ("", "2", ""),
                             ("", "", ""), ("Y", "3", "H"), ("Y", "1-2", "H"), ("Y", "1–2", "H")]:
    print(f"  keep={keep!r:6} strength={strength!r:7} confidence={conf!r:9} -> {wp1.answer(keep, strength, conf)}")

print("\n== 1.41 candidates ==")
cands = wp1.candidates(ROOT / "docs" / "clinical_expert_questionnaire.md")
print("  ", len(cands), "candidate links; first", cands[0], "; last", cands[-1])

print("\n== 1.42 read_table ==")
e1 = [["#", "Link (this year → next year)", "Keep?", "Strength", "Confidence"],
      ["", "**Motor signs and daily life**", "", "", ""],
      ["1", "Tremor → Daily function", "Y", "2", "H"],
      ["2", "Mood → Cognition", "Y", "?", "L"]]
e1b = [["Link (this year → next year)", "Strength", "Confidence"], ["RBD symptoms → Hallucinations", "1", "M"]]
f = [["#", "Link (this year → next year)", "Expected (P/T/0/?)", "Plausible (1–5)"], ["F1.1", "Axial and gait → Independence", "P", "4"]]
print("  E1: ", wp1.read_table(e1))
print("  E1b:", wp1.read_table(e1b))
print("  F:  ", wp1.read_table(f))

print("\n== 1.43-1.44 tally and round1 (five raters) ==")
A = wp1.answer
listed = [(1, "tremor", "function"), (2, "axial", "independence"), (3, "mood", "cognition")]
raters = {f"r{r}": {("tremor", "function"): A("Y", "2" if r else "1", "H"),
                    ("axial", "independence"): A("N" if r < 4 else "Y", "" if r < 4 else "0"),
                    ("mood", "cognition"): A("Y", "1" if r < 3 else "?")} for r in range(5)}
raters["r0"][("rbd", "hallucinations")] = {**A("", "1", "M"), "added": True}
r1 = wp1.round1(raters, listed)
for link, res in r1["links"].items():
    print(f"  {wp1.key(link):22s} {res['verdict']:8s} added={res['added']!s:5} tally={res['tally']}")

print("\n== 1.45 round2 (five raters) ==")
back = [l for l, res in r1["links"].items() if res["verdict"] == "round 2"]
two = {"r0": {back[0]: A("Y", "1"), back[1]: A("Y", "2")},
       "r1": {back[0]: A("Y", "2"), back[1]: A("Y", "1")},
       "r2": {back[0]: A("Y", "1"), back[1]: A("N", "")},
       "r3": {back[0]: A("N", ""), back[1]: A("?", "")},
       "r4": {back[0]: A("Y", "2"), back[1]: A("", "")}}
r2 = wp1.round2(two, back)
for link, res in r2["links"].items():
    t = res["tally"]
    print(f"  {wp1.key(link):22s} in={t['in']} out={t['out']} unsure={t['unsure']} missing={t['missing']} -> "
          f"{res['verdict']!r}, disputed={res['disputed']}")

print("\n== 1.46 build_map ==")
m = wp1.build_map(r1, r2, "made-up ratings")
print("  ", m)
r2d = wp1.round2(two, back, {"rbd->hallucinations": "in"})
m = wp1.build_map(r1, r2d, "made-up ratings")
print("   with the investigator's decision:", {k: m[k] for k in ("links", "disputed", "not_settled", "investigator_led")})
four = {f"q{r}": {("treatment", "axial"): A("Y", "2-" if r < 2 else "2"), ("treatment", "tremor"): A("Y", "1-" if r < 3 else "1")}
        for r in range(4)}
m4 = wp1.build_map(wp1.round1(four, [(1, "treatment", "axial"), (2, "treatment", "tremor")]), None, "four raters")
print("   four raters, treatment links:", {k: m4[k] for k in ("treatment_links", "improving", "investigator_led", "raters")})

print("\n== 1.47 round2_markdown ==")
print(wp1.round2_markdown(r1))
