"""Sections 1.48-1.52: the run scripts of the data layer, end to end on the repository's mock download
and on made-up questionnaire answers. Every folder a script writes to is redirected to a temporary
folder first (scripts/_common.py's DATA and RESULTS, wp1_consensus.OUT, and _common.ROOT for the
draft map), and checked before each run."""
import csv
import json
import shutil
import sys

import numpy as np
import pandas as pd
from _setup import TMP, mock_download, protect_script_paths

_common = protect_script_paths()
import build_analysis_panel  # noqa: E402
import build_ppmi_tables  # noqa: E402
import run_wp0_audit  # noqa: E402
import wp1_consensus  # noqa: E402

from pdx import ppmi  # noqa: E402
from pdx.data import load_panel  # noqa: E402


def run(module, *argv):
    assert TMP in _common.DATA.parents and TMP in _common.RESULTS.parents
    sys.argv = [module.__name__ + ".py", *map(str, argv)]
    try:
        module.main()
    except SystemExit as e:
        print("   SystemExit:", e)


print("== 1.48 _common ==")
print("ROOT is the repository:", (_common.ROOT / "PROTOCOL.md").exists())
print("require_inside, inside: ", _common.require_inside(TMP / "data" / "x.parquet", _common.DATA, "x").relative_to(TMP))
for args in [(TMP / "elsewhere.parquet", _common.DATA, "the visit table"), (TMP / "data", _common.DATA, "the folder itself")]:
    try:
        _common.require_inside(*args)
    except SystemExit as e:
        print("require_inside, outside:", str(e).replace(str(TMP), "<tmp>"))
for ref in (None, "ok", "synthetic run, 4 Nov 2026"):
    try:
        _common.require_approval(ref, "the demo")
        print(f"require_approval({ref!r}): passes")
    except SystemExit as e:
        print(f"require_approval({ref!r}): {str(e)[:60]}...")
print("to_json:", _common.to_json({"a": np.float64(0.5), "b": np.array([1, 2]), "c": float("nan"), "d": np.int64(3),
                                    "e": np.bool_(True), ("x", 1): [np.inf, 2.0]}))

print("\n== 1.49 build_ppmi_tables.py ==")
raw = mock_download()
run(build_ppmi_tables)                                                   # no approval: refused
run(build_ppmi_tables, "--approval", "synthetic run, 4 Nov 2026", "--raw", raw)
print("written:", sorted(str(p.relative_to(TMP)) for p in TMP.rglob("*") if p.is_file() and "mock" not in p.parts))
flow = json.loads((_common.RESULTS / "wp0" / "participant_flow.json").read_text())
print("participant_flow.json cells:", flow["cells"])
man = json.loads((_common.RESULTS / "wp0" / "manifest.json").read_text())
print("manifest:", {k: (len(v) if k == "files_sha256" else v) for k, v in man.items()})
print("retest_reliability.json:", json.loads((_common.RESULTS / "wp0" / "retest_reliability.json").read_text()))

print("\n   manifest and read when a second download and a longer table name sit in the folders:")
raw2 = TMP / "mock2"
shutil.copytree(raw, raw2)
old = pd.read_csv(raw2 / "_Subject_Characteristics" / "Participant_Status_01Jan2026.csv").assign(ENROLL_AGE=99.0)
old.to_csv(raw2 / "_Subject_Characteristics" / "Participant_Status_05Sep2025.csv", index=False)     # an older download
pd.DataFrame({"PATNO": [1]}).to_csv(raw2 / "Motor___MDS-UPDRS" / "MDS-UPDRS_Part_I_Patient_Questionnaire_01Jan2026.csv",
                                    index=False)
m2 = build_ppmi_tables.manifest(str(raw2), pd.DataFrame({"primary": [True]}), pd.DataFrame())
print("   stamps:", m2["download_stamps"], "| files hashed:", len(m2["files_sha256"]), "| includes the Part I patient file:",
      "MDS-UPDRS_Part_I_Patient_Questionnaire_01Jan2026.csv" in m2["files_sha256"])
print("   age at enrolment read by participants():", sorted(set(ppmi.participants(raw2).age_enroll)))

print("\n== 1.50 run_wp0_audit.py ==")
visits = _common.DATA / "derived" / "wp0_visits.parquet"
run(run_wp0_audit, "--visits", TMP / "wp0_visits.parquet", "--approval", "synthetic run, 4 Nov 2026")
run(run_wp0_audit, "--visits", visits, "--approval", "synthetic run, 4 Nov 2026")
rep = json.loads((_common.RESULTS / "wp0" / "wp0_audit.json").read_text())
print("audit keys:", list(rep))
print("participant_flow:", rep["participant_flow"])

print("\n== 1.51 build_analysis_panel.py ==")
meas = TMP / "measurement_synthetic.json"
meas.write_text(json.dumps({"source": "synthetic", "error_variance": {d: 0.25 for d in ppmi.DOMAINS}}))
run(build_analysis_panel, "--approval", "synthetic run, 4 Nov 2026", "--measurement", meas)
spec = json.loads((_common.DATA / "derived" / "panel_spec.json").read_text())
print("panel_spec.json:", {k: spec[k] for k in ("domains", "treated_domains", "meas_var", "measurement_source", "motor_on")})
p = load_panel(_common.DATA / "derived" / "panel_tdpigd.npz")
with np.load(_common.DATA / "derived" / "normal_scores.npz") as f:
    print("panel:", p.n, "participants,", p.d, "domains | normal_scores.npz arrays:", len(f.files))

print("\n== 1.52 wp1_consensus.py ==")
wp1_consensus.OUT = _common.RESULTS / "wp1"
_common.ROOT = TMP / "repo"
(TMP / "repo" / "configs").mkdir(parents=True)
sheet = _common.DATA / "wp1" / "answer_sheet.csv"
run(wp1_consensus, "sheet", sheet)
rows = list(csv.reader(open(sheet, encoding="utf-8")))
print("sheet:", len(rows) - 1, "links; first rows:", rows[:2])
answers = {1: ("Y", "2", "H"), 2: ("Y", "1", "M"), 5: ("N", "", "")}                  # every other link: unsure
for r in range(3):
    path = _common.DATA / "wp1" / "round1" / f"rater_{r}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(rows[0])
        for i, row in enumerate(rows[1:], 1):
            keep, strength, conf = answers.get(i, ("?", "", "L"))
            if i == 2 and r == 2:
                keep, strength = "Y", "?"                                             # one rater unsure on link 2
            w.writerow([row[0], keep, strength, conf])
run(wp1_consensus, "round1", *sorted((_common.DATA / "wp1" / "round1").glob("*.csv")))
r1 = json.loads((wp1_consensus.OUT / "round1.json").read_text())
print("round1.json:", r1["raters"], "raters;", {k: v["verdict"] for k, v in list(r1["links"].items())[:5]}, "...")
returning = [k for k, v in r1["links"].items() if v["verdict"] == "round 2"]
for r in range(3):
    path = _common.DATA / "wp1" / "round2" / f"rater_{r}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["link", "keep", "strength", "confidence"])
        for k in returning:
            a, b = k.split("->")
            verdict = ("Y", "1", "M") if (k == "brady_rigidity->function" or r < 2) else ("N", "", "")
            if k == "axial->function":
                verdict = [("Y", "2", "H"), ("N", "", ""), ("?", "", "")][r]           # no majority
            w.writerow([f"{wp1_consensus.wp1.DISPLAY[a]} → {wp1_consensus.wp1.DISPLAY[b]}", *verdict])
run(wp1_consensus, "round2", *sorted((_common.DATA / "wp1" / "round2").glob("*.csv")))
dec = _common.DATA / "wp1" / "decisions.json"
dec.write_text(json.dumps({"axial->function": "in"}))
run(wp1_consensus, "round2", *sorted((_common.DATA / "wp1" / "round2").glob("*.csv")), "--decisions", dec)
m = json.loads((TMP / "repo" / "configs" / "expert_map_draft.json").read_text())
print("expert_map_draft.json:", {k: (v if k not in ("links", "treatment_links") else len(v)) for k, v in m.items()})
