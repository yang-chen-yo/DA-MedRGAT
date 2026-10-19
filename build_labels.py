#!/usr/bin/env python3
"""
build_labels.py
---------------
Build one prediction label per patient in the knowledge graph, using the task
definitions of the paper (which follow GraphCare / PyHealth on MIMIC-IV):

  mortality    visit t has a next admission; y = 1 if the patient dies in
               that next admission (hospital_expire_flag), else 0
  readmission  visit t has a next admission; y = 1 if the next admission
               starts less than 15 days after the admission of visit t
  lenofstay    y = length of stay of visit t in whole days, mapped to 10
               classes: <1 day, 1, 2, ..., 7 days, 8-14 days, >14 days

Visit t is the most recent visit of the patient that is in the knowledge graph
(and, for mortality / readmission, that has a next admission in MIMIC-IV).
The label comes from visit t itself (LOS) or from the next admission
(mortality / readmission). build_patient_dataset() keeps only edges of
visits up to visit t, so the next admission is never part of the input.

Output: exp_data/labels/<task>.pkl, a list of
    {"patient_id", "visit_id", "visit_date", "label"}
which train.py reads through config.TASKS.

Usage:
    python build_labels.py                     # all three tasks
    python build_labels.py --task mortality
"""
import argparse, json, os, pickle
from collections import Counter
from pathlib import Path

import pandas as pd

from config import MIMIC4_ROOT, TASKS
from data.triple_loader import norm_patient

READMIT_DAYS = 15


def los_class(days: int) -> int:
    """10 LOS classes as in PyHealth's categorize_los."""
    if days < 1:
        return 0
    if days <= 7:
        return days
    if days <= 14:
        return 8
    return 9


def kg_visits(json_dir):
    """patient_id -> set of visit_ids that have edges in the KG."""
    visits = {}
    for fp in Path(json_dir).rglob("triples_*.json"):
        pid = norm_patient(fp.parent.name)
        visits.setdefault(pid, set()).update(t["visit_id"] for t in json.load(open(fp)))
    return visits


def load_admissions(root, pids):
    path = os.path.join(root, "admissions.csv")
    if not os.path.isfile(path):
        path += ".gz"
    adm = pd.read_csv(path, usecols=["subject_id", "hadm_id", "admittime",
                                     "dischtime", "hospital_expire_flag"])
    adm = adm[adm["subject_id"].astype(str).isin(pids)].copy()
    adm["visit_id"] = adm["hadm_id"].astype(str)
    adm["patient_id"] = adm["subject_id"].astype(str)
    adm["admit"] = pd.to_datetime(adm["admittime"])
    adm["disch"] = pd.to_datetime(adm["dischtime"])
    adm = adm.sort_values(["patient_id", "admit"])
    nxt = adm.groupby("patient_id")
    adm["next_admit"] = nxt["admit"].shift(-1)
    adm["next_expire"] = nxt["hospital_expire_flag"].shift(-1)
    return adm


def build(task, adm, visits):
    samples = []
    for pid, grp in adm.groupby("patient_id"):
        grp = grp[grp["visit_id"].isin(visits[pid])]
        if task in ("mortality", "readmission"):
            grp = grp[grp["next_admit"].notna()]
        else:
            grp = grp[grp["disch"].notna()]
        if grp.empty:
            continue
        v = grp.iloc[-1]
        if task == "mortality":
            label = int(v["next_expire"])
        elif task == "readmission":
            label = int((v["next_admit"] - v["admit"]).days < READMIT_DAYS)
        else:
            label = los_class((v["disch"] - v["admit"]).days)
        samples.append({
            "patient_id": pid,
            "visit_id": v["visit_id"],
            "visit_date": v["admittime"],   # same format as visit_date in the KG
            "label": label,
        })
    return samples


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", choices=["mortality", "readmission", "lenofstay"])
    pa.add_argument("--json_dir", default="triples_output")
    args = pa.parse_args()

    visits = kg_visits(args.json_dir)
    adm = load_admissions(MIMIC4_ROOT, set(visits))
    for task in [args.task] if args.task else ["mortality", "readmission", "lenofstay"]:
        samples = build(task, adm, visits)
        out = Path(TASKS[task]["pkl"])
        out.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(samples, open(out, "wb"))
        dist = dict(sorted(Counter(s["label"] for s in samples).items()))
        print(f"{task}: {len(samples)} patients, labels {dist} -> {out}")


if __name__ == "__main__":
    main()
