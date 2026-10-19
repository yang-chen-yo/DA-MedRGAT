#!/usr/bin/env python3
import argparse
import os
import pickle

from data.loader import load_mimic4_dataset, load_mappings, enrich_sample_with_names
from data.preprocess import preprocess_samples
from config import sample_dataset_path, UMLS_DATA_DIR, KG_MAX_HOPS
from fusion.kg_builder import build_patient_kg


def load_processed_samples():
    base = os.path.splitext(sample_dataset_path("mimic4", "drugrec"))[0]
    processed_path = base + "_preprocessed.pkl"

    if os.path.exists(processed_path):
        with open(processed_path, "rb") as f:
            return pickle.load(f)

    dataset = load_mimic4_dataset(True, "mimic4", "drugrec")
    raw_samples = preprocess_samples(dataset)
    cond_map, proc_map, drug_map, ndc_map = load_mappings()
    processed = [
        enrich_sample_with_names(s, cond_map, proc_map, drug_map, ndc_map)
        for s in raw_samples
    ]
    with open(processed_path, "wb") as f:
        pickle.dump(processed, f)
    return processed


def build_patient_context(sample):
    pid = sample["patient_id"]
    visit_id = sample["visit_id"]
    visit_date = sample.get("visit_date", "")
    conditions = [c["name"] for grp in sample.get("conditions", []) for c in grp]
    drugs = [d["name"] for grp in sample.get("drugs", []) for d in grp]
    procedures = [p["name"] for grp in sample.get("procedures", []) for p in grp]
    return pid, visit_id, visit_date, (
        f"PatientID: {pid}; VisitDate: {visit_date}; "
        f"Conditions: {conditions}; Drugs: {drugs}; Procedures: {procedures}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run the evidence fusion pipeline for one patient only.")
    parser.add_argument("--index", type=int, default=0, help="Index in the processed patient list")
    parser.add_argument("--pid", type=str, default=None, help="Optional patient_id override")
    parser.add_argument(
        "--max-hops", type=int, default=KG_MAX_HOPS,
        help=f"從 EHR entity 往外擴幾層文獻概念（default = config.KG_MAX_HOPS = {KG_MAX_HOPS}）",
    )
    args = parser.parse_args()

    processed = load_processed_samples()
    if not processed:
        raise RuntimeError("No processed patient samples found.")

    if args.pid is not None:
        sample = next((s for s in processed if str(s["patient_id"]) == str(args.pid)), None)
        if sample is None:
            raise ValueError(f"No sample with patient_id={args.pid}")
    else:
        sample = processed[args.index]

    pid, visit_id, visit_date, patient_context = build_patient_context(sample)
    patient_fields = {
        "conditions": sample.get("conditions", []),
        "procedures": sample.get("procedures", []),
        "drugs": sample.get("drugs", []),
    }

    print(f"[INFO] Running multi-hop patient KG for PID={pid}, VISIT={visit_id}, max_hops={args.max_hops}")
    print(f"[INFO] Context: {patient_context[:220]}...")

    triples = build_patient_kg(
        patient_id=pid,
        visit_id=visit_id,
        patient_context=patient_context,
        patient_fields=patient_fields,
        visit_date=visit_date,
        umls_dir=UMLS_DATA_DIR,
        pubmed_email=os.getenv("NCBI_EMAIL", ""),
        max_hops=args.max_hops,
    )

    out_path = os.path.join("triples_output", pid, f"single_patient_demo_{visit_id}.json")
    print(f"[INFO] Saved {len(triples)} accepted triples to {out_path}")
    for t in triples[:12]:
        print(" ", t.source, "|", t.relation, "|", t.head, "->", t.tail,
              "| ", (t.verifier_score if t.verifier_score is not None else t.support_score))


if __name__ == "__main__":
    main()
