# DA-MedRGAT

Code for **DA-MedRGAT: Literature-Evidenced Clinical Knowledge Graphs with
Dual-Attention Graph Networks for Multi-Task Prediction on EHRs**.

The pipeline builds a patient-specific clinical knowledge graph (KG) from
MIMIC-IV records, enriches it with UMLS concepts and PubMed evidence, keeps a
literature-derived relation only if a biomedical NLI model judges that a
retrieved PubMed sentence entails it, and trains a relational graph attention
network with a visit-recency prior on the resulting graphs.

## Pipeline

| Step | Script | Output |
|---|---|---|
| 1. Load and preprocess MIMIC-IV | `main_bigData.py` | `cache/prep/prep_*.pkl`, task sample files |
| 2. Build the KG with literature verification | `main_paper_kg.py` | `triples_output/<patient_id>/` |
| 3. Build task labels | `build_labels.py` | `exp_data/labels/<task>.pkl` |
| 4. Train and evaluate | `train.py` | `checkpoints/<task>/` |
| 5. Evaluate the verifier | `test/eval_verifier_scifact.py`, `test/aggregate_and_eval_negcontrol.py` | printed metrics |
| 6. Figures | `curves.py`, `run_single_patient_fast.py`, `single_patient_evidence_graph.py`, `viz/` | figure files |

## Data access

This repository does **not** include any patient data or licensed resources.

- **MIMIC-IV** requires credentialed access through
  [PhysioNet](https://physionet.org/content/mimiciv/). After approval, point
  `MIMIC4_ROOT` in `.env` to the `hosp/` directory.
- **UMLS** requires a license from the
  [U.S. National Library of Medicine](https://www.nlm.nih.gov/research/umls/).
  Place the extracted files under `resources/umls_data/`.
- `resources/CCSCM.csv`, `resources/CCSPROC.csv`, and `resources/ATC.csv`
  are public code-mapping tables and are included.

## Setup

```bash
conda create -n da-medrgat python=3.10
conda activate da-medrgat
pip install -r requirements.txt

cp .env.example .env   # then fill in GOOGLE_API_KEY, PUBMED_API_KEY, NCBI_EMAIL, MIMIC4_ROOT
```

API keys are read only from `.env` or environment variables; never put them in
source files.

The reported results use **Gemma 3 4B (`gemma-3-4b-it`)** through the Google
Generative AI API with temperature 0 for triple generation; this is the default
in `config.py` and `.env.example`. The `openai_compat` backend (e.g. Qwen2.5
served by Ollama or vLLM) and other Gemini models are optional alternative
backends, not used for the reported results.

## Usage

**1. Preprocess.** Set `TASK_NAME` at the top of `main_bigData.py`
(`mortality`, `readmission`, or `lenofstay`) and run:

```bash
python main_bigData.py
```

**2. Build the knowledge graph.** Builds graphs for the patient–visit pairs
used by the three tasks:

```bash
python main_paper_kg.py            # all pairs
python main_paper_kg.py --limit 50 # quick test
```

**3. Build task labels.** One label per patient in the KG, read from the
MIMIC-IV admissions table (see *Task definitions* below):

```bash
python build_labels.py
```

**4. Train.**

```bash
python train.py --task mortality   --kfold 10 --oversample
python train.py --task readmission --kfold 10 --oversample
python train.py --task lenofstay   --kfold 10 --oversample
```

Useful options: `--ablation {both,beta_only,heads_only,none}`,
`--no_literature` (UMLS-only ablation: drops literature edges), `--hid`,
`--layers`, `--heads`, `--lr`, `--dropout`, `--batch_size`, `--boot_mult`,
`--out_dir`. Per-fold reports, learning curves, and `metrics_cv.csv` (mean
and standard deviation over folds) are written to `checkpoints/<task>/`
unless `--out_dir` is given.

**5. Evaluate the verifier.**

```bash
python test/eval_verifier_scifact.py          # SciFact dev split
python test/aggregate_and_eval_negcontrol.py  # entity / evidence / relation / direction controls
```

SciFact data is not included; download it to `test/fixtures/scifact/`.

**6. Learning curves.** Mean ± standard deviation over folds, from the CSV
files written by `train.py`:

```bash
python curves.py --run checkpoints/readmission --metric val_auroc
```

## Task definitions

Labels are built by `build_labels.py` following GraphCare / PyHealth on
MIMIC-IV. For each patient, visit *t* is the most recent visit in the KG
(for mortality and readmission, the most recent one that has a next
admission). The input graph contains only edges from visits up to and
including visit *t*.

- **Mortality**: binary; positive if the patient dies during the **next**
  admission.
- **Readmission**: binary; positive if the next admission starts less than
  **15 days** after the admission of visit *t*.
- **Length of stay**: length of visit *t* in whole days, in **10 classes**
  (<1 day, 1, 2, …, 7 days, 8–14 days, >14 days).
- Splits are made over unique patient IDs, stratified by label, so no patient
  appears in more than one split. All random seeds are fixed to 42.
- Metrics: AUROC and AUPRC for binary tasks; AUROC (one-vs-one,
  support-weighted), quadratic-weighted Cohen's kappa, accuracy, and weighted F1
  for length of stay.

## Repository structure

```
config.py              settings (values overridable through .env)
main_bigData.py        MIMIC-IV loading and preprocessing
main_paper_kg.py       KG construction with literature verification
build_labels.py        task labels (mortality, readmission, LOS)
train.py               training and evaluation
data/                  dataset construction and label loading
embed/                 text encoder and FAISS index
retrieval/             PubMed and UMLS clients
fusion/                triple generation, NLI verifier, evidence scoring
kg/                    triple data structures
models/rgat_deep.py    relational GAT with visit-recency prior
test/                  unit tests and verifier evaluation
viz/                   knowledge-graph figures
resources/             public code-mapping tables
```

## License

Released under the [MIT License](LICENSE).
