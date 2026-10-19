import pickle, torch
from data.triple_loader import norm_patient
from collections import Counter

def load_labels(pkl_path, ent2id, task):
    samples = pickle.load(open(pkl_path, "rb"))
    N = len(ent2id)

    # 初始化 y
    if task == "drugrec":
        out_dim = samples[0]["label"]["drugs_ind"].shape[0]
        y = torch.zeros((N, out_dim), dtype=torch.float)
    else:
        y = torch.full((N,), -1., dtype=torch.float)

    # 每位病人標籤所屬就診的日期；同病人多筆樣本時與標籤一樣取最後一筆
    label_date = {}

    # 填入標籤
    for s in samples:
        pid_raw = str(s["patient_id"])
        pid_norm = norm_patient(pid_raw)
        if pid_norm not in ent2id:
            continue
        nid = int(ent2id[pid_norm])
        label_date[pid_norm] = str(s["visit_date"])

        if task == "drugrec":
            y[nid] = torch.tensor(s["label"]["drugs_ind"], dtype=torch.float)
        elif task == "mortality":
            lab = s["label"]
            val = lab["mortality"] if isinstance(lab, dict) else lab
            y[nid] = float(val)
        elif task == "readmission":
            lab = s["label"]
            val = lab["readmit_30d"] if isinstance(lab, dict) else lab
            y[nid] = float(val)
        else:  # lenofstay
            lab = s["label"]
            val = lab["los_bucket"] if isinstance(lab, dict) else lab
            y[nid] = int(val)

    if not label_date:
        raise RuntimeError("❌ 無任何標籤對應到圖中節點，請檢查命名規則！")
    if task != "drugrec":
        print(f"標籤分佈 ⇒ {Counter(y[y != -1].tolist())}")

    # 切分在 train.py 依病人做，這裡只回傳標籤
    return y, label_date

