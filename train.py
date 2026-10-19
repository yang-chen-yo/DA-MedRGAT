#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py —— Patient‑subgraph GNN
★ 支援：
    • 固定 70/15/15 split  (預設)
    • Stratified K‑fold     (--kfold 5 之類)
    • 任務‑依據指標：
        mortality / readmission     → AUROC  AUPRC
        lenofstay                   → AUROC  Kappa  Accuracy  F1
        drugrec                     → AUROC  AUPRC  F1  Jaccard
輸出：
    checkpoints/f{k}_best.pt
    checkpoints/f{k}_report.txt
    checkpoints/val_curve_f{k}.png
    checkpoints/loss_curve_f{k}.csv
    checkpoints/metrics_cv.csv   (Mean + Std for K‑fold)
"""

import argparse, random, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             cohen_kappa_score, classification_report,
                             average_precision_score, jaccard_score)

from tqdm.auto import tqdm

import config
from data.patient_dataset import build_patient_dataset
from models.rgat_deep import DeepRGAT   # 已修 RGAT edge‑dim=0 bug

warnings.filterwarnings(
    "ignore", message="An issue occurred while importing 'torch-sparse'"
)

# ---------- 小工具 ----------
def scalar(y):
    """把 y 轉成單一整數標籤（多類 one‑hot → int）"""
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    y = np.asarray(y).reshape(-1)
    return int(np.argmax(y)) if len(y) > 1 else int(y[0])


def moving_avg(arr, w=5):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


# ========================= main =========================
def main() -> None:
    pa = argparse.ArgumentParser()
    pa.add_argument("--task", required=True, choices=config.TASKS)
    pa.add_argument("--json_dir", default="triples_output")
    pa.add_argument("--device", default="cuda")
    pa.add_argument("--out_dir", default=None,
                    help="checkpoint / metrics directory (default: checkpoints/<task>)")
    pa.add_argument("--no_literature", action="store_true",
                    help="drop literature (RAG) edges: the UMLS-only ablation")

    # === 消融設定：只用 β、只用多頭、兩者皆用、兩者皆關 ===
    pa.add_argument("--ablation", default="both",
                    choices=["both", "beta_only", "heads_only", "none"],
                    help="both=β+多頭；beta_only=僅β；heads_only=僅多頭；none=兩者皆關")

    # HP
    pa.add_argument("--batch_size", type=int, default=32)
    pa.add_argument("--epochs", type=int, default=60)
    pa.add_argument("--hid", type=int, default=64)
    pa.add_argument("--layers", type=int, default=3)
    pa.add_argument("--heads", type=int, default=8)      # 由 ablation 決定實際使用值
    pa.add_argument("--lr", type=float, default=1e-4)
    pa.add_argument("--dropout", type=float, default=0.15)
    pa.add_argument("--patience", type=int, default=12)

    # oversample
    pa.add_argument("--oversample", action="store_true")
    pa.add_argument("--boot_mult", type=int, default=5)

    # CV
    pa.add_argument("--kfold", type=int, default=0,
                    help=">1 啟用 Stratified K-fold；0 為固定 split")
    pa.add_argument("--val_rate", type=float, default=0.15,
                    help="train 再切出此比例作 early-stop validation")


    args = pa.parse_args()

    # ----- 固定亂數種子與裝置 -----
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"   # 混合精度只在 GPU 上用；CPU 的 bfloat16 無法轉 numpy
    ckp_dir = Path(args.out_dir) if args.out_dir else Path("checkpoints") / args.task
    ckp_dir.mkdir(parents=True, exist_ok=True)

    # ----- 依 ablation 決定是否使用 α（語意注意力）與 β（就診新近度）-----
    use_alpha, use_beta = {
        "both":       (True,  True),
        "heads_only": (True,  False),   # 只有 α
        "beta_only":  (False, True),    # 只有 β
        "none":       (False, False),
    }[args.ablation.lower()]
    heads_eff = args.heads if use_alpha else 1

    print(f"[Ablation] use_alpha={use_alpha}  use_beta={use_beta}  heads={heads_eff}")

    # ----- 載入子圖 dataset -----
    # 若使用 β，就在資料集中帶入 edge_weight
    conf = config.TASKS[args.task]
    dataset, ent2id, rel2id, out_dim_orig = build_patient_dataset(
        args.json_dir, conf["pkl"], args.task, use_time=use_beta,
        use_literature=not args.no_literature
    )

    # lenofstay 的標籤已由 build_labels.py 分成 10 類；類別索引需為整數
    if args.task == "lenofstay":
        for d in dataset:
            d.y = d.y.long()
        out_dim = 10
    else:
        out_dim = out_dim_orig

    mode = {
        "mortality": "binary",
        "readmission": "binary",
        "lenofstay": "multiclass",
        "drugrec": "multilabel",
    }[args.task]

    # ── patient-level Stratified split ──
    labels      = np.array([scalar(d.y) for d in dataset])
    patient_ids = np.array([d.patient_id for d in dataset])
    uniq_pids   = np.unique(patient_ids)
    strat_lbl   = np.array([labels[patient_ids == pid][0] for pid in uniq_pids])

    folds = []
    if args.kfold and args.kfold > 1:     # 啟用 K-fold
        skf = StratifiedKFold(n_splits=args.kfold,
                              shuffle=True, random_state=42)
        for tr_idx, te_idx in skf.split(uniq_pids, strat_lbl):
            tr_pids_full = uniq_pids[tr_idx]
            te_pids = uniq_pids[te_idx]
            tr_labels = [labels[patient_ids == pid][0] for pid in tr_pids_full]
            tr_pids, va_pids = train_test_split(
                tr_pids_full, test_size=args.val_rate, random_state=42,
                stratify=tr_labels)
            folds.append((tr_pids, va_pids, te_pids))
    else:                                  # 固定 70 / 15 / 15
        tr_pids, tmp_pids = train_test_split(
            uniq_pids, test_size=0.30, random_state=42,
            stratify=strat_lbl)
        va_pids, te_pids = train_test_split(
            tmp_pids, test_size=0.50, random_state=42,
            stratify=[labels[patient_ids == pid][0] for pid in tmp_pids])
        folds.append((tr_pids, va_pids, te_pids))

    out_dim_task = out_dim

    # ----- 工具：Oversample sampler -----
    def build_sampler(ds, boot_mult, num_classes):
        y = np.array([scalar(d.y) for d in ds])
        if mode == "binary":
            neg, pos = (y == 0).sum(), (y == 1).sum()
            if pos == 0:
                raise RuntimeError("Train set 沒有任何正例，無法訓練！")
            w = np.where(y == 1, neg / pos, 1.0).astype(np.float32)
        else:
            cnt = np.bincount(y, minlength=num_classes)
            w = (cnt.sum() / (cnt[y] + 1e-8)).astype(np.float32)
        return WeightedRandomSampler(
            torch.tensor(w, dtype=torch.float32),
            num_samples=len(w) * boot_mult,
            replacement=True,
        )

    # ===== 開始 K-fold / 固定 split 訓練 =====
    all_metrics = []

    for fold_i, (tr_pids, va_pids, te_pids) in enumerate(folds, 1):
        print(f"\n====== Fold {fold_i}/{len(folds)} ======")

        # --- 依 patient_id 取子集 ---
        def pick_by_pid(pids):
            mask = np.isin(patient_ids, pids)
            return [dataset[i] for i in np.where(mask)[0]]

        train_ds = pick_by_pid(tr_pids)
        val_ds   = pick_by_pid(va_pids)
        test_ds  = pick_by_pid(te_pids)
        print(f"Train={len(train_ds)}  Val={len(val_ds)}  Test={len(test_ds)}")


        sampler = (build_sampler(train_ds, args.boot_mult, out_dim_task)
                   if args.oversample else None)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  sampler=sampler, shuffle=(sampler is None),
                                  drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

        # ----- Model & Optim -----
        model = DeepRGAT(
            num_nodes=len(ent2id), num_rel=len(rel2id), hid=args.hid,
            num_layers=args.layers, heads=heads_eff,
            out_dim=out_dim_task, dropout=args.dropout,
            use_alpha=use_alpha, use_beta=use_beta
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=6, min_lr=1e-6
        )
        scaler = GradScaler(enabled=use_amp)

        # ----- Loss -----
        # 類別不平衡只處理一次：開 --oversample 時 sampler 已平衡類別，loss 不再加權
        if mode == "binary":
            y_tr = np.array([scalar(d.y) for d in train_ds])
            pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=None if args.oversample else torch.tensor(pos_w, device=device))
        elif mode == "multiclass":
            y_tr = np.array([scalar(d.y) for d in train_ds])
            cnt = np.bincount(y_tr, minlength=out_dim_task)
            w = torch.tensor(1 / np.clip(cnt, 1, None),
                             dtype=torch.float32).to(device)
            loss_fn = nn.CrossEntropyLoss(weight=None if args.oversample else w)
        else:  # multilabel
            loss_fn = nn.BCEWithLogitsLoss()

        # ----- Train loop -----
        history = {"ep": [], "loss": [], "val_auroc": []}
        best, wait = 0, 0
        for ep in range(1, args.epochs + 1):
            model.train(); total = 0
            for batch in tqdm(train_loader, leave=False,
                              desc=f"Fold{fold_i} E{ep:02d}"):
                batch = batch.to(device)
                optimizer.zero_grad()
                with autocast(device_type=device.type, enabled=use_amp):
                    out = model(
                        batch.edge_index, batch.edge_type,
                        batch.edge_weight if use_beta else None,  # 只有用 β 才傳
                        batch.batch, batch.x)
                    if mode == "multiclass":
                        yb = batch.y
                        if yb.dim() > 1: yb = yb.argmax(-1)
                        loss = loss_fn(out, yb.long())
                    else:
                        loss = loss_fn(out.squeeze(),
                                       batch.y.float().view(-1))
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
                total += loss.item()

            # ---- validation ----
            model.eval()
            with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
                outs, ys = [], []
                for vb in val_loader:
                    vb = vb.to(device)
                    outs.append(model(
                        vb.edge_index, vb.edge_type,
                        vb.edge_weight if use_beta else None,
                        vb.batch, vb.x
                    ))
                    ys.append(vb.y)
                val_out = torch.cat(outs)
                val_y   = torch.cat(ys)

                truth = val_y.view(-1).cpu().numpy()
                if mode == "multiclass":
                    truth_oh = np.eye(out_dim_task)[truth]
                    probs    = val_out.softmax(-1).cpu().numpy()
                    if np.unique(truth).size < 2:
                        val_auroc = np.nan
                    else:
                        val_auroc = roc_auc_score(
                            truth_oh, probs,
                            multi_class="ovo", average="weighted"
                        )
                else:
                    prob = val_out.sigmoid().view(-1).cpu().numpy() \
                           if mode == "binary" else \
                           val_out.sigmoid().cpu().numpy()
                    if np.unique(truth).size < 2:
                        val_auroc = np.nan
                    else:
                        val_auroc = roc_auc_score(
                            truth, prob,
                            average="macro" if mode == "multilabel" else None
                        )

            key_score = val_auroc
            if not np.isnan(key_score):
                scheduler.step(key_score)
            cur_lr = optimizer.param_groups[0]["lr"]

            history["ep"].append(ep)
            history["loss"].append(total / len(train_loader))
            history["val_auroc"].append(val_auroc)

            print(f"E{ep:02d} "
                  f"loss={total/len(train_loader):.6f} "
                  f"ValAUROC={val_auroc:.6f} "
                  f"LR={cur_lr:.2e}")

            if not np.isnan(key_score) and key_score > best:
                best, wait = key_score, 0
                torch.save(model.state_dict(), ckp_dir / f"f{fold_i}_best.pt")
            else:
                wait += 1
                if wait >= args.patience:
                    print("Early stop!")
                    break

        # ----- Test -----
        model.load_state_dict(torch.load(ckp_dir / f"f{fold_i}_best.pt"))
        model.eval()
        with torch.no_grad(), autocast(device_type=device.type, enabled=use_amp):
            outs, ys = [], []
            for tb in test_loader:
                tb = tb.to(device)
                outs.append(model(tb.edge_index, tb.edge_type,
                                  tb.edge_weight if use_beta else None,
                                  tb.batch, tb.x))
                ys.append(tb.y)
        test_out = torch.cat(outs); test_y = torch.cat(ys)

        # ---------- 指標計算 ----------
        if args.task in ["mortality", "readmission"]:           # 二分類
            prob  = test_out.sigmoid().view(-1).cpu().numpy()
            truth = test_y.view(-1).cpu().numpy()
            auroc = roc_auc_score(truth, prob)
            auprc = average_precision_score(truth, prob)
            print(f"★ AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
            pred  = (prob >= 0.5).astype(int)
            rep   = classification_report(truth, pred, digits=4)
            fold_metric = {"AUROC": auroc, "AUPRC": auprc}

        elif args.task == "lenofstay":                          # 多分類
            probs = test_out.softmax(-1).cpu().numpy()
            truth = test_y.cpu().numpy()
            pred  = probs.argmax(-1)
            acc   = accuracy_score(truth, pred)
            kappa = cohen_kappa_score(truth, pred, weights="quadratic")
            truth_1h = np.eye(out_dim_task)[truth]
            auroc = roc_auc_score(truth_1h, probs,
                                  multi_class="ovo", average="weighted")
            f1    = f1_score(truth, pred, average="weighted")
            print(f"★ AUROC={auroc:.4f}  Kappa={kappa:.4f}  "
                  f"Acc={acc:.4f}  F1={f1:.4f}")
            rep   = classification_report(truth, pred, digits=4)
            fold_metric = {"AUROC": auroc, "Kappa": kappa,
                           "Acc": acc, "F1": f1}

        elif args.task == "drugrec":                            # 多標籤（目前不用）
            prob  = test_out.sigmoid().cpu().numpy()
            truth = test_y.cpu().numpy()
            pred  = (prob >= 0.5).astype(int)
            auroc = roc_auc_score(truth, prob, average="macro")
            auprc = average_precision_score(truth, prob, average="macro")
            f1    = f1_score(truth, pred, average="samples")
            jacc  = jaccard_score(truth, pred, average="samples")
            print(f"★ AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
                  f"F1={f1:.4f}  Jaccard={jacc:.4f}")
            rep   = classification_report(truth, pred, digits=4)
            fold_metric = {"AUROC": auroc, "AUPRC": auprc,
                           "F1": f1, "Jaccard": jacc}
        else:
            prob  = test_out.sigmoid().cpu().numpy()
            truth = test_y.cpu().numpy()
            pred  = (prob >= 0.5).astype(int)
            acc   = accuracy_score(truth, pred)
            print(f"★ ACC={acc:.4f}")
            rep   = classification_report(truth, pred, digits=4)
            fold_metric = {"Acc": acc}

        # 儲存報告與曲線
        Path(ckp_dir / f"f{fold_i}_report.txt").write_text(rep)
        np.savetxt(ckp_dir / f"loss_curve_f{fold_i}.csv",
           np.vstack([history["ep"], history["loss"]]).T,
           fmt="%.6f", delimiter=",", header="epoch,loss")
        np.savetxt(
            ckp_dir / f"val_auroc_curve_f{fold_i}.csv",
            np.vstack([history["ep"], history["val_auroc"]]).T,
            fmt="%.6f", delimiter=",",
            header="epoch,val_auroc"
        )
        all_metrics.append(fold_metric)

        # ==== 繪製 Val AUROC 曲線 ====
        plt.figure()
        epochs = np.array(history["ep"])
        aurocs = np.array(history["val_auroc"])
        valid  = ~np.isnan(aurocs)
        plt.plot(epochs[valid], moving_avg(aurocs)[valid], "-s")
        plt.xlabel("Epoch")
        plt.ylabel("Val AUROC")
        plt.tight_layout()
        plt.savefig(ckp_dir / f"val_auroc_curve_f{fold_i}.png")
        plt.close()

        # ==== 繪製 Train Loss 曲線（跳過 NaN） ====
        plt.figure()
        losses = np.array(history["loss"])
        valid_l = ~np.isnan(losses)
        plt.plot(epochs[valid_l], losses[valid_l], "-o")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss")
        plt.tight_layout()
        plt.savefig(ckp_dir / f"loss_curve_f{fold_i}.png")
        plt.close()

    # ====== 聚合結果 ======
    df = pd.DataFrame(all_metrics)
    df.loc["Mean"] = df.mean()
    df.loc["Std"]  = df.std()
    df.to_csv(ckp_dir / "metrics_cv.csv", float_format="%.4f")

    print("\n=== K-fold / 固定 split 指標 ===")
    print(df)


if __name__ == "__main__":
    main()

