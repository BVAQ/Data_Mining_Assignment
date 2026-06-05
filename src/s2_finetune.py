"""
Stage 2 (new) — fine-tune a scientific transformer (regression head) for QWK.
5-fold OOF on train + averaged test predictions. Saves OOF/test .npy so the
stacker can consume it as another base view.

Target: regression on label (1..5) with MSE; QWK thresholds optimized on OOF.
Input text: "venue | year | title. abstract" (abstract injected when available).
"""
import os, sys, json, warnings, gc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

BASE = r"D:/HK252/Data Mining/Assignment"
RAW = os.path.join(BASE, "data", "raw")
PRE = os.path.join(BASE, "data", "preprocessed")
OUT = os.path.join(BASE, "outputs")
SEED = 42
N_SPLITS = 5
MODEL_NAME = os.environ.get("FT_MODEL", "allenai/scibert_scivocab_uncased")
TAG = os.environ.get("FT_TAG", "scibert")
EPOCHS = int(os.environ.get("FT_EPOCHS", "4"))
MAXLEN = 320
BATCH = 16
LR = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TORCH_SEED = int(os.environ.get("FT_SEED", str(SEED)))
torch.manual_seed(TORCH_SEED); np.random.seed(TORCH_SEED)


def qwk(a, b):
    return cohen_kappa_score(a, b, weights="quadratic")

def apply_th(p, th):
    th = sorted(th); out = np.ones(len(p), int)
    for i, t in enumerate(th): out[p > t] = i + 2
    return out

def opt_th(yt, p):
    best = None
    for s in [(1.5, 2.5, 3.5, 4.5), (1.6, 2.4, 3.4, 4.3)]:
        r = minimize(lambda th: -qwk(yt, apply_th(p, th)), s, method="Nelder-Mead",
                     options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 3000})
        if best is None or -r.fun > best[1]:
            best = (sorted(r.x), -r.fun)
    return best[0]


def load():
    tr = pd.read_csv(os.path.join(RAW, "train.csv"))
    pub = pd.read_csv(os.path.join(RAW, "public_test.csv"))
    pri = pd.read_csv(os.path.join(RAW, "private_test.csv"))
    for d in (tr, pub, pri):
        d["title"] = d["title"].fillna(""); d["venue"] = d["venue"].fillna("na")
        d["authors"] = d["authors"].fillna("")
        d["year"] = pd.to_numeric(d["year"], errors="coerce").fillna(2020).astype(int)
    return tr, pub, pri


def build_text(df):
    ab = json.load(open(os.path.join(PRE, "stage2_abstracts_cache.json"), encoding="utf-8"))
    out = []
    for _, r in df.iterrows():
        a = ab.get(str(r["id"]), ""); a = a if isinstance(a, str) else ""
        head = f"venue: {r['venue']} | year: {r['year']} | {r['title']}"
        out.append(f"{head}. {a}" if len(a) > 20 else f"{head}.")
    return out


class DS(Dataset):
    def __init__(self, texts, tok, y=None):
        self.enc = tok(texts, padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt")
        self.y = y
    def __len__(self): return self.enc["input_ids"].shape[0]
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.y is not None: item["y"] = torch.tensor(self.y[i], dtype=torch.float)
        return item


class Reg(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(name)
        h = self.backbone.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(h, 128), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(128, 1))
    def forward(self, **kw):
        y = kw.pop("y", None)
        out = self.backbone(**kw).last_hidden_state[:, 0]
        return self.head(out).squeeze(-1)


def run_model(texts, ids):
    return None


def main():
    tr, pub, pri = load()
    y = tr["Label"].values.astype(float)
    df = pd.concat([tr, pub, pri], ignore_index=True)
    n_tr = len(tr); n_test = len(pub) + len(pri)
    texts = build_text(df)
    tr_texts = texts[:n_tr]; te_texts = texts[n_tr:]
    ids_test = df["id"].iloc[n_tr:].values
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(tr, tr["Label"]))
    oof = np.zeros(n_tr); test_pred = np.zeros(n_test)

    for fi, (trn, val) in enumerate(folds):
        print(f"\n=== fold {fi} ===", flush=True)
        model = Reg(MODEL_NAME).to(DEVICE)
        dl_tr = DataLoader(DS([tr_texts[i] for i in trn], tok, y[trn]), batch_size=BATCH, shuffle=True)
        dl_val = DataLoader(DS([tr_texts[i] for i in val], tok), batch_size=32)
        dl_te = DataLoader(DS(te_texts, tok), batch_size=32)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        steps = len(dl_tr) * EPOCHS
        sch = get_cosine_schedule_with_warmup(opt, int(0.1 * steps), steps)
        lossf = nn.MSELoss()
        for ep in range(EPOCHS):
            model.train()
            for batch in dl_tr:
                yb = batch.pop("y").to(DEVICE)
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                opt.zero_grad()
                pred = model(**batch)
                loss = lossf(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step()
            # quick val
            model.eval(); vp = []
            with torch.no_grad():
                for batch in dl_val:
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    vp.append(model(**batch).cpu().numpy())
            vp = np.concatenate(vp)
            print(f"  ep{ep} val-QWK(round)={qwk(tr['Label'].values[val], np.clip(np.round(vp),1,5)):.4f}", flush=True)
        # final fold predictions
        model.eval(); vp = []; tp = []
        with torch.no_grad():
            for batch in dl_val:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                vp.append(model(**batch).cpu().numpy())
            for batch in dl_te:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                tp.append(model(**batch).cpu().numpy())
        oof[val] = np.concatenate(vp)
        test_pred += np.concatenate(tp) / N_SPLITS
        del model; gc.collect(); torch.cuda.empty_cache()

    th = opt_th(tr["Label"].values, oof)
    print(f"\n=== {TAG} OOF-QWK thresh={qwk(tr['Label'].values, apply_th(oof, th)):.4f} "
          f"round={qwk(tr['Label'].values, np.clip(np.round(oof),1,5)):.4f} ===")
    print("thresholds", np.round(th, 4))
    np.save(os.path.join(OUT, f"ft_{TAG}_oof.npy"), oof)
    np.save(os.path.join(OUT, f"ft_{TAG}_test.npy"), test_pred)
    np.save(os.path.join(OUT, f"ft_{TAG}_ids_test.npy"), ids_test)
    print(f"saved ft_{TAG}_oof.npy / ft_{TAG}_test.npy")


if __name__ == "__main__":
    main()
