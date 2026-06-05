"""
Stage 2 (new): cache frozen scientific-text embeddings for train+test.
Outputs .npy matrices aligned to the row order of train / pub / pri concatenation.
We embed `title` and `title + abstract` separately with two models:
  - allenai/specter2_base  (scientific paper embeddings)
  - BAAI/bge-small-en-v1.5 (strong general sentence embedding)
Each row's vector is L2-normalized. Saved to outputs/emb/.
"""
import os, sys, json
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"D:/HK252/Data Mining/Assignment"
RAW = os.path.join(BASE, "data", "raw")
PRE = os.path.join(BASE, "data", "preprocessed")
OUT = os.path.join(BASE, "outputs", "emb")
os.makedirs(OUT, exist_ok=True)


def load_all():
    tr = pd.read_csv(os.path.join(RAW, "train.csv"))
    pub = pd.read_csv(os.path.join(RAW, "public_test.csv"))
    pri = pd.read_csv(os.path.join(RAW, "private_test.csv"))
    for d in (tr, pub, pri):
        d["title"] = d["title"].fillna("")
        d["venue"] = d["venue"].fillna("na")
        d["authors"] = d["authors"].fillna("")
    return tr, pub, pri


def get_abstracts(ids):
    ab = json.load(open(os.path.join(PRE, "stage2_abstracts_cache.json"), encoding="utf-8"))
    out = []
    for i in ids:
        v = ab.get(str(i), "")
        out.append(v if isinstance(v, str) else "")
    return out


def embed(model_name, texts, batch=64, max_len=320):
    import torch
    from transformers import AutoTokenizer, AutoModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
    vecs = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            chunk = texts[s:s + batch]
            enc = tok(chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
            out = mdl(**enc)
            # CLS token (specter2 convention) ; bge also fine with CLS pooling
            emb = out.last_hidden_state[:, 0]
            emb = torch.nn.functional.normalize(emb, dim=1)
            vecs.append(emb.cpu().numpy().astype("float32"))
            if s % (batch * 10) == 0:
                print(f"  {model_name.split('/')[-1]}: {s}/{len(texts)}", flush=True)
    return np.vstack(vecs)


def main():
    tr, pub, pri = load_all()
    df = pd.concat([tr, pub, pri], ignore_index=True)
    n_tr, n_pub, n_pri = len(tr), len(pub), len(pri)
    print(f"rows tr={n_tr} pub={n_pub} pri={n_pri} total={len(df)}")
    abstracts = get_abstracts(df["id"].tolist())

    title_txt = [f"{t}" for t in df["title"]]
    ta_txt = []
    for t, a in zip(df["title"], abstracts):
        ta_txt.append(f"{t}. {a}" if len(a) > 20 else f"{t}")

    jobs = [
        ("allenai/specter2_base", "specter2", title_txt, "title"),
        ("allenai/specter2_base", "specter2", ta_txt, "titleabs"),
        ("BAAI/bge-small-en-v1.5", "bge", title_txt, "title"),
        ("BAAI/bge-small-en-v1.5", "bge", ta_txt, "titleabs"),
    ]
    for model_name, tag, texts, field in jobs:
        fn = os.path.join(OUT, f"{tag}_{field}.npy")
        if os.path.exists(fn):
            print("skip exists", fn)
            continue
        print("embedding", model_name, field, flush=True)
        m = embed(model_name, texts)
        np.save(fn, m)
        print("saved", fn, m.shape)
    # save split sizes
    json.dump({"n_tr": n_tr, "n_pub": n_pub, "n_pri": n_pri},
              open(os.path.join(OUT, "splits.json"), "w"))
    print("done")


if __name__ == "__main__":
    main()
