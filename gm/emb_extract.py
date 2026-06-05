"""Stage 1: extract dense embeddings (title-only and title+abstract) from cached models.
Saves gm/emb/<tag>.npy aligned to ids in gm/emb/ids_{split}.npy.
Robust: skips a model if it errors; caches so reruns are cheap.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa  (utf-8, np, pd, norm_text)
import torch
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

OUT = 'gm/emb'
os.makedirs(OUT, exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', DEV)

train = pd.read_parquet(f'{PRE}/s5_train.parquet')
public = pd.read_parquet(f'{PRE}/s5_public.parquet')
private = pd.read_parquet(f'{PRE}/s5_private.parquet')
for df in (train, public, private):
    df['title_c'] = df['title'].map(norm_text)
    # raw (un-normalized) text is better for cased transformers; keep original casing
    df['abs_raw'] = df['abstract'].fillna('').astype(str)
    df['title_raw'] = df['title'].fillna('').astype(str)

splits = {'train': train, 'public': public, 'private': private}
for s, df in splits.items():
    np.save(f'{OUT}/ids_{s}.npy', df['id'].values)

def texts_for(df, variant):
    if variant == 'title':
        return df['title_raw'].tolist()
    if variant == 'titabs':
        # title [SEP] abstract — SPECTER-style. Use plain concat; tokenizer adds SEP.
        return [(t + '. ' + a).strip() for t, a in zip(df['title_raw'], df['abs_raw'])]
    raise ValueError(variant)

@torch.no_grad()
def encode_hf(model_name, texts, max_len=320, bs=16, pool='mean'):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mdl = AutoModel.from_pretrained(model_name, local_files_only=True).to(DEV).eval()
    if DEV == 'cuda':
        mdl = mdl.half()
    out = []
    for i in range(0, len(texts), bs):
        b = texts[i:i+bs]
        enc = tok(b, padding=True, truncation=True, max_length=max_len, return_tensors='pt').to(DEV)
        o = mdl(**enc)
        last = o.last_hidden_state  # (B,T,H)
        if pool == 'cls':
            emb = last[:, 0]
        else:
            mask = enc['attention_mask'].unsqueeze(-1).type_as(last)
            emb = (last * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        out.append(emb.float().cpu().numpy())
    del mdl
    if DEV == 'cuda':
        torch.cuda.empty_cache()
    return np.concatenate(out, 0)

# (model_name, tag, variant, pool, max_len)
JOBS = [
    ('allenai/specter2_base', 'specter2_titabs', 'titabs', 'cls', 384),
    ('allenai/specter2_base', 'specter2_title',  'title',  'cls', 64),
    ('BAAI/bge-base-en-v1.5', 'bge_titabs',      'titabs', 'cls', 384),
    ('BAAI/bge-base-en-v1.5', 'bge_title',       'title',  'cls', 64),
    ('sentence-transformers/all-mpnet-base-v2', 'mpnet_titabs', 'titabs', 'mean', 384),
    ('allenai/scibert_scivocab_uncased', 'scibert_titabs', 'titabs', 'mean', 384),
]

for model_name, tag, variant, pool, ml in JOBS:
    done = all(os.path.exists(f'{OUT}/{tag}_{s}.npy') for s in splits)
    if done:
        print('skip (cached):', tag); continue
    try:
        for s, df in splits.items():
            E = encode_hf(model_name, texts_for(df, variant), max_len=ml, pool=pool)
            np.save(f'{OUT}/{tag}_{s}.npy', E)
        print('OK', tag, 'dim', E.shape[1])
    except Exception as e:
        print('FAIL', tag, type(e).__name__, str(e)[:160])
print('done embeddings')
