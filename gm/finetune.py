"""Stage 5: fine-tune a transformer with REGRESSION (MSE) objective for QWK.
Configurable. Produces 5-fold OOF + full-train test preds (submission order).
Input variants:
  title        : title only
  venue_title  : "venue: <venue>. <title>"  (gives model the strong venue signal)
Saves gm/ft/<tag>_oof.npy, <tag>_test.npy (aligned to Test_Submission id order).
Usage: python gm/finetune.py MODEL_KEY VARIANT EPOCHS SEEDS
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
import logging, warnings
logging.getLogger('transformers').setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
import transformers as _tf
try: _tf.logging.set_verbosity_error()
except Exception: pass
from transformers import AutoTokenizer, AutoModel

MODELS = {
    'scibert': 'allenai/scibert_scivocab_uncased',
    'deberta': 'microsoft/deberta-v3-base',
    'specter2': 'allenai/specter2_base',
    'bge': 'BAAI/bge-base-en-v1.5',
}
mkey = sys.argv[1] if len(sys.argv)>1 else 'scibert'
variant = sys.argv[2] if len(sys.argv)>2 else 'venue_title'
EPOCHS = int(sys.argv[3]) if len(sys.argv)>3 else 4
SEEDS = [int(x) for x in (sys.argv[4].split(',') if len(sys.argv)>4 else ['0'])]
MODEL = MODELS[mkey]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
MAXLEN = {'title': 64, 'venue_title': 96, 'venue_titabs': 192}[variant]
print(f'FT model={mkey} variant={variant} epochs={EPOCHS} seeds={SEEDS} maxlen={MAXLEN} dev={DEV}')

train = pd.read_parquet(f'{PRE}/s5_train.parquet')
public = pd.read_parquet(f'{PRE}/s5_public.parquet')
private = pd.read_parquet(f'{PRE}/s5_private.parquet')
sub = pd.read_csv(f'{RAW}/Test_Submission.csv')
test = pd.concat([public, private], ignore_index=True)
tmap = {int(i):j for j,i in enumerate(test['id'].values)}
test = test.iloc[[tmap[int(i)] for i in sub['id'].values]].reset_index(drop=True)
y = train['Label'].values.astype('float32')

def mk_text(df):
    if variant=='title':
        return df['title'].fillna('').astype(str).tolist()
    if variant=='venue_titabs':
        out=[]
        for v,t,a in zip(df['venue'].fillna('').astype(str), df['title'].fillna('').astype(str),
                         df['abstract'].fillna('').astype(str)):
            out.append(f"venue: {v}. {t}. {a}".strip())
        return out
    return [f"venue: {v}. {t}" for v,t in zip(df['venue'].fillna('').astype(str), df['title'].fillna('').astype(str))]
tr_text = mk_text(train); te_text = mk_text(test)

tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
class DS(Dataset):
    def __init__(self, texts, labels=None):
        self.t=texts; self.y=labels
    def __len__(self): return len(self.t)
    def __getitem__(self, i):
        return self.t[i], (0.0 if self.y is None else float(self.y[i]))
def collate(batch):
    txts=[b[0] for b in batch]; ys=torch.tensor([b[1] for b in batch],dtype=torch.float32)
    enc=tok(txts, padding=True, truncation=True, max_length=MAXLEN, return_tensors='pt')
    return enc, ys

class Reg(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.enc=AutoModel.from_pretrained(name, local_files_only=True)
        h=self.enc.config.hidden_size
        self.head=nn.Sequential(nn.Dropout(0.1), nn.Linear(h,1))
    def forward(self, enc):
        o=self.enc(**enc)
        x=o.last_hidden_state[:,0]            # CLS
        return self.head(x).squeeze(-1)

def train_one(tr_idx, seed, X_text, ylab):
    torch.manual_seed(seed); np.random.seed(seed)
    mdl=Reg(MODEL).to(DEV).float()   # ensure FP32 master weights (some models load FP16); AMP handles autocast
    bs = 8 if MAXLEN >= 256 else 16
    dl=DataLoader(DS([X_text[i] for i in tr_idx], ylab[tr_idx]), batch_size=bs, shuffle=True, collate_fn=collate)
    opt=torch.optim.AdamW(mdl.parameters(), lr=2e-5, weight_decay=0.01)
    scaler=torch.amp.GradScaler('cuda', enabled=(DEV=='cuda'))
    lossf=nn.MSELoss()
    steps=len(dl)*EPOCHS
    sched=torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-5, total_steps=steps, pct_start=0.1)
    mdl.train()
    for ep in range(EPOCHS):
        for enc,ys in dl:
            enc={k:v.to(DEV) for k,v in enc.items()}; ys=ys.to(DEV)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(DEV=='cuda')):
                pred=mdl(enc); loss=lossf(pred, ys)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    return mdl

@torch.no_grad()
def predict(mdl, X_text):
    mdl.eval(); out=[]
    dl=DataLoader(DS(X_text), batch_size=32, shuffle=False, collate_fn=collate)
    for enc,_ in dl:
        enc={k:v.to(DEV) for k,v in enc.items()}
        with torch.amp.autocast('cuda', enabled=(DEV=='cuda')):
            out.append(mdl(enc).float().cpu().numpy())
    return np.concatenate(out)

skf=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
ylab=y
oof=np.zeros(len(train)); te=np.zeros(len(sub))
for k,(tri,vai) in enumerate(skf.split(tr_text, train['Label'].values)):
    pv=np.zeros(len(vai)); pt=np.zeros(len(sub))
    for s in SEEDS:
        mdl=train_one(tri, s, tr_text, ylab)
        pv += predict(mdl, [tr_text[i] for i in vai])/len(SEEDS)
        pt += predict(mdl, te_text)/len(SEEDS)
        del mdl; torch.cuda.empty_cache()
    oof[vai]=pv; te+=pt/5
    th,_=opt_th(train['Label'].values[:len(oof)][vai] if False else train['Label'].values, oof) # placeholder
    print(f'  fold{k} done')

th,q=opt_th(train['Label'].values, oof)
tag=f'ft2_{mkey}_{variant}'
os.makedirs('gm/ft', exist_ok=True)
np.save(f'gm/ft/{tag}_oof.npy', oof); np.save(f'gm/ft/{tag}_test.npy', te)
print(f'{tag}: OOF QWK={q:.4f}  th={th.round(3)}')
print('saved gm/ft/'+tag)
