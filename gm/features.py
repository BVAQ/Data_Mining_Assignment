"""Stage 3: assemble ALL base features into aligned OOF (2494) + TEST (596) matrices.
Base signals:
  - 4 precomputed fine-tuned OOF/test (scibert, scibert_s2, specter2ft, specter2ft_s2)
  - dense emb (specter2_title, bge_title, specter2_titabs) -> ridge & logreg-EV OOF/test
  - title TFIDF logreg-EV
  - venue target-encoding, venue+yearbin TE  (CV-safe)
  - frontmatter flag
  - kNN-retrieval label (cosine over specter2_title) OOF/test
Saves gm/_feats.npz
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

EMB = 'gm/emb'
train = pd.read_parquet(f'{PRE}/s5_train.parquet')
public = pd.read_parquet(f'{PRE}/s5_public.parquet')
private = pd.read_parquet(f'{PRE}/s5_private.parquet')
sub = pd.read_csv(f'{RAW}/Test_Submission.csv')   # id order for final submission

y = train['Label'].values
N = len(train)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
# fixed fold assignment reused everywhere for consistency
fold = np.zeros(N, int)
for k, (_, vi) in enumerate(skf.split(np.arange(N), y)):
    fold[vi] = k
folds = [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(5)]

# ---- assemble test frame in submission order ----
test = pd.concat([public, private], ignore_index=True)
tmap = {int(i): j for j, i in enumerate(test['id'].values)}
order = [tmap[int(i)] for i in sub['id'].values]   # rows of `test` in submission order
test = test.iloc[order].reset_index(drop=True)
assert (test['id'].values == sub['id'].values).all()
test_venue = test['venue'].values
train_venue = train['venue'].values

feat_oof, feat_test, names = [], [], []

def add(name, oof, te):
    feat_oof.append(np.asarray(oof, float)); feat_test.append(np.asarray(te, float)); names.append(name)

# ---------- 1) precomputed FT models ----------
ids_tr = train['id'].values
for tag in ['ft_scibert','ft_scibert_s2','ft_specter2ft','ft_specter2ft_s2']:
    op, tp, ip = f'outputs/{tag}_oof.npy', f'outputs/{tag}_test.npy', f'outputs/{tag}_ids_test.npy'
    if not (os.path.exists(op) and os.path.exists(tp) and os.path.exists(ip)):
        print('skip FT', tag); continue
    oof = np.load(op); tst = np.load(tp); tids = np.load(ip)
    # align test preds to submission order
    m = {int(i): j for j, i in enumerate(tids)}
    te = np.array([tst[m[int(i)]] for i in sub['id'].values])
    add(tag, oof, te)
print('FT feats:', [n for n in names])

# ---------- 2) dense embedding heads ----------
def emb_head(tag, kind='ridge', alpha=10.0, C=2.0):
    Xtr = normalize(np.load(f'{EMB}/{tag}_train.npy'))
    Xpu = normalize(np.load(f'{EMB}/{tag}_public.npy'))
    Xpr = normalize(np.load(f'{EMB}/{tag}_private.npy'))
    # test in submission order
    ids_pub = np.load(f'{EMB}/ids_public.npy'); ids_pri = np.load(f'{EMB}/ids_private.npy')
    Xte_all = np.vstack([Xpu, Xpr]); ids_all = np.concatenate([ids_pub, ids_pri])
    mm = {int(i): j for j, i in enumerate(ids_all)}
    Xte = Xte_all[[mm[int(i)] for i in sub['id'].values]]
    oof = np.zeros(N); te_acc = np.zeros(len(sub))
    for tri, vai in folds:
        sc = StandardScaler().fit(Xtr[tri]); A, B = sc.transform(Xtr[tri]), sc.transform(Xtr[vai]); T = sc.transform(Xte)
        if kind == 'ridge':
            m_ = Ridge(alpha=alpha).fit(A, y[tri]); oof[vai] = m_.predict(B); te_acc += m_.predict(T) / 5
        else:
            m_ = LogisticRegression(C=C, max_iter=3000, class_weight='balanced').fit(A, y[tri])
            oof[vai] = (m_.predict_proba(B) * m_.classes_).sum(1)
            te_acc += (m_.predict_proba(T) * m_.classes_).sum(1) / 5
    return oof, te_acc

for tag in ['specter2_title','bge_title','specter2_titabs']:
    oof, te = emb_head(tag, 'ridge', alpha=10.0); add(f'{tag}_ridge', oof, te)
    oof2, te2 = emb_head(tag, 'logreg', C=2.0); add(f'{tag}_lrev', oof2, te2)

# ---------- 3) title TFIDF logreg EV ----------
ttr = train['title'].map(norm_text).values
tte = test['title'].map(norm_text).values
oof = np.zeros(N); te_acc = np.zeros(len(sub))
for tri, vai in folds:
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
    A = vec.fit_transform(ttr[tri]); B = vec.transform(ttr[vai]); T = vec.transform(tte)
    clf = LogisticRegression(C=2.0, max_iter=3000, class_weight='balanced').fit(A, y[tri])
    oof[vai] = (clf.predict_proba(B) * clf.classes_).sum(1)
    te_acc += (clf.predict_proba(T) * clf.classes_).sum(1) / 5
add('title_tfidf_lrev', oof, te_acc)

# ---------- 4) venue & venue+yearbin target encoding (CV-safe, smoothed) ----------
gm = y.mean()
def te_encode(keys_tr, keys_te, smooth=10):
    oof = np.zeros(N);
    for tri, vai in folds:
        df = pd.DataFrame({'k': keys_tr[tri], 'y': y[tri]})
        agg = df.groupby('k')['y'].agg(['sum','count'])
        enc = (agg['sum'] + smooth*gm) / (agg['count'] + smooth)
        oof[vai] = pd.Series(keys_tr[vai]).map(enc).fillna(gm).values
    df = pd.DataFrame({'k': keys_tr, 'y': y}); agg = df.groupby('k')['y'].agg(['sum','count'])
    enc = (agg['sum'] + smooth*gm) / (agg['count'] + smooth)
    te = pd.Series(keys_te).map(enc).fillna(gm).values
    return oof, te
oof, te = te_encode(train_venue, test_venue); add('te_venue', oof, te)
yb_tr = (train['year'].values // 2).astype(str); yb_te = (test['year'].values // 2).astype(str)
vy_tr = np.char.add(np.char.add(train_venue.astype(str), '_'), yb_tr)
vy_te = np.char.add(np.char.add(test_venue.astype(str), '_'), yb_te)
oof, te = te_encode(vy_tr, vy_te, smooth=15); add('te_venue_year', oof, te)

# ---------- 5) frontmatter deterministic flag ----------
def is_frontmatter(title):
    t = str(title).lower()
    if 'artificial intelligence in medicine' in t: return 0.0
    if re.search(r'\d+(st|nd|rd|th)\s+international\s+(conference|workshop|symposium)', t): return 1.0
    if 'proceedings' in t and ('workshop' in t or 'conference' in t or 'symposium' in t or 'co-located' in t): return 1.0
    if 'co-located with' in t: return 1.0
    return 0.0
add('frontmatter', train['title'].map(is_frontmatter).values, test['title'].map(is_frontmatter).values)

# ---------- 6) kNN retrieval label over specter2_title ----------
Xtr = normalize(np.load(f'{EMB}/specter2_title_train.npy'))
ids_pub = np.load(f'{EMB}/ids_public.npy'); ids_pri = np.load(f'{EMB}/ids_private.npy')
Xte_all = np.vstack([normalize(np.load(f'{EMB}/specter2_title_public.npy')),
                     normalize(np.load(f'{EMB}/specter2_title_private.npy'))])
ids_all = np.concatenate([ids_pub, ids_pri]); mm = {int(i): j for j, i in enumerate(ids_all)}
Xte = Xte_all[[mm[int(i)] for i in sub['id'].values]]
for K in [10, 25]:
    oof = np.zeros(N)
    for tri, vai in folds:
        S = Xtr[vai] @ Xtr[tri].T
        idx = np.argpartition(-S, K, axis=1)[:, :K]
        oof[vai] = y[tri][idx].mean(1)
    Ste = Xte @ Xtr.T; idx = np.argpartition(-Ste, K, axis=1)[:, :K]
    te = y[idx].mean(1)
    add(f'knn{K}', oof, te)

X_oof = np.vstack(feat_oof).T
X_test = np.vstack(feat_test).T
print('\nfeature matrix:', X_oof.shape, X_test.shape)
print('features:', names)
np.savez('gm/_feats.npz', X_oof=X_oof, X_test=X_test, y=y, names=np.array(names),
         train_venue=train_venue, test_venue=test_venue, sub_ids=sub['id'].values, fold=fold)
print('saved gm/_feats.npz')

# quick per-feature QWK
print('\nper-feature OOF QWK:')
for j, nm in enumerate(names):
    th, _ = opt_th(y, X_oof[:, j]); print(f'  {nm:22s} {qwk(y, apply_th(X_oof[:,j], th)):.4f}')
