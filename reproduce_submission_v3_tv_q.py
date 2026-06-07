"""
Reproduce the selected final submission: outputs/submission_v3_tv_q.csv.

This is the artifact-based v3 reproduction script. It is designed to reproduce
the exact selected leaderboard submission deterministically from:

    1. raw CSV files in data/raw/
    2. saved model-prediction artifacts produced during experimentation

Default behavior is fast and exact:

    python reproduce_submission_v3_tv_q.py

It loads outputs/rebuild_v3_oof.npz, applies the same tv_q calibration and
frontmatter post-processing, writes outputs/reproduced_submission_v3_tv_q.csv,
and verifies exact agreement with outputs/submission_v3_tv_q.csv.

Optional audit rebuild:

    python reproduce_submission_v3_tv_q.py --rebuild-stack

This rebuilds the final v3 stack from cached base predictions such as
gm/_feats.npz, gm/ft/ft2_*_*.npy, outputs/s2new_oof.npz, and fresh TF-IDF
Ridge features. It is slower because it repeats cross-validation and threshold
searches, but it does not retrain the expensive transformer base models.

Important:
    The exact historical final submission cannot be reproduced from raw CSV
    files alone, because several base predictions came from transformer and
    embedding models trained or extracted earlier. Those intermediate
    predictions are the saved artifacts used here.
"""
from __future__ import annotations

import argparse
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


SEED = 42
N_SPLITS = 5
TEST_VENUES = {"cav", "lics", "kr", "lpnmr"}
EXCLUDE_EXTRA_FT = {"ft2_scibert_venue_titabs", "ft2_deberta_venue_title"}


def project_root() -> Path:
    return Path(__file__).resolve().parent


def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def apply_thresholds(scores: np.ndarray, thresholds: list[float] | np.ndarray) -> np.ndarray:
    thresholds = sorted(float(t) for t in thresholds)
    labels = np.ones(len(scores), dtype=int)
    for i, threshold in enumerate(thresholds):
        labels[scores > threshold] = i + 2
    return labels


def coord_ascent_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_round: int = 5,
    grid_size: int = 600,
) -> tuple[list[float], float]:
    """Coordinate-search four cut points that maximize OOF QWK."""
    lo, hi = float(scores.min() - 0.1), float(scores.max() + 0.1)
    best_thresholds: list[float] | None = None
    best_score = -1.0
    initial_quantiles = [
        [0.30, 0.50, 0.70, 0.85],
        [0.35, 0.55, 0.73, 0.88],
        [0.36, 0.57, 0.74, 0.89],
        [0.38, 0.60, 0.76, 0.90],
    ]

    for init_q in initial_quantiles:
        thresholds = list(np.quantile(scores, init_q))
        grid = np.linspace(lo, hi, grid_size)
        local_best = qwk(y_true, apply_thresholds(scores, thresholds))
        for _ in range(n_round):
            improved = False
            for i in range(4):
                for grid_value in grid:
                    candidate = thresholds.copy()
                    candidate[i] = float(grid_value)
                    candidate = sorted(candidate)
                    if not candidate[0] < candidate[1] < candidate[2] < candidate[3]:
                        continue
                    score = qwk(y_true, apply_thresholds(scores, candidate))
                    if score > local_best:
                        local_best = score
                        thresholds = candidate
                        improved = True
            if not improved:
                break
        if local_best > best_score:
            best_score = local_best
            best_thresholds = sorted(thresholds)

    if best_thresholds is None:
        raise RuntimeError("Threshold optimization failed.")
    return best_thresholds, best_score


def quantile_calibrate(scores: np.ndarray, target_dist: np.ndarray) -> np.ndarray:
    """Map continuous scores to labels so the predicted marginal follows target_dist."""
    cumulative = np.cumsum(np.asarray(target_dist, dtype=float))
    ranks = pd.Series(scores).rank(pct=True).values
    labels = np.ones(len(ranks), dtype=int)
    for i, rank in enumerate(ranks):
        labels[i] = 1 + int(np.searchsorted(cumulative, rank, side="right"))
    return np.clip(labels, 1, 5)


def detect_frontmatter(titles: pd.Series) -> np.ndarray:
    patterns = [
        r"proceedings?\s+of\s+the",
        r"foreword",
        r"preface",
        r"\d+(?:st|nd|rd|th)\s+international\s+(?:conference|workshop|symposium)",
        r"editorial",
        r"table\s+of\s+contents",
    ]
    joined = "|".join(patterns)
    return titles.fillna("").str.lower().str.contains(joined, regex=True, na=False).values


def require_file(path: Path, purpose: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {purpose}: {path}")


def load_raw(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = root / "data" / "raw"
    paths = {
        "train": raw / "train.csv",
        "public": raw / "public_test.csv",
        "private": raw / "private_test.csv",
        "sample": raw / "Test_Submission.csv",
    }
    for name, path in paths.items():
        require_file(path, name)

    train = pd.read_csv(paths["train"])
    public = pd.read_csv(paths["public"])
    private = pd.read_csv(paths["private"])
    sample = pd.read_csv(paths["sample"])
    test = pd.concat([public, private], ignore_index=True)
    return train, public, private, sample, test


def prepare_frames(*frames: pd.DataFrame) -> None:
    """Apply the fill/normalization used by the final v3 stack."""
    for df in frames:
        df["title"] = df["title"].fillna("")
        df["venue"] = df["venue"].fillna("unknown")
        if "authors" in df.columns:
            df["authors"] = df["authors"].fillna("")
        if "doi" in df.columns:
            df["doi"] = df["doi"].fillna("")
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce")


def print_data_checks(
    train: pd.DataFrame,
    public: pd.DataFrame,
    private: pd.DataFrame,
    sample: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    print("\nDATA VALIDATION")
    print("-" * 80)
    for name, df in [
        ("train", train),
        ("public_test", public),
        ("private_test", private),
        ("Test_Submission", sample),
    ]:
        print(f"{name:16s} shape={df.shape} duplicate_ids={int(df['id'].duplicated().sum())}")
        print(f"{name:16s} missing={df.isna().sum().to_dict()}")

    counts = train["Label"].value_counts().sort_index()
    pct = (counts / len(train) * 100).round(2)
    print("\nTrain label distribution:")
    for label in range(1, 6):
        print(f"  Label {label}: {int(counts.get(label, 0)):4d} ({pct.get(label, 0):5.2f}%)")

    test_id_order = test["id"].values
    sample_id_order = sample["id"].values
    print(f"\nPublic+private order matches Test_Submission: {bool(np.array_equal(test_id_order, sample_id_order))}")
    print(f"Train venues: {sorted(train['venue'].unique().tolist())}")
    print(f"Test venues:  {sorted(test['venue'].unique().tolist())}")


def load_cached_base_features(root: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load base OOF/test feature matrix used by the selected v3 stack."""
    gm_path = root / "gm" / "_feats.npz"
    s2_path = root / "outputs" / "s2new_oof.npz"
    require_file(gm_path, "GM feature artifact")
    require_file(s2_path, "Stage-2 stack artifact")

    gm = np.load(gm_path, allow_pickle=True)
    X_oof = gm["X_oof"].copy()
    X_test = gm["X_test"].copy()
    names = [str(x) for x in gm["names"]]

    for oof_path_str in sorted(glob.glob(str(root / "gm" / "ft" / "ft2_*_oof.npy"))):
        oof_path = Path(oof_path_str)
        tag = oof_path.name.replace("_oof.npy", "")
        test_path = root / "gm" / "ft" / f"{tag}_test.npy"
        if tag in EXCLUDE_EXTRA_FT or not test_path.exists():
            continue
        X_oof = np.column_stack([X_oof, np.load(oof_path)])
        X_test = np.column_stack([X_test, np.load(test_path)])
        names.append(tag)

    s2 = np.load(s2_path, allow_pickle=True)
    X_oof = np.column_stack([X_oof, s2["meta_oof"]])
    X_test = np.column_stack([X_test, s2["meta_te"]])
    names.append("s2_meta")

    return X_oof, X_test, names


def add_tfidf_ridge_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    X_oof: np.ndarray,
    X_test: np.ndarray,
    names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(train, y))

    print("\nRebuilding cheap TF-IDF Ridge features:")
    for alpha in [3.0]:
        oof_word = np.zeros(len(train))
        test_word = np.zeros(len(test))
        for train_idx, val_idx in folds:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
            )
            vectorizer.fit(train["title"].iloc[train_idx])
            model = Ridge(alpha=alpha).fit(vectorizer.transform(train["title"].iloc[train_idx]), y[train_idx])
            oof_word[val_idx] = model.predict(vectorizer.transform(train["title"].iloc[val_idx]))
            test_word += model.predict(vectorizer.transform(test["title"])) / N_SPLITS
        X_oof = np.column_stack([X_oof, oof_word])
        X_test = np.column_stack([X_test, test_word])
        names.append(f"tfidf_word_a{alpha}")
        _, score = coord_ascent_thresholds(y, oof_word)
        print(f"  tfidf_word_a{alpha}: OOF QWK={score:.4f}")

    for alpha in [3.0]:
        oof_char = np.zeros(len(train))
        test_char = np.zeros(len(test))
        for train_idx, val_idx in folds:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 6),
                min_df=3,
                max_features=50000,
                sublinear_tf=True,
            )
            vectorizer.fit(train["title"].iloc[train_idx])
            model = Ridge(alpha=alpha).fit(vectorizer.transform(train["title"].iloc[train_idx]), y[train_idx])
            oof_char[val_idx] = model.predict(vectorizer.transform(train["title"].iloc[val_idx]))
            test_char += model.predict(vectorizer.transform(test["title"])) / N_SPLITS
        X_oof = np.column_stack([X_oof, oof_char])
        X_test = np.column_stack([X_test, test_char])
        names.append(f"tfidf_char_a{alpha}")
        _, score = coord_ascent_thresholds(y, oof_char)
        print(f"  tfidf_char_a{alpha}: OOF QWK={score:.4f}")

    oof_combined = np.zeros(len(train))
    test_combined = np.zeros(len(test))
    for train_idx, val_idx in folds:
        word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=2,
            max_features=60000,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            min_df=3,
            max_features=60000,
            sublinear_tf=True,
        )
        word_vectorizer.fit(train["title"].iloc[train_idx])
        char_vectorizer.fit(train["title"].iloc[train_idx])
        X_train = sp.hstack(
            [
                word_vectorizer.transform(train["title"].iloc[train_idx]),
                char_vectorizer.transform(train["title"].iloc[train_idx]),
            ]
        )
        X_val = sp.hstack(
            [
                word_vectorizer.transform(train["title"].iloc[val_idx]),
                char_vectorizer.transform(train["title"].iloc[val_idx]),
            ]
        )
        X_t = sp.hstack([word_vectorizer.transform(test["title"]), char_vectorizer.transform(test["title"])])
        model = Ridge(alpha=3.0).fit(X_train, y[train_idx])
        oof_combined[val_idx] = model.predict(X_val)
        test_combined += model.predict(X_t) / N_SPLITS
    X_oof = np.column_stack([X_oof, oof_combined])
    X_test = np.column_stack([X_test, test_combined])
    names.append("tfidf_combined")
    _, score = coord_ascent_thresholds(y, oof_combined)
    print(f"  tfidf_combined: OOF QWK={score:.4f}")

    return X_oof, X_test, names


def train_multiseed_stack(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: np.ndarray,
    tv_mask: np.ndarray,
    X_oof: np.ndarray,
    X_test: np.ndarray,
) -> dict:
    configs: list[dict] = []
    print("\nMulti-seed Ridge stack:")
    for alpha in [0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        seed_oofs = []
        seed_tests = []
        seed_qwk_all = []
        seed_qwk_tv = []

        for seed in [42, 123, 456, 789, 1234]:
            folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(train, y))
            oof = np.zeros(len(train))
            test_pred = np.zeros(len(test))

            for train_idx, val_idx in folds:
                scaler = StandardScaler().fit(X_oof[train_idx])
                model = Ridge(alpha=alpha, random_state=seed).fit(scaler.transform(X_oof[train_idx]), y[train_idx])
                oof[val_idx] = model.predict(scaler.transform(X_oof[val_idx]))
                test_pred += model.predict(scaler.transform(X_test)) / N_SPLITS

            _, q_all = coord_ascent_thresholds(y, oof)
            _, q_tv = coord_ascent_thresholds(y[tv_mask], oof[tv_mask])
            seed_oofs.append(oof)
            seed_tests.append(test_pred)
            seed_qwk_all.append(q_all)
            seed_qwk_tv.append(q_tv)

        avg_oof = np.mean(seed_oofs, axis=0)
        avg_test = np.mean(seed_tests, axis=0)
        thresholds_all, q_avg_all = coord_ascent_thresholds(y, avg_oof)
        thresholds_tv, q_avg_tv = coord_ascent_thresholds(y[tv_mask], avg_oof[tv_mask])

        print(
            f"  alpha={alpha:4.1f} "
            f"per-seed all={np.mean(seed_qwk_all):.4f}±{np.std(seed_qwk_all):.4f}, "
            f"tv={np.mean(seed_qwk_tv):.4f}±{np.std(seed_qwk_tv):.4f}; "
            f"avg all={q_avg_all:.4f}, tv={q_avg_tv:.4f}"
        )
        configs.append(
            {
                "alpha": alpha,
                "avg_oof": avg_oof,
                "avg_test": avg_test,
                "q_avg_all": q_avg_all,
                "q_avg_tv": q_avg_tv,
                "std_tv": float(np.std(seed_qwk_tv)),
                "thresholds_all": thresholds_all,
                "thresholds_tv": thresholds_tv,
            }
        )

    configs.sort(key=lambda item: item["q_avg_tv"], reverse=True)
    return configs[0]


def build_submission(
    sample: pd.DataFrame,
    test_scores: np.ndarray,
    y: np.ndarray,
    tv_mask: np.ndarray,
    frontmatter_test: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    tv_dist = np.bincount(y[tv_mask].astype(int), minlength=6)[1:] / int(tv_mask.sum())
    labels = quantile_calibrate(test_scores, tv_dist)
    labels[frontmatter_test] = 1
    submission = pd.DataFrame({"id": sample["id"].values, "Label": labels.astype(int)})
    return submission, tv_dist


def print_final_submission_summary(submission: pd.DataFrame, output_path: Path, tv_dist: np.ndarray) -> None:
    print("\nFINAL SUBMISSION RESULT")
    print("-" * 80)
    print(f"Output file: {output_path}")
    print(f"Rows: {len(submission)}")
    print(f"Columns: {list(submission.columns)}")
    print(f"Valid labels 1..5: {bool(submission['Label'].between(1, 5).all())}")
    print("\nFirst 10 rows:")
    print(submission.head(10).to_string(index=False))
    print("\nPrediction distribution:")
    counts = submission["Label"].value_counts().sort_index()
    for label in range(1, 6):
        count = int(counts.get(label, 0))
        pct = 100 * count / len(submission)
        print(f"  Label {label}: {count:4d} ({pct:5.2f}%)")
    print(f"\nTV marginal calibration target: {np.round(tv_dist, 4).tolist()}")
    print("\nRecorded leaderboard result for selected submission_v3_tv_q.csv:")
    print("  Public QWK : 0.75557")
    print("  Private QWK: 0.73479")


def compare_with_expected(output: pd.DataFrame, expected_path: Path | None) -> None:
    if expected_path is None:
        return
    if not expected_path.exists():
        print(f"\nExpected selected submission not found, skipped comparison: {expected_path}")
        return
    expected = pd.read_csv(expected_path)
    same_ids = bool(np.array_equal(output["id"].values, expected["id"].values))
    same_labels = bool(np.array_equal(output["Label"].values, expected["Label"].values))
    agreement = float((output["Label"].values == expected["Label"].values).mean()) if same_ids else float("nan")
    print("\nComparison with selected final CSV:")
    print(f"  expected file: {expected_path}")
    print(f"  id order match: {same_ids}")
    print(f"  exact label match: {same_labels}")
    print(f"  label agreement: {agreement:.4f}")
    if same_ids and not same_labels:
        diff = int((output["Label"].values != expected["Label"].values).sum())
        print(f"  differing rows: {diff}")


def write_all_calibration_strategies(
    root: Path,
    sample: pd.DataFrame,
    best: dict,
    y: np.ndarray,
    tv_mask: np.ndarray,
    frontmatter_test: np.ndarray,
) -> None:
    train_dist = np.bincount(y.astype(int), minlength=6)[1:] / len(y)
    tv_dist = np.bincount(y[tv_mask].astype(int), minlength=6)[1:] / int(tv_mask.sum())
    winning_dist = np.array([0.409, 0.174, 0.159, 0.094, 0.163])
    strategies = {
        "opt_all": apply_thresholds(best["avg_test"], best["thresholds_all"]),
        "opt_tv": apply_thresholds(best["avg_test"], best["thresholds_tv"]),
        "train_q": quantile_calibrate(best["avg_test"], train_dist),
        "tv_q": quantile_calibrate(best["avg_test"], tv_dist),
        "winning_q": quantile_calibrate(best["avg_test"], winning_dist),
    }
    for alpha_blend in [0.3, 0.5, 0.7]:
        blend = alpha_blend * winning_dist + (1 - alpha_blend) * tv_dist
        strategies[f"blend_{alpha_blend}"] = quantile_calibrate(best["avg_test"], blend / blend.sum())

    out_dir = root / "outputs"
    for name, labels in strategies.items():
        labels = labels.copy()
        labels[frontmatter_test] = 1
        pd.DataFrame({"id": sample["id"].values, "Label": labels.astype(int)}).to_csv(
            out_dir / f"reproduced_submission_v3_{name}.csv",
            index=False,
        )


def reproduce_from_saved_v3_artifact(
    root: Path,
    sample: pd.DataFrame,
    y: np.ndarray,
    computed_tv_mask: np.ndarray,
    computed_frontmatter_test: np.ndarray,
    output_path: Path,
    expected_path: Path | None,
) -> None:
    artifact_path = root / "outputs" / "rebuild_v3_oof.npz"
    require_file(artifact_path, "final v3 OOF/test artifact")
    artifact = np.load(artifact_path, allow_pickle=True)

    saved_y = artifact["y"]
    saved_tv_mask = artifact["tv_mask"].astype(bool)
    saved_fm_test = artifact["fm_test"].astype(bool)
    saved_ids = artifact["sub_ids"]
    if not np.array_equal(saved_y, y):
        raise ValueError("Saved v3 artifact target labels do not match train.csv.")
    if not np.array_equal(saved_tv_mask, computed_tv_mask):
        raise ValueError("Saved v3 artifact test-venue mask does not match train.csv.")
    if not np.array_equal(saved_fm_test, computed_frontmatter_test):
        raise ValueError("Saved v3 artifact frontmatter mask does not match current test data.")
    if not np.array_equal(saved_ids, sample["id"].values):
        raise ValueError("Saved v3 artifact ids do not match Test_Submission.csv.")

    submission, tv_dist = build_submission(sample, artifact["test"], saved_y, saved_tv_mask, saved_fm_test)
    assert len(submission) == len(sample) == 596
    assert submission["Label"].between(1, 5).all()
    submission.to_csv(output_path, index=False)

    feature_names = [str(x) for x in artifact["feature_names"]]
    print("\nReproduced from saved final v3 artifact:")
    print(f"  artifact: {artifact_path}")
    print(f"  selected stack alpha: {artifact['alpha'].item() if hasattr(artifact['alpha'], 'item') else artifact['alpha']}")
    print(f"  thresholds_all: {np.round(artifact['th'], 6).tolist()}")
    print(f"  thresholds_tv:  {np.round(artifact['th_tv'], 6).tolist()}")
    print(f"  feature count: {len(feature_names)}")
    print(f"  feature names: {feature_names}")
    print_final_submission_summary(submission, output_path, tv_dist)
    compare_with_expected(submission, expected_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="outputs/reproduced_submission_v3_tv_q.csv",
        help="Output CSV path relative to project root unless absolute.",
    )
    parser.add_argument(
        "--expected",
        default=None,
        help=(
            "Optional existing selected final CSV used for exact comparison. "
            "Example: --expected outputs/submission_v3_tv_q.csv"
        ),
    )
    parser.add_argument(
        "--write-all-strategies",
        action="store_true",
        help="Also write reproduced calibration variants for documentation.",
    )
    parser.add_argument(
        "--rebuild-stack",
        action="store_true",
        help=(
            "Recompute the v3 stack from cached base predictions and fresh TF-IDF features. "
            "Default is the faster exact reproduction from outputs/rebuild_v3_oof.npz."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    start = time.time()

    print("=" * 80)
    print("REPRODUCE submission_v3_tv_q.csv")
    print("=" * 80)
    print(f"Project root: {root}")

    train, public, private, sample, test = load_raw(root)
    print_data_checks(train, public, private, sample, test)
    prepare_frames(train, public, private, test)

    y = train["Label"].values
    tv_mask = train["venue"].isin(TEST_VENUES).values
    frontmatter_train = detect_frontmatter(train["title"])
    frontmatter_test = detect_frontmatter(test["title"])
    print(f"\nTest-venue train rows: {int(tv_mask.sum())}/{len(train)}")
    print(f"Frontmatter rows: train={int(frontmatter_train.sum())}, test={int(frontmatter_test.sum())}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_path = Path(args.expected) if args.expected else None
    if expected_path is not None and not expected_path.is_absolute():
        expected_path = root / expected_path

    if not args.rebuild_stack:
        reproduce_from_saved_v3_artifact(
            root,
            sample,
            y,
            tv_mask,
            frontmatter_test,
            output_path,
            expected_path,
        )
        print(f"\nComplete in {time.time() - start:.1f}s")
        print("\nFor a slower rebuild from cached base predictions, run with --rebuild-stack.")
        return

    print("\nLoading cached base predictions:")
    X_oof, X_test, feature_names = load_cached_base_features(root)
    print(f"  cached matrix: X_oof={X_oof.shape}, X_test={X_test.shape}")
    print(f"  cached/base names: {feature_names}")

    X_oof, X_test, feature_names = add_tfidf_ridge_features(train, test, y, X_oof, X_test, feature_names)
    X_oof = np.nan_to_num(X_oof, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"\nFinal stack feature matrix: X_oof={X_oof.shape}, X_test={X_test.shape}")
    print(f"Final feature names ({len(feature_names)}): {feature_names}")

    best = train_multiseed_stack(train, test, y, tv_mask, X_oof, X_test)
    print("\nSelected stack configuration:")
    print(f"  alpha={best['alpha']}")
    print(f"  OOF QWK all={best['q_avg_all']:.4f}")
    print(f"  OOF QWK test-venue subset={best['q_avg_tv']:.4f}")
    print(f"  thresholds_all={np.round(best['thresholds_all'], 6).tolist()}")
    print(f"  thresholds_tv={np.round(best['thresholds_tv'], 6).tolist()}")

    submission, tv_dist = build_submission(sample, best["avg_test"], y, tv_mask, frontmatter_test)
    assert len(submission) == len(sample) == 596
    assert np.array_equal(submission["id"].values, sample["id"].values)
    assert submission["Label"].between(1, 5).all()

    submission.to_csv(output_path, index=False)
    print_final_submission_summary(submission, output_path, tv_dist)
    compare_with_expected(submission, expected_path)

    if args.write_all_strategies:
        write_all_calibration_strategies(root, sample, best, y, tv_mask, frontmatter_test)
        print("Wrote reproduced calibration variants to outputs/reproduced_submission_v3_*.csv")

    print(f"\nComplete in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
