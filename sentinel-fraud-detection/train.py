"""Train the Sentinel fraud detection model."""

import datetime as dt
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sentinel.datagen import make_dataset
from sentinel.features import (
    CAT_FEATURES, FEATURE_LABELS, FEATURES, NUM_FEATURES,
    build_features, validate_and_clean
)
from sentinel.model import METRICS_PATH, MODEL_PATH, assign_tiers, build_pipeline, score

ROOT = os.path.dirname(os.path.abspath(__file__))


def prepare(df):
    clean = validate_and_clean(
        df.drop(columns=["is_fraud", "fraud_type"], errors="ignore")
    )
    return clean, build_features(clean)


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": p, "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-9)
    }


def main():
    start = time.time()
    print("1/6 generating data ...")

    train_df = make_dataset(60000, 7000, .035, 11)
    hold_df = make_dataset(15000, 1800, .035, 7)

    _, X_train_all = prepare(train_df)
    _, X_hold = prepare(hold_df)
    y_train_all = train_df.is_fraud.to_numpy()
    y_hold = hold_df.is_fraud.to_numpy()

    print("2/6 splitting train / validation ...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=.2,
        stratify=y_train_all, random_state=1
    )

    print("3/6 baseline logistic regression ...")
    baseline = Pipeline([
        ("pre", ColumnTransformer([
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES)
        ])),
        ("clf", LogisticRegression(max_iter=800, class_weight="balanced"))
    ])
    baseline.fit(X_train[FEATURES], y_train)
    baseline_p = baseline.predict_proba(X_hold[FEATURES])[:, 1]

    print("4/6 training gradient boosting model ...")
    pipe = build_pipeline()
    pipe.fit(X_train[FEATURES], y_train)
    n_iter = int(pipe.named_steps["clf"].n_iter_)

    p_val = score(pipe, X_val)
    p_hold = score(pipe, X_hold)

    # Select risk thresholds using validation data
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    precision, recall = precision[:-1], recall[:-1]

    high_idx = np.where(precision >= .85)[0]
    medium_idx = np.where(recall >= .90)[0]

    high = float(thresholds[high_idx[0]]) if len(high_idx) else .7
    medium = float(thresholds[medium_idx[-1]]) if len(medium_idx) else .3

    high = float(np.clip(high, .5, .9))
    medium = float(np.clip(medium, .15, high - .15))
    tiers_threshold = {"high": round(high, 3), "medium": round(medium, 3)}
    print("   thresholds:", tiers_threshold)

    print("5/6 evaluating holdout + feature importance ...")
    tiers = assign_tiers(p_hold, tiers_threshold)
    flagged = (tiers != "Low").astype(int)
    high_only = (tiers == "High").astype(int)

    per_type = {}
    for fraud_type in sorted(set(hold_df.fraud_type) - {""}):
        mask = (hold_df.fraud_type == fraud_type).to_numpy()
        per_type[fraud_type] = {
            "count": int(mask.sum()),
            "caught": int(flagged[mask].sum()),
            "recall": float(flagged[mask].mean())
        }

    metrics = {
        "model": "HistGradientBoostingClassifier",
        "baseline": "LogisticRegression (class-balanced)",
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "holdout_rows": int(len(X_hold)),
        "train_fraud_rate": float(y_train.mean()),
        "holdout_fraud_rate": float(y_hold.mean()),
        "boosting_iterations": n_iter,
        "thresholds": tiers_threshold,
        "holdout": {
            "roc_auc": float(roc_auc_score(y_hold, p_hold)),
            "pr_auc": float(average_precision_score(y_hold, p_hold)),
            "flagged_medium_plus": prf(y_hold, flagged),
            "flagged_high_only": prf(y_hold, high_only),
            "by_fraud_type": per_type
        },
        "baseline_holdout": {
            "roc_auc": float(roc_auc_score(y_hold, baseline_p)),
            "pr_auc": float(average_precision_score(y_hold, baseline_p))
        },
        "features": [
            {"name": f, "label": FEATURE_LABELS[f]} for f in FEATURES
        ]
    }

    importance = permutation_importance(
        pipe, X_hold[FEATURES], y_hold,
        scoring="average_precision", n_repeats=3,
        random_state=3, n_jobs=1
    )

    scores = {}
    for feature, value in zip(FEATURES, importance.importances_mean):
        key = {
            "log_amount": "amount",
            "is_night": "hour",
            "is_weekend": "dow"
        }.get(feature, feature)
        scores[key] = scores.get(key, 0.0) + max(float(value), 0.0)

    total = sum(scores.values()) or 1.0
    metrics["importance"] = sorted(
        [
            {
                "feature": key,
                "label": FEATURE_LABELS[key],
                "importance": value / total
            }
            for key, value in scores.items()
        ],
        key=lambda x: x["importance"],
        reverse=True
    )

    print("6/6 saving artefacts ...")
    background = X_train_all.sample(200, random_state=5).reset_index(drop=True)

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    joblib.dump({
        "pipeline": pipe,
        "thresholds": tiers_threshold,
        "background": background,
        "features": FEATURES,
        "version": "sentinel-upgraded-1.0"
    }, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    hold_df.to_csv(
        os.path.join(ROOT, "data", "sample_transactions_labeled.csv"),
        index=False
    )
    hold_df.drop(columns=["is_fraud", "fraud_type"]).to_csv(
        os.path.join(ROOT, "data", "sample_transactions.csv"),
        index=False
    )

    h = metrics["holdout"]
    b = metrics["baseline_holdout"]

    print(
        f"\nHoldout ROC-AUC {h['roc_auc']:.4f} | PR-AUC {h['pr_auc']:.4f} "
        f"(baseline LR: {b['roc_auc']:.4f} / {b['pr_auc']:.4f})"
    )

    m = h["flagged_medium_plus"]
    print(
        f"Medium+High flags: precision {m['precision']:.3f} "
        f"recall {m['recall']:.3f} ({m['tp'] + m['fp']} flagged of {len(y_hold)})"
    )

    m = h["flagged_high_only"]
    print(f"High only        : precision {m['precision']:.3f} recall {m['recall']:.3f}")

    for key, value in per_type.items():
        print(f"  {key:15s} recall {value['recall']:.2f} ({value['caught']}/{value['count']})")

    print("Top features:", [
        (x["feature"], round(x["importance"], 3))
        for x in metrics["importance"][:6]
    ])
    print(f"done in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
