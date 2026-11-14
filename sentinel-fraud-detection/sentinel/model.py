"""Model pipeline, batch scoring, risk tiers and explanations."""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .features import CAT_FEATURES, FEATURES, FEATURE_LABELS


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "sentinel_model.joblib")
METRICS_PATH = os.path.join(ROOT, "models", "metrics.json")

MERGE = {
    "log_amount": "amount",
    "is_night": "hour",
    "is_weekend": "dow"
}


def build_pipeline(seed=42):
    """Create the ML model and preprocessing pipeline."""

    encoder = ColumnTransformer([
        ("cat", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan
        ), CAT_FEATURES)
    ], remainder="passthrough")

    model = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=500,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        categorical_features=list(range(len(CAT_FEATURES))),
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=25,
        random_state=seed
    )

    return Pipeline([("pre", encoder), ("clf", model)])


def _ordered(X):
    return X[FEATURES]


def score(pipe, X):
    """Predict fraud probability."""
    return pipe.predict_proba(_ordered(X))[:, 1]


def assign_tiers(p, thresholds):
    """Classify transactions by risk."""
    return np.where(
        p >= thresholds["high"], "High",
        np.where(p >= thresholds["medium"], "Medium", "Low")
    )


def load_bundle():
    """Load trained model and evaluation metrics."""

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not trained yet. Run: python train.py"
        )

    bundle = joblib.load(MODEL_PATH)

    with open(METRICS_PATH) as file:
        bundle["metrics"] = json.load(file)

    return bundle


def contributions(pipe, X, background, n_bg=24, seed=0):
    """Calculate Shapley-style feature contributions."""

    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]

    columns = CAT_FEATURES + [
        f for f in FEATURES if f not in CAT_FEATURES
    ]

    Z = pre.transform(_ordered(X))
    Zb = pre.transform(_ordered(background))

    n, total = Z.shape
    rng = np.random.default_rng(seed)
    rows = n * n_bg

    Xr = np.repeat(Z, n_bg, axis=0)
    Br = Zb[rng.integers(0, len(Zb), size=rows)]

    perm = np.argsort(rng.random((rows, total)), axis=1)
    rank = np.argsort(perm, axis=1)

    previous = clf.predict_proba(Br)[:, 1]
    result = np.zeros((rows, total))
    indexes = np.arange(rows)

    for k in range(1, total + 1):
        current = clf.predict_proba(
            np.where(rank < k, Xr, Br)
        )[:, 1]

        result[indexes, perm[:, k - 1]] = current - previous
        previous = current

    result = result.reshape(n, n_bg, total).mean(axis=1)

    output = np.zeros((n, total))

    for j, feature in enumerate(columns):
        output[:, FEATURES.index(feature)] = result[:, j]

    return output


def fold(contrib_row):
    """Combine related feature contributions."""

    result = dict(zip(FEATURES, map(float, contrib_row)))

    for source, target in MERGE.items():
        result[target] += result.pop(source, 0.0)

    return result


def describe(feature, row):
    """Convert feature values into readable explanations."""

    value = row.get(feature)

    try:
        if feature == "amount":
            amount = float(value)
            if amount < 10:
                return f"Micro-amount (${amount:,.2f}), a card-testing pattern"
            return f"Large amount (${amount:,.2f})"

        if feature == "amount_to_user_mean":
            return f"Amount is {float(value):.1f}x this user's usual spend"

        if feature == "amount_zscore_user":
            return f"Amount is {float(value):.1f} std devs above the user's norm"

        if feature in ("txn_count_1h", "txn_count_24h"):
            period = "within an hour" if feature == "txn_count_1h" else "in 24h"
            return f"{int(value)} transactions from this user {period}"

        if feature == "mins_since_prev":
            minutes = float(value)
            if minutes < 1:
                return "Within seconds of the user's previous transaction"
            return f"Only {minutes:.0f} min after the user's previous transaction"

        if feature == "failed_attempts_24h":
            return f"{int(value)} failed attempts in the last 24h"

        messages = {
            "new_device": "Device not previously seen for this user",
            "city_mismatch": "City differs from the user's usual location",
            "country_mismatch": "Country differs from the user's usual location",
            "dow": "Unusual day of week",
            "round_amount": "Round-number amount (cash-out pattern)"
        }

        if feature in messages:
            return messages[feature]

        if feature == "cities_24h":
            return f"{int(value)} different cities within 24h"

        if feature == "hour":
            return f"Unusual time of day ({int(value):02d}:00)"

        if feature in ("merchant_category", "device_type"):
            return f"{feature.replace('_', ' ').title()}: {str(value).replace('_', ' ')}"

        if feature == "transaction_type":
            types = {
                "ECOM": "Card-not-present online purchase",
                "TRANSFER": "Account-to-account transfer",
                "ATM": "ATM cash withdrawal",
                "POS": "In-store card payment",
                "BILL_PAY": "Bill payment"
            }
            return types.get(str(value), f"Transaction type: {value}")

        if feature == "country":
            return f"Country: {value}"

        if feature == "account_age_days":
            return f"Account age: {int(value)} days"

    except (TypeError, ValueError):
        pass

    return FEATURE_LABELS.get(feature, feature)


def top_reasons(contrib_row, row, k=3, min_c=0.015):
    """Return the strongest fraud reasons."""

    items = sorted(
        fold(contrib_row).items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        (feature, value, describe(feature, row))
        for feature, value in items[:k]
        if value >= min_c
    ]


def risk_actions(tier):
    """Suggest action according to risk level."""

    actions = {
        "High": "Hold the transaction and send to an analyst now; trigger step-up verification with the customer.",
        "Medium": "Queue for analyst review and request step-up authentication if the user is active.",
        "Low": "No action. Transaction looks consistent with normal behaviour."
    }

    return actions[tier]
