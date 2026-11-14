"""Feature engineering - identical code path for training and batch scoring"""

import numpy as np
import pandas as pd

# Required and optional CSV columns
REQUIRED = ["transaction_id", "timestamp", "user_id", "amount"]

OPTIONAL_DEFAULTS = {
    "transaction_type": "UNKNOWN",
    "merchant_category": "unknown",
    "city": "unknown",
    "country": "unknown",
    "device_type": "unknown",
    "device_id": "",
    "failed_attempts_24h": 0,
    "account_age_days": 365,
}

CAT_FEATURES = [
    "transaction_type", "merchant_category",
    "device_type", "country"
]

NUM_FEATURES = [
    "amount", "log_amount", "hour", "dow", "is_night",
    "is_weekend", "round_amount", "failed_attempts_24h",
    "account_age_days", "txn_count_1h", "txn_count_24h",
    "mins_since_prev", "amount_to_user_mean",
    "amount_zscore_user", "new_device", "city_mismatch",
    "country_mismatch", "cities_24h"
]

FEATURES = NUM_FEATURES + CAT_FEATURES

FEATURE_LABELS = {
    "amount": "Transaction amount",
    "log_amount": "Transaction amount (log scale)",
    "hour": "Hour of day",
    "dow": "Day of week",
    "is_night": "Night-time activity",
    "is_weekend": "Weekend activity",
    "round_amount": "Round-number amount",
    "failed_attempts_24h": "Failed attempts (24h)",
    "account_age_days": "Account age",
    "txn_count_1h": "Transactions in last hour",
    "txn_count_24h": "Transactions in last 24h",
    "mins_since_prev": "Time since previous transaction",
    "amount_to_user_mean": "Amount vs user's usual spend",
    "amount_zscore_user": "Amount deviation from user norm",
    "new_device": "Unrecognised device",
    "city_mismatch": "Unusual city for user",
    "country_mismatch": "Unusual country for user",
    "cities_24h": "Distinct cities in 24h",
    "transaction_type": "Transaction type",
    "merchant_category": "Merchant category",
    "device_type": "Device type",
    "country": "Country"
}


class SchemaError(ValueError):
    pass


def validate_and_clean(df):
    """Validate CSV columns and clean transaction data."""

    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(" ", "_")
        for col in df.columns
    ]

    missing = [col for col in REQUIRED if col not in df.columns]

    if missing:
        raise SchemaError(
            f"Missing columns: {', '.join(missing)}. "
            f"Required: {', '.join(REQUIRED)}"
        )

    # Add optional columns with default values
    for col, default in OPTIONAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    # Convert data types
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )
    df["amount"] = pd.to_numeric(
        df["amount"], errors="coerce"
    )

    # Remove invalid transactions
    bad = (
        df["timestamp"].isna()
        | df["amount"].isna()
        | df["user_id"].isna()
    )

    if bad.all():
        raise SchemaError(
            "No valid rows. Check timestamp and amount formats."
        )

    dropped = int(bad.sum())
    df = df[~bad].copy()

    # Clean numeric columns
    df["failed_attempts_24h"] = (
        pd.to_numeric(df["failed_attempts_24h"], errors="coerce")
        .fillna(0).clip(lower=0)
    )

    df["account_age_days"] = (
        pd.to_numeric(df["account_age_days"], errors="coerce")
        .fillna(365).clip(lower=0)
    )

    # Fill missing categorical values
    text_cols = [
        "transaction_type", "merchant_category", "city",
        "country", "device_type", "device_id"
    ]

    for col in text_cols:
        df[col] = (
            df[col].fillna(OPTIONAL_DEFAULTS[col])
            .astype(str).str.strip()
        )

    df["user_id"] = df["user_id"].astype(str)
    df["transaction_id"] = df["transaction_id"].astype(str)

    df.attrs["dropped_rows"] = dropped

    return df.reset_index(drop=True)


def _window_counts(ts, window):
    """Count transactions within the given time window."""

    left = np.searchsorted(ts, ts - window, side="right")
    return np.arange(len(ts)) - left + 1


def build_features(df):
    """Generate all ML features from transaction data."""

    n = len(df)
    ts = df["timestamp"]

    # Convert timestamps into seconds
    seconds = (
        (ts - pd.Timestamp("1970-01-01"))
        // pd.Timedelta(seconds=1)
    ).to_numpy().astype("int64")

    out = pd.DataFrame(index=df.index)

    # Basic transaction features
    out["amount"] = df["amount"].astype(float)
    out["log_amount"] = np.log1p(
        out["amount"].clip(lower=0)
    )
    out["hour"] = ts.dt.hour
    out["dow"] = ts.dt.dayofweek

    out["is_night"] = (
        (out["hour"] >= 22) | (out["hour"] <= 5)
    ).astype(int)

    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    out["round_amount"] = (
        (out["amount"] % 100 == 0) &
        (out["amount"] >= 100)
    ).astype(int)

    out["failed_attempts_24h"] = df["failed_attempts_24h"].astype(float)
    out["account_age_days"] = df["account_age_days"].astype(float)

    # Sort transactions by user and time
    order = np.lexsort((
        seconds, df["user_id"].to_numpy()
    ))

    users = df["user_id"].to_numpy()[order]
    sorted_ts = seconds[order]
    amounts = out["amount"].to_numpy()[order]

    cities = df["city"].to_numpy()[order]
    countries = df["country"].to_numpy()[order]
    devices = df["device_id"].to_numpy()[order]
    types = df["transaction_type"].to_numpy()[order]

    # Find the start and end of each user's transactions
    bounds = np.flatnonzero(
        np.r_[True, users[1:] != users[:-1], True]
    )

    # Initialize behavioural features
    count_1h = np.zeros(n)
    count_24h = np.zeros(n)
    since_prev = np.full(n, 10080.0)
    amount_ratio = np.ones(n)
    zscore = np.zeros(n)

    new_device = np.zeros(n)
    city_mismatch = np.zeros(n)
    country_mismatch = np.zeros(n)
    cities_24h = np.ones(n)

    # Calculate behaviour separately for each user
    for start, end in zip(bounds[:-1], bounds[1:]):

        time = sorted_ts[start:end]
        amount = amounts[start:end]
        size = end - start

        # Transaction frequency
        count_1h[start:end] = _window_counts(time, 3600)
        count_24h[start:end] = _window_counts(time, 86400)

        if size > 1:
            since_prev[start + 1:end] = np.minimum(
                np.diff(time) / 60.0, 10080.0
            )

            # Compare each transaction with other transactions
            mean = (amount.sum() - amount) / (size - 1)
            amount_ratio[start:end] = (
                amount / np.maximum(mean, 1.0)
            )

            # Calculate deviation from usual spending
            variance = np.maximum(
                (np.sum(amount ** 2) - amount ** 2)
                / (size - 1) - mean ** 2,
                0
            )

            zscore[start:end] = (
                (amount - mean) /
                np.maximum(np.sqrt(variance), 0.25 * np.maximum(mean, 1.0))
            )

        # Detect unusual city and country
        user_cities = pd.Series(cities[start:end])
        usual_city = user_cities.mode().iloc[0]

        city_mismatch[start:end] = (
            (user_cities.to_numpy() != usual_city) & (size > 1)
        )

        user_countries = pd.Series(countries[start:end])
        usual_country = user_countries.mode().iloc[0]

        country_mismatch[start:end] = (
            (user_countries.to_numpy() != usual_country) & (size > 1)
        )

        # Detect unfamiliar devices in digital transactions
        digital = np.isin(
            types[start:end],
            ["ECOM", "TRANSFER", "BILL_PAY"]
        )

        user_devices = pd.Series(devices[start:end])

        if digital.any() and size > 1:
            usual_device = (
                user_devices[digital]
                .astype(str).value_counts().index[0]
            )

            new_device[start:end] = (
                digital &
                (user_devices.to_numpy() != usual_device) &
                (user_devices.to_numpy() != "")
            )

        # Count distinct cities in the previous 24 hours
        if size > 1:
            left = np.searchsorted(
                time, time - 86400, side="right"
            )

            for i in range(size):
                if i > left[i]:
                    cities_24h[start + i] = len(
                        set(cities[start + left[i]:start + i + 1])
                    )

    # Restore original transaction order
    def restore_order(values):
        result = np.empty(n)
        result[order] = values
        return result

    out["txn_count_1h"] = restore_order(count_1h)
    out["txn_count_24h"] = restore_order(count_24h)
    out["mins_since_prev"] = restore_order(since_prev)
    out["amount_to_user_mean"] = restore_order(amount_ratio)

    out["amount_zscore_user"] = np.clip(
        restore_order(zscore), -10, 25
    )

    out["new_device"] = restore_order(new_device)
    out["city_mismatch"] = restore_order(city_mismatch)
    out["country_mismatch"] = restore_order(country_mismatch)
    out["cities_24h"] = restore_order(cities_24h)

    # Add categorical ML features
    for col in CAT_FEATURES:
        out[col] = df[col].astype(str)

    return out[FEATURES]
