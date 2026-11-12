import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


def demo_data(seed=42):
    """Synthetic monthly E. coli / ciprofloxacin data (replace with real data)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", "2025-12-01", freq="MS")
    n = len(dates)
    base = 28 + 0.12 * np.arange(n) + 2 * np.sin(2 * np.pi * np.arange(n) / 12)
    tested = rng.integers(80, 160, n)
    pct = np.clip(base + rng.normal(0, 1.8, n), 0, 100)
    resistant = rng.binomial(tested, pct / 100)
    return pd.DataFrame({"date": dates, "tested": tested, "resistant": resistant})


def prepare(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df["rate"] = 100 * df["resistant"] / df["tested"]
    return df.asfreq("MS").interpolate()


def forecast(df, horizon=6):
    y = df["rate"]
    seasonal = "add" if len(y) >= 36 else None
    fit = ExponentialSmoothing(
        y, trend="add", damped_trend=True,
        seasonal=seasonal, seasonal_periods=12 if seasonal else None,
    ).fit()
    mean = fit.forecast(horizon)
    sd = np.std(fit.resid, ddof=1)
    steps = np.sqrt(np.arange(1, horizon + 1))
    out = pd.DataFrame({
        "forecast": mean,
        "lower": mean - 1.96 * sd * steps,
        "upper": mean + 1.96 * sd * steps,
    })
    return out.clip(0, 100), fit


def risk_level(fc, amber=30.0, red=40.0):
    peak = fc["forecast"].max()
    if peak >= red:
        return "RED", peak
    if peak >= amber:
        return "AMBER", peak
    return "GREEN", peak


def anomaly(df, window=12, z=2.0):
    """Flag latest month if it sits far above the recent average."""
    recent = df["rate"].iloc[-window - 1:-1]
    score = (df["rate"].iloc[-1] - recent.mean()) / (recent.std(ddof=1) or 1)
    return bool(score > z), float(score)
