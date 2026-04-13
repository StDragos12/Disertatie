import numpy as np
import pandas as pd

SYNTHETIC_SERIES_META = {
    "white-noise": {
        "title": "White Noise",
        "category": "Staționară / fără sezonalitate",
        "description": (
            "Serie sintetică de zgomot alb, utilă pentru ilustrarea unei serii aproximativ "
            "staționare fără structură temporală pronunțată."
        ),
    },
    "random-walk": {
        "title": "Random Walk",
        "category": "Nestaționară / fără sezonalitate clară",
        "description": (
            "Serie sintetică de tip random walk, clasică pentru demonstrarea "
            "nestaționarității."
        ),
    },
    "linear-trend": {
        "title": "Trend liniar + zgomot",
        "category": "Nestaționară / trend",
        "description": (
            "Serie sintetică cu trend liniar crescător și componentă de zgomot."
        ),
    },
    "seasonal-noise": {
        "title": "Sinusoidală + zgomot",
        "category": "Sezonieră",
        "description": (
            "Serie sintetică sezonieră, bazată pe o componentă sinusoidală anuală și zgomot."
        ),
    },
    "trend-seasonal": {
        "title": "Trend + sezonalitate + zgomot",
        "category": "Exemplu complet",
        "description": (
            "Serie sintetică combinată, conținând trend, sezonalitate și zgomot."
        ),
    },
}


def build_series_dataframe(series: pd.Series, name: str, category: str, description: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": series.index,
        "value": series.values,
        "series_name": name,
        "category": category,
        "description": description,
    })


def generate_synthetic_series(series_key: str, periods: int = 72) -> pd.DataFrame:
    if series_key not in SYNTHETIC_SERIES_META:
        raise ValueError("Cheie serie sintetică invalidă.")

    meta = SYNTHETIC_SERIES_META[series_key]
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2018-01-01", periods=periods, freq="MS")
    t = np.arange(periods)

    if series_key == "white-noise":
        values = rng.normal(loc=0.0, scale=1.0, size=periods)
    elif series_key == "random-walk":
        steps = rng.normal(loc=0.0, scale=0.8, size=periods)
        values = np.cumsum(steps)
    elif series_key == "linear-trend":
        values = 0.45 * t + rng.normal(loc=0.0, scale=1.2, size=periods)
    elif series_key == "seasonal-noise":
        values = 8 * np.sin(2 * np.pi * t / 12) + rng.normal(loc=0.0, scale=1.2, size=periods)
    elif series_key == "trend-seasonal":
        values = 0.22 * t + 7 * np.sin(2 * np.pi * t / 12) + rng.normal(loc=0.0, scale=1.0, size=periods)
    else:
        raise ValueError("Seria sintetică nu este definită.")

    series = pd.Series(values, index=dates, name="value")
    return build_series_dataframe(
        series=series,
        name=meta["title"],
        category=meta["category"],
        description=meta["description"],
    )


def generate_temperature_demo_series(periods: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range(start="2017-01-01", periods=periods, freq="MS")
    t = np.arange(periods)

    seasonal = 10 * np.sin(2 * np.pi * t / 12 - np.pi / 6)
    trend = 0.03 * t
    noise = rng.normal(0, 1.1, size=periods)
    values = 12 + seasonal + trend + noise

    series = pd.Series(values, index=dates, name="value")
    return build_series_dataframe(
        series=series,
        name="Temperatură lunară demonstrativă",
        category="Serie climatică demonstrativă",
        description=(
            "Serie lunară demonstrativă cu sezonalitate anuală puternică, ușor trend și zgomot."
        ),
    )