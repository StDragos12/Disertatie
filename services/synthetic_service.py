import numpy as np
import pandas as pd


SYNTHETIC_SERIES_META = {
    "white-noise": {
        "title": "White Noise",
        "category": "Serie staționară",
        "description": (
            "Serie sintetică staționară, fără trend și fără sezonalitate, "
            "utilizată ca exemplu de referință pentru zgomot aleator."
        ),
    },
    "random-walk": {
        "title": "Random Walk",
        "category": "Serie nestaționară",
        "description": (
            "Serie sintetică nestaționară, în care valoarea curentă depinde "
            "de valoarea anterioară și de o variație aleatoare."
        ),
    },
    "linear-trend": {
        "title": "Trend liniar + zgomot",
        "category": "Serie cu trend",
        "description": (
            "Serie sintetică formată dintr-o componentă liniară de trend și "
            "o componentă aleatoare de zgomot."
        ),
    },
    "seasonal-noise": {
        "title": "Sinusoidală + zgomot",
        "category": "Serie sezonieră",
        "description": (
            "Serie sintetică periodică, cu variație sinusoidală și zgomot, "
            "utilizată pentru ilustrarea sezonalității."
        ),
    },
    "trend-seasonal": {
        "title": "Trend + sezonalitate",
        "category": "Serie cu trend și sezonalitate",
        "description": (
            "Serie sintetică ce combină o componentă de trend cu o componentă "
            "sezonieră, apropiată de comportamentul unor serii reale."
        ),
    },
}


def generate_synthetic_series(series_key: str, periods: int = 72) -> pd.DataFrame:
    if series_key not in SYNTHETIC_SERIES_META:
        series_key = "white-noise"

    rng = np.random.default_rng(42)
    dates = pd.date_range(
        start="2017-01-01",
        periods=periods,
        freq="MS",
    )

    t = np.arange(periods)

    if series_key == "white-noise":
        values = rng.normal(
            loc=0.0,
            scale=1.0,
            size=periods,
        )

    elif series_key == "random-walk":
        steps = rng.normal(
            loc=0.0,
            scale=0.6,
            size=periods,
        )
        values = np.cumsum(steps)

    elif series_key == "linear-trend":
        trend = 0.08 * t
        noise = rng.normal(
            loc=0.0,
            scale=0.8,
            size=periods,
        )
        values = trend + noise

    elif series_key == "seasonal-noise":
        seasonal = 2.5 * np.sin(
            2 * np.pi * t / 12
        )
        noise = rng.normal(
            loc=0.0,
            scale=0.5,
            size=periods,
        )
        values = seasonal + noise

    elif series_key == "trend-seasonal":
        trend = 0.04 * t
        seasonal = 2.0 * np.sin(
            2 * np.pi * t / 12
        )
        noise = rng.normal(
            loc=0.0,
            scale=0.45,
            size=periods,
        )
        values = trend + seasonal + noise

    else:
        values = rng.normal(
            loc=0.0,
            scale=1.0,
            size=periods,
        )

    meta = SYNTHETIC_SERIES_META[series_key]

    return pd.DataFrame({
        "date": dates,
        "value": values,
        "series_name": meta["title"],
        "category": meta["category"],
        "description": meta["description"],
    })


def generate_temperature_demo_series(periods: int = 72) -> pd.DataFrame:
    rng = np.random.default_rng(123)

    dates = pd.date_range(
        start="2017-01-01",
        periods=periods,
        freq="MS",
    )

    t = np.arange(periods)

    seasonal = 10 * np.sin(
        2 * np.pi * (t - 2) / 12
    )

    trend = 0.015 * t

    noise = rng.normal(
        loc=0.0,
        scale=1.2,
        size=periods,
    )

    values = 14 + seasonal + trend + noise

    return pd.DataFrame({
        "date": dates,
        "value": values,
        "series_name": "Temperatură lunară demonstrativă",
        "category": "Serie climatică demonstrativă",
        "description": (
            "Serie demonstrativă de temperatură lunară, utilizată pentru "
            "ilustrarea sezonalității anuale într-un context climatic."
        ),
    })