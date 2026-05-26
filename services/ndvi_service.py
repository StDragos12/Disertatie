import pandas as pd
from config import DATA_PATH, SITE_LABELS


def normalize_site_name(value: str) -> str:
    s = str(value).strip()
    mapping = {
        "ParcBucuresti": "ParcBucuresti",
        "Parc Bucuresti": "ParcBucuresti",
        "Pădure": "ParcBucuresti",
        "Padure": "ParcBucuresti",
        "UrbanCentral": "UrbanCentral",
        "Urban Central": "UrbanCentral",
        "Urban": "UrbanCentral",
        "AgricolIlfov": "AgricolIlfov",
        "Agricol Ilfov": "AgricolIlfov",
        "TerenAgricol": "AgricolIlfov",
        "Agricol": "AgricolIlfov",
    }
    return mapping.get(s, s)


def pretty_site_name(site_code: str) -> str:
    return SITE_LABELS.get(site_code, site_code)


def load_ndvi():

    df = pd.read_csv(DATA_PATH)

    df.columns = [
        c.strip().lower()
        for c in df.columns
    ]

    if "roi" not in df.columns:
        raise ValueError(
            "Lipsește coloana roi"
        )

    if "index" not in df.columns:
        raise ValueError(
            "Lipsește coloana index"
        )

    value_col = None

    for col in [
        "value",
        "ndvi",
        "mean",
    ]:

        if col in df.columns:
            value_col = col
            break

    if value_col is None:
        raise ValueError(
            "Nu există coloană de valori."
        )

    if "date" in df.columns:
        df["date"] = pd.to_datetime(
            df["date"]
        )

    elif "timestamp" in df.columns:
        df["date"] = pd.to_datetime(
            df["timestamp"]
        )

    else:
        raise ValueError(
            "Nu există coloană temporală."
        )

    df["site"] = df["roi"]

    df["ndvi"] = df[value_col]

    return df


def get_sites(df: pd.DataFrame) -> list[str]:
    return sorted(df["site"].unique().tolist())