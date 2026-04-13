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


def load_ndvi() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Nu am găsit CSV-ul: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"date", "site", "ndvi"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Lipsesc coloane obligatorii în CSV: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
    df["site"] = df["site"].astype(str).str.strip()
    df["site_raw"] = df["site"]
    df["site"] = df["site"].apply(normalize_site_name)

    df = df[["date", "site", "site_raw", "ndvi"]]
    df = df.dropna(subset=["date", "site", "ndvi"]).sort_values(["site", "date"]).reset_index(drop=True)
    return df


def get_sites(df: pd.DataFrame) -> list[str]:
    return sorted(df["site"].unique().tolist())