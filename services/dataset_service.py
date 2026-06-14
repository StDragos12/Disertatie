
import io
import json
import os
import re
import unicodedata
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

try:
    from google.cloud import storage
except Exception:
    storage = None

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA


DEMO_DATASET_ID = "demo"
LOCAL_DATASET_DIR = Path("data") / "user_datasets"
DATASETS_INDEX_BLOB = "user_datasets/index.json"
STATUS_FILENAME = "status.json"
ROI_TIMESERIES_FILENAME = "roi_timeseries.csv"
REQUIRED_COLUMNS = {"date", "roi", "index", "value"}
PIXEL_COLUMNS = {"pixel_id", "row", "col"}
SUPPORTED_UPLOAD_EXTENSIONS = {".csv", ".npy", ".zip"}
KNOWN_SPECTRAL_INDICES = {"NDVI", "NDMI", "SAVI", "AVI", "EVI", "GNDVI"}



def _bucket_name() -> str | None:
    return os.getenv("GCS_BUCKET_NAME")


def _storage_client():
    if storage is None:
        raise ImportError(
            "google-cloud-storage nu este instalat. Adaugă google-cloud-storage în requirements.txt."
        )
    return storage.Client()


def using_gcs() -> bool:
    return bool(_bucket_name())


def slugify(value: str) -> str:
    value = str(value or "dataset").strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "dataset"


def normalize_dataset_id(dataset_id: str | None) -> str:
    dataset_id = str(dataset_id or DEMO_DATASET_ID).strip().lower()
    if not dataset_id:
        return DEMO_DATASET_ID
    return slugify(dataset_id)


def _local_dataset_path(dataset_id: str) -> Path:
    return LOCAL_DATASET_DIR / normalize_dataset_id(dataset_id) / "indices_timeseries.csv"


def _local_roi_dataset_path(dataset_id: str) -> Path:
    return LOCAL_DATASET_DIR / normalize_dataset_id(dataset_id) / ROI_TIMESERIES_FILENAME


def _local_manifest_path(dataset_id: str) -> Path:
    return LOCAL_DATASET_DIR / normalize_dataset_id(dataset_id) / "manifest.json"


def _gcs_dataset_blob(dataset_id: str) -> str:
    return f"user_datasets/{normalize_dataset_id(dataset_id)}/indices_timeseries.csv"


def _gcs_roi_dataset_blob(dataset_id: str) -> str:
    return f"user_datasets/{normalize_dataset_id(dataset_id)}/{ROI_TIMESERIES_FILENAME}"


def _gcs_manifest_blob(dataset_id: str) -> str:
    return f"user_datasets/{normalize_dataset_id(dataset_id)}/manifest.json"


def _gcs_status_blob(dataset_id: str) -> str:
    return f"user_datasets/{normalize_dataset_id(dataset_id)}/status.json"


def _local_status_path(dataset_id: str) -> Path:
    return LOCAL_DATASET_DIR / normalize_dataset_id(dataset_id) / STATUS_FILENAME


def _write_json_payload_local(dataset_id: str, filename: str, payload: dict) -> None:
    dataset_dir = LOCAL_DATASET_DIR / normalize_dataset_id(dataset_id)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_json_payload_gcs(dataset_id: str, blob_filename: str, payload: dict) -> None:
    bucket_name = _bucket_name()
    if not bucket_name:
        return _write_json_payload_local(dataset_id, blob_filename, payload)

    client = _storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"user_datasets/{normalize_dataset_id(dataset_id)}/{blob_filename}")
    blob.upload_from_string(
        json.dumps(payload, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8",
    )


def write_dataset_status(
    dataset_id: str,
    status: str,
    message: str,
    extra: dict | None = None,
) -> dict:
    payload = {
        "dataset_id": normalize_dataset_id(dataset_id),
        "status": status,
        "message": message,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    if extra:
        payload.update(extra)

    if using_gcs():
        _write_json_payload_gcs(dataset_id, STATUS_FILENAME, payload)
    else:
        _write_json_payload_local(dataset_id, STATUS_FILENAME, payload)

    return payload


def read_dataset_status(dataset_id: str) -> dict:
    dataset_id = normalize_dataset_id(dataset_id)

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        blob = bucket.blob(_gcs_status_blob(dataset_id))
        if blob.exists():
            return json.loads(blob.download_as_text(encoding="utf-8"))
        return {}

    path = _local_status_path(dataset_id)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def update_dataset_status(
    dataset_id: str,
    status: str,
    message: str,
    extra: dict | None = None,
) -> dict:
    """
    Actualizează status.json și record-ul din user_datasets/index.json.
    """
    dataset_id = normalize_dataset_id(dataset_id)
    payload = write_dataset_status(dataset_id, status, message, extra=extra)

    index = _read_index_gcs() if using_gcs() else _read_index_local()
    changed = False
    for item in index.get("datasets", []):
        if item.get("dataset_id") == dataset_id:
            item["status"] = status
            item["status_message"] = message
            item["updated_at"] = payload["updated_at"]
            if extra:
                item.update(extra)
            changed = True
            break

    if changed:
        if using_gcs():
            _write_index_gcs(index)
        else:
            _write_index_local(index)

    return payload


def user_ml_result_blob_name(dataset_id: str, index_name: str, roi: str, pixel_count: int) -> str:
    return (
        f"user_datasets/{normalize_dataset_id(dataset_id)}/"
        f"ml/{str(index_name).upper()}/{str(roi).lower()}/"
        f"pixels_{int(pixel_count)}/result.json"
    )


def upload_user_ml_result_json(
    dataset_id: str,
    index_name: str,
    roi: str,
    pixel_count: int,
    payload: dict,
) -> str:
    """
    Scrie result.json pentru un dataset utilizator.

    În producție scrie în Cloud Storage.
    Local scrie în data/user_datasets/<dataset_id>/ml/...
    """
    dataset_id = normalize_dataset_id(dataset_id)
    index_name = str(index_name).upper()
    roi = str(roi).lower()
    pixel_count = int(pixel_count)

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        blob_name = user_ml_result_blob_name(dataset_id, index_name, roi, pixel_count)
        bucket.blob(blob_name).upload_from_string(
            json.dumps(payload, ensure_ascii=False),
            content_type="application/json; charset=utf-8",
        )
        return f"gs://{_bucket_name()}/{blob_name}"

    local_path = (
        LOCAL_DATASET_DIR
        / dataset_id
        / "ml"
        / index_name
        / roi
        / f"pixels_{pixel_count}"
        / "result.json"
    )
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return str(local_path)


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "CSV-ul trebuie să conțină cel puțin coloanele: date, roi, index, value. "
            f"Lipsesc: {', '.join(sorted(missing))}."
        )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["roi"] = df["roi"].astype(str).str.strip().str.lower()
    df["index"] = df["index"].astype(str).str.strip().str.upper()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["date", "roi", "index", "value"])
    df = df[df["roi"] != ""]
    df = df[df["index"] != ""]

    if df.empty:
        raise ValueError("CSV-ul nu conține observații valide după curățare.")

    return df


def is_pixel_level_dataframe(df: pd.DataFrame) -> bool:
    cols = {str(c).strip().lower() for c in df.columns}
    return PIXEL_COLUMNS.issubset(cols)


def validate_indices_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validează un CSV de utilizator.

    Sunt acceptate două formate:
    1) ROI-level: date, roi, index, value
    2) Pixel-level: date, roi, index, pixel_id, row, col, value

    Pentru ROI-level se agregă la date/roi/index.
    Pentru pixel-level se păstrează pixel_id/row/col, iar modulele ROI-level agregă automat media.
    """
    df = _standardize_dataframe(df)
    pixel_level = is_pixel_level_dataframe(df)

    if pixel_level:
        df["pixel_id"] = df["pixel_id"].astype(str).str.strip()
        df["row"] = pd.to_numeric(df["row"], errors="coerce").astype("Int64")
        df["col"] = pd.to_numeric(df["col"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["pixel_id", "row", "col"])
        df = df[df["pixel_id"] != ""]
        if df.empty:
            raise ValueError("CSV-ul pixel-level nu conține pixeli valizi după curățare.")
        df = (
            df.groupby(["date", "roi", "index", "pixel_id", "row", "col"], as_index=False)["value"]
            .mean()
            .sort_values(["index", "roi", "pixel_id", "date"])
        )
        df["row"] = df["row"].astype(int)
        df["col"] = df["col"].astype(int)
        return df

    df = (
        df.groupby(["date", "roi", "index"], as_index=False)["value"]
        .mean()
        .sort_values(["index", "roi", "date"])
    )
    return df


def as_roi_level_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = validate_indices_dataframe(df)
    return (
        df.groupby(["date", "roi", "index"], as_index=False)["value"]
        .mean()
        .sort_values(["index", "roi", "date"])
    )


def build_roi_timeseries_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construiește varianta ROI-level mică a unui dataset.

    Pentru dataseturi pixel-level, reduce datele la:
      date, roi, index, value

    Această versiune este folosită de paginile temporale, Cross-Index,
    staționaritate și forecast, ca să nu se mai citească CSV-ul pixel-level
    mare la fiecare request.
    """
    return as_roi_level_dataframe(df)


def _write_roi_timeseries(dataset_id: str, roi_df: pd.DataFrame) -> None:
    dataset_id = normalize_dataset_id(dataset_id)
    csv_text = roi_df.to_csv(index=False)

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        bucket.blob(_gcs_roi_dataset_blob(dataset_id)).upload_from_string(
            csv_text,
            content_type="text/csv; charset=utf-8",
        )
        return

    path = _local_roi_dataset_path(dataset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(csv_text, encoding="utf-8")


def _read_roi_timeseries_if_exists(dataset_id: str) -> pd.DataFrame | None:
    dataset_id = normalize_dataset_id(dataset_id)

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        blob = bucket.blob(_gcs_roi_dataset_blob(dataset_id))

        if not blob.exists():
            return None

        df = pd.read_csv(io.StringIO(blob.download_as_text(encoding="utf-8")))
        return validate_indices_dataframe(df)

    path = _local_roi_dataset_path(dataset_id)

    if not path.exists():
        return None

    df = pd.read_csv(path)
    return validate_indices_dataframe(df)



def _read_index_local() -> dict:
    index_path = LOCAL_DATASET_DIR / "index.json"
    if not index_path.exists():
        return {"datasets": []}
    try:
        return json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {"datasets": []}


def _write_index_local(index: dict):
    LOCAL_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (LOCAL_DATASET_DIR / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_index_gcs() -> dict:
    bucket_name = _bucket_name()
    if not bucket_name:
        return _read_index_local()

    client = _storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(DATASETS_INDEX_BLOB)
    if not blob.exists():
        return {"datasets": []}
    try:
        return json.loads(blob.download_as_text(encoding="utf-8"))
    except Exception:
        return {"datasets": []}


def _write_index_gcs(index: dict):
    bucket_name = _bucket_name()
    if not bucket_name:
        return _write_index_local(index)
    client = _storage_client()
    bucket = client.bucket(bucket_name)
    bucket.blob(DATASETS_INDEX_BLOB).upload_from_string(
        json.dumps(index, ensure_ascii=False, indent=2),
        content_type="application/json; charset=utf-8",
    )


def _upsert_dataset_record(record: dict):
    index = _read_index_gcs() if using_gcs() else _read_index_local()
    datasets = [d for d in index.get("datasets", []) if d.get("dataset_id") != record["dataset_id"]]
    datasets.append(record)
    datasets = sorted(datasets, key=lambda d: d.get("display_name", d.get("dataset_id", "")).lower())
    index["datasets"] = datasets
    if using_gcs():
        _write_index_gcs(index)
    else:
        _write_index_local(index)


def list_datasets(include_demo: bool = True) -> list[dict]:
    datasets = []
    if include_demo:
        datasets.append({
            "dataset_id": DEMO_DATASET_ID,
            "display_name": "ROI demonstrative",
            "source": "demo",
            "status": "uploaded",
            "input_type": "demo",
        })
    index = _read_index_gcs() if using_gcs() else _read_index_local()
    for item in index.get("datasets", []):
        if item.get("dataset_id") != DEMO_DATASET_ID:
            datasets.append(item)
    return datasets


def get_dataset_record(dataset_id: str) -> dict | None:
    dataset_id = normalize_dataset_id(dataset_id)
    for dataset in list_datasets(include_demo=True):
        if dataset.get("dataset_id") == dataset_id:
            return dataset
    return None


def get_dataset_display_name(dataset_id: str) -> str:
    record = get_dataset_record(dataset_id)
    if record:
        return record.get("display_name", dataset_id)
    return normalize_dataset_id(dataset_id)


def build_dataset_options_html(selected_dataset: str = DEMO_DATASET_ID) -> str:
    selected_dataset = normalize_dataset_id(selected_dataset)
    html = ""
    for dataset in list_datasets(include_demo=True):
        dataset_id = dataset["dataset_id"]
        selected = "selected" if dataset_id == selected_dataset else ""
        label = dataset.get("display_name", dataset_id)
        html += f"<option value='{dataset_id}' {selected}>{label}</option>"
    return html


SUPPORTED_INDICES = ["NDVI", "NDMI", "SAVI", "AVI", "EVI", "GNDVI"]


def infer_index_from_filename(filename: str) -> str:
    name = str(filename or "").upper()

    for index_name in SUPPORTED_INDICES:
        if index_name in name:
            return index_name

    raise ValueError(
        "Nu s-a putut deduce indicele spectral din numele fișierului. "
        "Redenumește fișierul folosind unul dintre indicii acceptați: "
        "NDVI.npy, NDMI.npy, SAVI.npy, AVI.npy, EVI.npy sau GNDVI.npy."
    )




def infer_roi_from_zip_member(member_name: str, default_roi: str = "dataset_roi") -> str:
    """
    Dedu ROI-ul din structura unei arhive ZIP.

    Reguli:
    - NDVI.npy -> ROI-ul introdus în formular
    - data1/NDVI.npy -> data1
    - ferma/data1/NDVI.npy -> data1

    Astfel, un ZIP cu subfoldere poate reprezenta mai multe parcele/ROI-uri
    în același dataset.
    """
    default_roi = str(default_roi or "dataset_roi").strip().lower() or "dataset_roi"

    path = PurePosixPath(str(member_name or ""))
    parts = [
        part.strip()
        for part in path.parts
        if part.strip()
        and part.strip() not in {".", ".."}
        and part.strip().lower() != "__macosx"
    ]

    # Dacă există cel puțin un folder înainte de fișier, ultimul folder devine ROI.
    if len(parts) >= 2:
        return slugify(parts[-2])

    return slugify(default_roi)


def _load_npy_from_bytes(data: bytes, filename: str) -> np.ndarray:
    try:
        return np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Fișierul {filename} nu este un .npy valid: {exc}")



def _monthly_dates(start_date: str, periods: int) -> pd.DatetimeIndex:
    try:
        start = pd.to_datetime(start_date or "2021-01-01")
    except Exception:
        start = pd.Timestamp("2021-01-01")
    start = start.to_period("M").to_timestamp()
    return pd.date_range(start=start, periods=int(periods), freq="MS")


def npy_to_pixel_dataframe(
    npy_array: np.ndarray,
    roi: str,
    index_name: str,
    start_date: str = "2021-01-01",
    max_pixels: int | None = None,
) -> pd.DataFrame:
    """
    Transformă un .npy 3D într-un CSV pixel-level standardizat.

    Format acceptat: array 3D cu forma [timp, rânduri, coloane].
    Fiecare poziție (row, col) devine pixel_id, iar fiecare lună devine observație temporală.
    """
    arr = np.asarray(npy_array)

    if arr.ndim != 3:
        raise ValueError(
            "Pentru upload .npy este acceptat momentan doar un array 3D cu forma "
            "[timp, rânduri, coloane]."
        )

    arr = arr.astype(float)
    time_len, height, width = arr.shape
    dates = _monthly_dates(start_date, time_len)

    # pixel valid = are cel puțin o valoare finită în timp
    valid_mask = np.isfinite(arr).any(axis=0)
    positions = np.argwhere(valid_mask)

    if len(positions) == 0:
        raise ValueError("Array-ul .npy nu conține pixeli validați.")

    if max_pixels is not None and len(positions) > int(max_pixels):
        rng = np.random.default_rng(42)
        idx = rng.choice(np.arange(len(positions)), size=int(max_pixels), replace=False)
        positions = positions[idx]

    rows = []
    roi = str(roi or "dataset_roi").strip().lower() or "dataset_roi"
    index_name = str(index_name or "NDVI").strip().upper() or "NDVI"

    for r, c in positions:
        pixel_values = arr[:, int(r), int(c)]
        pixel_id = f"r{int(r)}_c{int(c)}"
        for date, value in zip(dates, pixel_values):
            if np.isfinite(value):
                rows.append({
                    "date": date,
                    "roi": roi,
                    "index": index_name,
                    "pixel_id": pixel_id,
                    "row": int(r),
                    "col": int(c),
                    "value": float(value),
                })

    if not rows:
        raise ValueError("Array-ul .npy nu a generat observații valide.")

    return validate_indices_dataframe(pd.DataFrame(rows))


def _inspect_npy_array(arr: np.ndarray, filename: str) -> dict:
    arr_float = arr.astype(float) if np.issubdtype(arr.dtype, np.number) else None

    try:
        inferred_index = infer_index_from_filename(filename)
    except ValueError:
        inferred_index = "nedetectat"

    info = {
        "filename": filename,
        "shape": tuple(int(x) for x in arr.shape),
        "ndim": int(arr.ndim),
        "dtype": str(arr.dtype),
        "inferred_index": inferred_index,
        "supported_for_upload": bool(arr.ndim == 3 and np.issubdtype(arr.dtype, np.number)),
    }

    if arr_float is not None:
        finite = np.isfinite(arr_float)
        info.update({
            "finite_values": int(finite.sum()),
            "nan_values": int(np.isnan(arr_float).sum()),
            "min": float(np.nanmin(arr_float)) if finite.any() else None,
            "max": float(np.nanmax(arr_float)) if finite.any() else None,
            "mean": float(np.nanmean(arr_float)) if finite.any() else None,
        })
        if arr.ndim == 3:
            info.update({
                "time_steps": int(arr.shape[0]),
                "height": int(arr.shape[1]),
                "width": int(arr.shape[2]),
                "valid_pixels": int(np.isfinite(arr_float).any(axis=0).sum()),
            })
    return info


def inspect_npy_file(file_storage) -> dict:
    """
    Inspectează fie un .npy singular, fie un .zip cu mai multe .npy.
    Pentru .zip, criteriul de identificare a indicelui este numele fișierului:
    NDVI.npy, NDMI.npy, SAVI.npy etc.
    """
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise ValueError("Nu a fost selectat niciun fișier .npy sau .zip.")

    filename = file_storage.filename
    suffix = Path(filename).suffix.lower()

    if suffix == ".npy":
        arr = np.load(file_storage, allow_pickle=False)
        return _inspect_npy_array(arr, filename)

    if suffix == ".zip":
        data = file_storage.read()
        files_info = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            npy_names = [name for name in zf.namelist() if name.lower().endswith(".npy") and not name.endswith("/")]
            if not npy_names:
                raise ValueError("Arhiva ZIP nu conține fișiere .npy.")

            for name in sorted(npy_names):
                short_name = Path(name).name
                arr = _load_npy_from_bytes(zf.read(name), short_name)
                item = _inspect_npy_array(arr, short_name)
                item["zip_path"] = name
                item["inferred_roi"] = infer_roi_from_zip_member(name, default_roi="")
                files_info.append(item)

        supported_files = [item for item in files_info if item.get("supported_for_upload")]
        indices = []
        for item in supported_files:
            inferred = item.get("inferred_index")
            if inferred and inferred != "nedetectat":
                indices.append(inferred)

        return {
            "filename": filename,
            "type": "zip_npy_collection",
            "files_count": len(files_info),
            "supported_files": len(supported_files),
            "indices_detected": sorted(set(indices)),
            "supported_for_upload": len(supported_files) > 0,
            "files": files_info,
        }

    raise ValueError("Pentru inspectare sunt acceptate doar .npy sau .zip.")

def _npy_zip_to_pixel_dataframe(
    file_storage,
    roi_name: str,
    start_date: str,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Citește o arhivă ZIP cu fișiere .npy și le convertește într-un singur CSV pixel-level.

    Reguli:
    - numele fișierului .npy definește indicele spectral: NDVI.npy, NDMI.npy etc.
    - dacă fișierul este în subfolder, ultimul folder înainte de fișier devine ROI:
        data1/NDVI.npy -> roi=data1, index=NDVI
        data2/NDMI.npy -> roi=data2, index=NDMI
    - dacă ZIP-ul este flat:
        NDVI.npy -> roi=roi_name introdus în formular
    """
    data = file_storage.read()
    frames = []
    detected_indices = []

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        npy_names = [
            name
            for name in zf.namelist()
            if name.lower().endswith(".npy")
            and not name.endswith("/")
            and "__macosx" not in name.lower()
        ]

        if not npy_names:
            raise ValueError("Arhiva ZIP nu conține fișiere .npy.")

        for name in sorted(npy_names):
            short_name = Path(name).name
            inferred_index = infer_index_from_filename(name)
            inferred_roi = infer_roi_from_zip_member(name, default_roi=roi_name)

            arr = _load_npy_from_bytes(zf.read(name), short_name)
            frame = npy_to_pixel_dataframe(
                npy_array=arr,
                roi=inferred_roi,
                index_name=inferred_index,
                start_date=start_date,
            )

            frames.append(frame)
            detected_indices.append(inferred_index)

    if not frames:
        raise ValueError("Nu s-a putut genera niciun DataFrame din arhiva ZIP.")

    return (
        validate_indices_dataframe(pd.concat(frames, ignore_index=True)),
        sorted(set(detected_indices)),
    )


def save_uploaded_dataset(
    display_name: str,
    file_storage,
    roi_name: str = "dataset_roi",
    start_date: str = "2021-01-01",
) -> dict:
    if file_storage is None or not getattr(file_storage, "filename", ""):
        raise ValueError("Nu a fost selectat niciun fișier CSV, NPY sau ZIP.")

    suffix_ext = Path(file_storage.filename).suffix.lower()
    if suffix_ext not in SUPPORTED_UPLOAD_EXTENSIONS:
        raise ValueError("Sunt acceptate doar fișiere .csv, .npy sau .zip.")

    display_name = str(display_name or "Dataset utilizator").strip()
    dataset_id_base = slugify(display_name)
    dataset_id = dataset_id_base

    existing_ids = {d["dataset_id"] for d in list_datasets(include_demo=False)}
    suffix = 2
    while dataset_id in existing_ids or dataset_id == DEMO_DATASET_ID:
        dataset_id = f"{dataset_id_base}_{suffix}"
        suffix += 1

    detected_indices = []

    if suffix_ext == ".csv":
        df = pd.read_csv(file_storage)
        df = validate_indices_dataframe(df)
        original_input_type = "csv"
        index_detection = "din coloana index a CSV-ului"
    elif suffix_ext == ".npy":
        detected_index = infer_index_from_filename(file_storage.filename)
        arr = np.load(file_storage, allow_pickle=False)
        df = npy_to_pixel_dataframe(
            npy_array=arr,
            roi=roi_name,
            index_name=detected_index,
            start_date=start_date,
        )
        original_input_type = "npy_3d"
        detected_indices = [detected_index]
        index_detection = "dedus strict din numele fișierului .npy"
    else:
        df, detected_indices = _npy_zip_to_pixel_dataframe(
            file_storage=file_storage,
            roi_name=roi_name,
            start_date=start_date,
        )
        original_input_type = "npy_zip"
        index_detection = "indice dedus din numele fișierelor; ROI dedus din subfolderele ZIP sau din câmpul implicit"

    pixel_level = is_pixel_level_dataframe(df)
    roi_df = build_roi_timeseries_dataframe(df)

    csv_text = df.to_csv(index=False)
    roi_csv_text = roi_df.to_csv(index=False)

    now = datetime.now(timezone.utc).isoformat()
    rois = sorted(df["roi"].unique().tolist())
    indices = sorted(df["index"].unique().tolist())

    record = {
        "dataset_id": dataset_id,
        "display_name": display_name,
        "source": "uploaded_dataset",
        "status": "completed",
        "input_type": "pixel_csv" if pixel_level else "roi_csv",
        "original_input_type": original_input_type,
        "index_detection": index_detection,
        "created_at": now,
        "rows": int(len(df)),
        "rois": rois,
        "indices": indices,
        "csv_path": f"user_datasets/{dataset_id}/indices_timeseries.csv",
        "roi_csv_path": f"user_datasets/{dataset_id}/{ROI_TIMESERIES_FILENAME}",
        "roi_rows": int(len(roi_df)),
    }
    manifest = dict(record)
    manifest["message"] = (
        "Dataset pixel-level validat și disponibil pentru analiză ROI-level și ML pe pixeli."
        if pixel_level
        else "Dataset ROI-level validat și disponibil pentru analiză, Cross-Index și forecast."
    )
    if detected_indices:
        manifest["detected_indices"] = detected_indices

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        bucket.blob(_gcs_dataset_blob(dataset_id)).upload_from_string(
            csv_text, content_type="text/csv; charset=utf-8"
        )
        bucket.blob(_gcs_roi_dataset_blob(dataset_id)).upload_from_string(
            roi_csv_text, content_type="text/csv; charset=utf-8"
        )
        bucket.blob(_gcs_manifest_blob(dataset_id)).upload_from_string(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )
    else:
        dataset_dir = LOCAL_DATASET_DIR / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "indices_timeseries.csv").write_text(csv_text, encoding="utf-8")
        (dataset_dir / ROI_TIMESERIES_FILENAME).write_text(roi_csv_text, encoding="utf-8")
        (dataset_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    _upsert_dataset_record(record)

    write_dataset_status(
        dataset_id=dataset_id,
        status=record["status"],
        message=(
            "Dataset încărcat și standardizat. Așteaptă pornirea jobului Cloud Run pentru ML."
            if pixel_level
            else "Dataset ROI-level încărcat și disponibil pentru analiză temporală, Cross-Index și forecast."
        ),
        extra={
            "input_type": record["input_type"],
            "indices": indices,
            "rois": rois,
            "rows": int(len(df)),
            "roi_rows": int(len(roi_df)),
            "roi_csv_path": f"user_datasets/{dataset_id}/{ROI_TIMESERIES_FILENAME}",
        },
    )

    return record

def load_dataset_dataframe(dataset_id: str) -> pd.DataFrame:
    dataset_id = normalize_dataset_id(dataset_id)
    if dataset_id == DEMO_DATASET_ID:
        raise ValueError("Datasetul demo este citit prin fișierul standard data/indices_timeseries.csv.")

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        blob = bucket.blob(_gcs_dataset_blob(dataset_id))
        if not blob.exists():
            raise FileNotFoundError(f"Datasetul {dataset_id} nu există în Cloud Storage.")
        df = pd.read_csv(io.StringIO(blob.download_as_text(encoding="utf-8")))
    else:
        path = _local_dataset_path(dataset_id)
        if not path.exists():
            raise FileNotFoundError(f"Datasetul {dataset_id} nu există local: {path}")
        df = pd.read_csv(path)
    return validate_indices_dataframe(df)


def load_dataset_roi_dataframe(dataset_id: str) -> pd.DataFrame:
    """
    Încarcă varianta ROI-level preagregată a datasetului.

    Pentru dataseturile pixel-level mari, această funcție NU trebuie să citească
    indices_timeseries.csv la fiecare request. Citește roi_timeseries.csv.
    Dacă fișierul lipsește pentru un dataset vechi, îl construiește o singură dată
    din CSV-ul complet și îl salvează pentru requesturile următoare.
    """
    dataset_id = normalize_dataset_id(dataset_id)

    if dataset_id == DEMO_DATASET_ID:
        raise ValueError("Datasetul demo este citit prin services.indices_service.")

    cached_df = _read_roi_timeseries_if_exists(dataset_id)

    if cached_df is not None:
        return cached_df

    full_df = load_dataset_dataframe(dataset_id)
    roi_df = build_roi_timeseries_dataframe(full_df)
    _write_roi_timeseries(dataset_id, roi_df)

    return roi_df


def has_pixel_level_data(dataset_id: str) -> bool:
    dataset_id = normalize_dataset_id(dataset_id)

    if dataset_id == DEMO_DATASET_ID:
        return False

    record = get_dataset_record(dataset_id)

    if record:
        return record.get("input_type") == "pixel_csv"

    try:
        return is_pixel_level_dataframe(load_dataset_dataframe(dataset_id))
    except Exception:
        return False


def get_dataset_rois(dataset_id: str) -> list[str]:
    dataset_id = normalize_dataset_id(dataset_id)

    if dataset_id == DEMO_DATASET_ID:
        return ["roi1", "roi2"]

    record = get_dataset_record(dataset_id)

    if record and record.get("rois"):
        return sorted(
            str(roi).lower()
            for roi in record.get("rois", [])
            if str(roi).strip()
        )

    df = load_dataset_roi_dataframe(dataset_id)

    return sorted(
        df["roi"]
        .dropna()
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )


def get_dataset_indices(dataset_id: str) -> list[str]:
    dataset_id = normalize_dataset_id(dataset_id)

    if dataset_id == DEMO_DATASET_ID:
        return []

    record = get_dataset_record(dataset_id)

    if record and record.get("indices"):
        return sorted(
            str(index_name).upper()
            for index_name in record.get("indices", [])
            if str(index_name).strip()
        )

    df = load_dataset_roi_dataframe(dataset_id)

    return sorted(
        df["index"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )


def get_dataset_csv_bytes(dataset_id: str) -> tuple[bytes, str]:
    dataset_id = normalize_dataset_id(dataset_id)
    if dataset_id == DEMO_DATASET_ID:
        path = Path("data") / "indices_timeseries.csv"
        if not path.exists():
            raise FileNotFoundError("Fișierul demo data/indices_timeseries.csv nu există.")
        return path.read_bytes(), "demo_indices_timeseries.csv"

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        blob = bucket.blob(_gcs_dataset_blob(dataset_id))
        if not blob.exists():
            raise FileNotFoundError(f"CSV-ul datasetului {dataset_id} nu există în Cloud Storage.")
        return blob.download_as_bytes(), f"{dataset_id}_indices_timeseries.csv"

    path = _local_dataset_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"CSV-ul datasetului {dataset_id} nu există local.")
    return path.read_bytes(), f"{dataset_id}_indices_timeseries.csv"



def delete_dataset(dataset_id: str) -> None:
    dataset_id = normalize_dataset_id(dataset_id)
    if dataset_id == DEMO_DATASET_ID:
        raise ValueError("Datasetul demonstrativ nu poate fi șters.")

    index = _read_index_gcs() if using_gcs() else _read_index_local()
    index["datasets"] = [
        d for d in index.get("datasets", [])
        if d.get("dataset_id") != dataset_id
    ]

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        prefix = f"user_datasets/{dataset_id}/"
        for blob in bucket.list_blobs(prefix=prefix):
            blob.delete()
        _write_index_gcs(index)
    else:
        import shutil
        dataset_dir = LOCAL_DATASET_DIR / dataset_id
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
        _write_index_local(index)


def _safe_metric(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return round(value, 4)
    except Exception:
        pass
    return None


def build_pixel_ml_payload_from_dataset(
    dataset_id: str,
    index_name: str,
    roi: str,
    pixel_count: int = 1000,
    n_clusters: int = 4,
) -> dict:
    """
    Construiește un payload compatibil cu /ml-features pentru CSV-uri pixel-level.
    Nu necesită imagini sau .npy; folosește coloanele date, roi, index, pixel_id, row, col, value.
    """
    df = load_dataset_dataframe(dataset_id)
    if not is_pixel_level_dataframe(df):
        raise ValueError(
            "Datasetul selectat este ROI-level. Pentru ML pe pixeli trebuie CSV pixel-level cu "
            "date, roi, index, pixel_id, row, col, value."
        )

    index_name = index_name.upper()
    roi = roi.lower()
    df = df[(df["index"] == index_name) & (df["roi"] == roi)].copy()
    if df.empty:
        raise ValueError(f"Nu există date pixel-level pentru {index_name} - {roi}.")

    wide = df.pivot_table(index="pixel_id", columns="date", values="value", aggfunc="mean").sort_index(axis=1)
    meta_pos = df.groupby("pixel_id")[["row", "col"]].first()
    common_pixels = wide.index.intersection(meta_pos.index)
    wide = wide.loc[common_pixels]
    meta_pos = meta_pos.loc[common_pixels]

    wide = wide.interpolate(axis=1).bfill(axis=1).ffill(axis=1)
    wide = wide.dropna(axis=0, how="any")
    meta_pos = meta_pos.loc[wide.index]

    if len(wide) < 4:
        raise ValueError("Sunt necesari cel puțin 4 pixeli validați pentru ML pe pixeli.")

    sample_n = min(int(pixel_count), len(wide))
    rng = np.random.default_rng(42)
    sample_ids = rng.choice(wide.index.to_numpy(), size=sample_n, replace=False)
    sample = wide.loc[sample_ids]

    X_sample = sample.values.astype(float)
    X_all = wide.values.astype(float)

    # standardizare simplă pe timp
    mean = X_sample.mean(axis=0)
    std = X_sample.std(axis=0)
    std[std == 0] = 1.0
    X_sample_scaled = (X_sample - mean) / std
    X_all_scaled = (X_all - mean) / std

    n_clusters = min(n_clusters, max(2, sample_n))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    sample_labels_zero = kmeans.fit_predict(X_sample_scaled)
    all_labels_zero = kmeans.predict(X_all_scaled)
    all_labels = all_labels_zero + 1

    iso = IsolationForest(contamination="auto", random_state=42)
    iso.fit(X_sample_scaled)
    # mai mare = mai atipic
    risk_all = -iso.decision_function(X_all_scaled)
    risk_sample = -iso.decision_function(X_sample_scaled)

    rows = meta_pos["row"].astype(int).to_numpy()
    cols = meta_pos["col"].astype(int).to_numpy()
    min_row, min_col = int(rows.min()), int(cols.min())
    rows0 = rows - min_row
    cols0 = cols - min_col
    height = int(rows0.max()) + 1
    width = int(cols0.max()) + 1

    cluster_grid = np.full((height, width), np.nan)
    risk_grid = np.full((height, width), np.nan)
    for r, c, lab, risk in zip(rows0, cols0, all_labels, risk_all):
        cluster_grid[int(r), int(c)] = int(lab)
        risk_grid[int(r), int(c)] = float(risk)

    dates = [pd.to_datetime(c).strftime("%Y-%m-%d") for c in wide.columns]

    cluster_summary = []
    cluster_profiles = []
    for cluster_id in range(1, n_clusters + 1):
        mask_all = all_labels == cluster_id
        mask_sample = (sample_labels_zero + 1) == cluster_id
        if not mask_all.any():
            continue
        values = X_all[mask_all]
        profile = values.mean(axis=0)
        mean_value = float(values.mean())
        amplitude = float(profile.max() - profile.min())
        trend = float(np.polyfit(np.arange(len(profile)), profile, 1)[0]) if len(profile) > 1 else 0.0
        risk_mean = float(risk_all[mask_all].mean())
        mapped_pixels = int(mask_all.sum())
        sample_pixels = int(mask_sample.sum())
        if risk_mean >= np.nanpercentile(risk_all, 75):
            interp = "zonă de urmărit / comportament atipic"
        elif amplitude >= np.nanpercentile([float((X_all[all_labels == c].mean(axis=0).max() - X_all[all_labels == c].mean(axis=0).min())) for c in range(1, n_clusters + 1) if (all_labels == c).any()], 75):
            interp = "sezonalitate puternică"
        elif trend < 0:
            interp = "tendință descendentă"
        else:
            interp = "comportament stabil"
        cluster_summary.append({
            "cluster": int(cluster_id),
            "sample_pixels": sample_pixels,
            "mapped_pixels": mapped_pixels,
            "windows": int(sample_pixels * max(len(dates) - 1, 1)),
            "mean": round(mean_value, 6),
            "amplitude": round(amplitude, 6),
            "trend": round(trend, 6),
            "risk_score": round(risk_mean, 6),
            "interpretation": interp,
        })
        cluster_profiles.append({
            "label": f"Cluster {cluster_id}",
            "dates": dates,
            "values": [round(float(v), 6) for v in profile],
        })

    dominant = max(cluster_summary, key=lambda r: r["mapped_pixels"])
    risk_cluster = max(cluster_summary, key=lambda r: r["risk_score"])
    seasonal_cluster = max(cluster_summary, key=lambda r: r["amplitude"])
    low_cluster = min(cluster_summary, key=lambda r: r["mean"])

    # PCA points pe eșantion
    pca_points = []
    n_components = min(3, X_sample_scaled.shape[1], X_sample_scaled.shape[0])
    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=42)
        pca_coords = pca.fit_transform(X_sample_scaled)
        if n_components == 2:
            pca_coords = np.column_stack([pca_coords, np.zeros(len(pca_coords))])
        sample_pos = meta_pos.loc[sample.index]
        for i, pixel_id in enumerate(sample.index):
            pca_points.append({
                "PC1": float(pca_coords[i, 0]),
                "PC2": float(pca_coords[i, 1]),
                "PC3": float(pca_coords[i, 2]),
                "cluster_label": f"Cluster {int(sample_labels_zero[i] + 1)}",
                "pixel_id": str(pixel_id),
                "risk_score": float(risk_sample[i]),
                "mean": float(X_sample[i].mean()),
                "amplitude": float(X_sample[i].max() - X_sample[i].min()),
                "window_start": dates[0],
                "window_end": dates[-1],
                "row": int(sample_pos.loc[pixel_id, "row"]),
                "col": int(sample_pos.loc[pixel_id, "col"]),
            })

    # DTW simplificat: distanță euclidiană între profile, ca fallback ușor pentru CSV user.
    labels = [p["label"] for p in cluster_profiles]
    profiles = [np.array(p["values"], dtype=float) for p in cluster_profiles]
    matrix = []
    for a in profiles:
        row = []
        for b in profiles:
            row.append(float(np.sqrt(np.mean((a - b) ** 2))))
        matrix.append(row)

    risk_idx = int(np.argmax(risk_all))
    rep_idx = int(np.argmin(np.abs(risk_all - np.median(risk_all))))
    risk_pixel = wide.index[risk_idx]
    rep_pixel = wide.index[rep_idx]

    # métrici clustering doar pe eșantion
    if n_clusters > 1 and len(set(sample_labels_zero)) > 1 and len(X_sample_scaled) > n_clusters:
        sil = _safe_metric(silhouette_score(X_sample_scaled, sample_labels_zero))
        ch = _safe_metric(calinski_harabasz_score(X_sample_scaled, sample_labels_zero))
        db = _safe_metric(davies_bouldin_score(X_sample_scaled, sample_labels_zero))
    else:
        sil, ch, db = None, None, None

    return {
        "metadata": {
            "dataset_id": normalize_dataset_id(dataset_id),
            "pixel_count": int(sample_n),
            "mapped_pixels": int(len(wide)),
            "windows_extracted": int(sample_n * max(len(dates) - 1, 1)),
            "n_clusters": int(n_clusters),
        },
        "metrics": {
            "silhouette": sil if sil is not None else "n/a",
            "calinski_harabasz": ch if ch is not None else "n/a",
            "davies_bouldin": db if db is not None else "n/a",
        },
        "highlights": {
            "dominant_cluster": dominant["cluster"],
            "dominant_pixels": dominant["mapped_pixels"],
            "risk_cluster": risk_cluster["cluster"],
            "risk_score": risk_cluster["risk_score"],
            "seasonal_cluster": seasonal_cluster["cluster"],
            "seasonal_amplitude": seasonal_cluster["amplitude"],
            "low_vegetation_cluster": low_cluster["cluster"],
            "low_vegetation_mean": low_cluster["mean"],
        },
        "cluster_grid": cluster_grid.tolist(),
        "risk_grid": risk_grid.tolist(),
        "cluster_profiles": cluster_profiles,
        "pca_points": pca_points,
        "tsne_points": [],
        "umap_points": [],
        "dtw": {"labels": labels, "matrix": matrix},
        "cluster_summary": cluster_summary,
        "pixel_compare": {
            "dtw": round(float(np.sqrt(np.mean((wide.loc[risk_pixel].values - wide.loc[rep_pixel].values) ** 2))), 6),
            "risk": {"dates": dates, "values": [float(v) for v in wide.loc[risk_pixel].values]},
            "representative": {"dates": dates, "values": [float(v) for v in wide.loc[rep_pixel].values]},
        },
    }

def _gcs_status_blob(dataset_id: str) -> str:
    return f"user_datasets/{normalize_dataset_id(dataset_id)}/status.json"


def _local_status_path(dataset_id: str) -> Path:
    return LOCAL_DATASET_DIR / normalize_dataset_id(dataset_id) / "status.json"


def update_dataset_status(
    dataset_id: str,
    status: str,
    message: str = "",
    extra: dict | None = None,
) -> dict:
    """
    Actualizează statusul unui dataset atât în index.json, cât și în status.json.

    Statusuri recomandate:
    - uploaded
    - processing
    - completed
    - failed
    """

    dataset_id = normalize_dataset_id(dataset_id)

    if dataset_id == DEMO_DATASET_ID:
        return {
            "dataset_id": DEMO_DATASET_ID,
            "status": "completed",
            "message": "Dataset demonstrativ.",
        }

    now = datetime.now(timezone.utc).isoformat()

    record = get_dataset_record(dataset_id) or {
        "dataset_id": dataset_id,
        "display_name": dataset_id,
        "source": "uploaded_dataset",
    }

    record["status"] = str(status or "unknown").strip().lower()
    record["status_message"] = str(message or "")
    record["status_updated_at"] = now

    if extra:
        record.update(extra)

    _upsert_dataset_record(record)

    status_payload = {
        "dataset_id": dataset_id,
        "status": record["status"],
        "message": record["status_message"],
        "updated_at": now,
    }

    if extra:
        status_payload.update(extra)

    if using_gcs():
        client = _storage_client()
        bucket = client.bucket(_bucket_name())
        bucket.blob(_gcs_status_blob(dataset_id)).upload_from_string(
            json.dumps(status_payload, ensure_ascii=False, indent=2),
            content_type="application/json; charset=utf-8",
        )
    else:
        path = _local_status_path(dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(status_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return record