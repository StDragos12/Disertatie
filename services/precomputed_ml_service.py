import json
import os
from pathlib import Path

try:
    from google.cloud import storage
except Exception:
    storage = None


DEFAULT_PIXEL_COUNTS = [500, 1000, 2000, 5000]


def get_precomputed_blob_name(index_name: str, roi: str, pixel_count: int) -> str:
    return (
        f"data/precomputed_ml/"
        f"{index_name.upper()}/"
        f"{roi.lower()}/"
        f"pixels_{int(pixel_count)}/"
        f"result.json"
    )


def get_local_precomputed_path(index_name: str, roi: str, pixel_count: int) -> Path:
    return (
        Path("data")
        / "precomputed_ml"
        / index_name.upper()
        / roi.lower()
        / f"pixels_{int(pixel_count)}"
        / "result.json"
    )


def load_precomputed_ml_payload(index_name: str, roi: str, pixel_count: int) -> dict:
    index_name = index_name.upper()
    roi = roi.lower()
    pixel_count = int(pixel_count)

    local_path = get_local_precomputed_path(index_name, roi, pixel_count)

    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))

    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not bucket_name:
        raise FileNotFoundError(
            "Nu există rezultate precompute local și GCS_BUCKET_NAME nu este setat."
        )

    if storage is None:
        raise ImportError(
            "google-cloud-storage nu este instalat."
        )

    tmp_path = (
        Path("/tmp")
        / "precomputed_ml"
        / index_name
        / roi
        / f"pixels_{pixel_count}"
        / "result.json"
    )

    tmp_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if tmp_path.exists():
        return json.loads(tmp_path.read_text(encoding="utf-8"))

    blob_name = get_precomputed_blob_name(
        index_name,
        roi,
        pixel_count
    )

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(
            f"Rezultatul precompute nu există: gs://{bucket_name}/{blob_name}"
        )

    blob.download_to_filename(str(tmp_path))

    return json.loads(tmp_path.read_text(encoding="utf-8"))