import json
import os
from pathlib import Path

try:
    from google.cloud import storage
except Exception:
    storage = None

try:
    from services.dataset_service import (
        DEMO_DATASET_ID,
        normalize_dataset_id,
        user_ml_result_blob_name,
    )
except Exception:
    DEMO_DATASET_ID = "demo"

    def normalize_dataset_id(dataset_id):
        return str(dataset_id or DEMO_DATASET_ID).strip().lower() or DEMO_DATASET_ID

    def user_ml_result_blob_name(dataset_id, index_name, roi, pixel_count):
        return (
            f"user_datasets/{normalize_dataset_id(dataset_id)}/"
            f"ml/{str(index_name).upper()}/{str(roi).lower()}/"
            f"pixels_{int(pixel_count)}/result.json"
        )


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


def get_local_user_precomputed_path(
    dataset_id: str,
    index_name: str,
    roi: str,
    pixel_count: int,
) -> Path:
    return (
        Path("data")
        / "user_datasets"
        / normalize_dataset_id(dataset_id)
        / "ml"
        / index_name.upper()
        / roi.lower()
        / f"pixels_{int(pixel_count)}"
        / "result.json"
    )


def _load_json_from_gcs(blob_name: str) -> dict:
    bucket_name = os.getenv("GCS_BUCKET_NAME")

    if not bucket_name:
        raise FileNotFoundError(
            "Nu există rezultate precompute local și GCS_BUCKET_NAME nu este setat."
        )

    if storage is None:
        raise ImportError("google-cloud-storage nu este instalat.")

    tmp_path = Path("/tmp") / blob_name
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    if tmp_path.exists():
        return json.loads(tmp_path.read_text(encoding="utf-8"))

    client = storage.Client(
        project=os.getenv("GOOGLE_CLOUD_PROJECT", "plucky-environs-416709")
    )
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        raise FileNotFoundError(
            f"Rezultatul precompute nu există: gs://{bucket_name}/{blob_name}"
        )

    blob.download_to_filename(str(tmp_path))
    return json.loads(tmp_path.read_text(encoding="utf-8"))


def load_precomputed_ml_payload(
    index_name: str,
    roi: str,
    pixel_count: int,
    dataset_id: str = DEMO_DATASET_ID,
) -> dict:
    """
    Încarcă result.json pentru ML Features.

    dataset_id='demo' folosește structura existentă:
      data/precomputed_ml/<INDEX>/<roi>/pixels_<N>/result.json

    dataset_id!='demo' folosește structura user:
      user_datasets/<dataset_id>/ml/<INDEX>/<roi>/pixels_<N>/result.json
    """
    index_name = index_name.upper()
    roi = roi.lower()
    pixel_count = int(pixel_count)
    dataset_id = normalize_dataset_id(dataset_id)

    if dataset_id == DEMO_DATASET_ID:
        local_path = get_local_precomputed_path(index_name, roi, pixel_count)
        if local_path.exists():
            return json.loads(local_path.read_text(encoding="utf-8"))

        blob_name = get_precomputed_blob_name(index_name, roi, pixel_count)
        return _load_json_from_gcs(blob_name)

    local_path = get_local_user_precomputed_path(dataset_id, index_name, roi, pixel_count)
    if local_path.exists():
        return json.loads(local_path.read_text(encoding="utf-8"))

    blob_name = user_ml_result_blob_name(dataset_id, index_name, roi, pixel_count)
    return _load_json_from_gcs(blob_name)
