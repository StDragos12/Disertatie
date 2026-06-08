import argparse
import os
import traceback

from services.dataset_service import (
    DEMO_DATASET_ID,
    get_dataset_indices,
    get_dataset_rois,
    has_pixel_level_data,
    update_dataset_status,
    build_pixel_ml_payload_from_dataset,
    upload_user_ml_result_json,
    normalize_dataset_id,
)


DEFAULT_PIXEL_COUNTS = [500, 1000, 2000, 5000]


def parse_csv(value: str) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def parse_pixels(value: str) -> list[int]:
    pixels = []
    for item in parse_csv(value):
        pixels.append(int(item))
    return pixels or DEFAULT_PIXEL_COUNTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--indices", default="")
    parser.add_argument("--rois", default="")
    parser.add_argument("--pixels", default=",".join(str(x) for x in DEFAULT_PIXEL_COUNTS))
    args = parser.parse_args()

    dataset_id = normalize_dataset_id(args.dataset_id)
    pixel_counts = parse_pixels(args.pixels)

    if dataset_id == DEMO_DATASET_ID:
        raise ValueError("Acest job este pentru dataseturi utilizator, nu pentru demo.")

    update_dataset_status(
        dataset_id=dataset_id,
        status="processing",
        message="Cloud Run Job a început preprocesarea ML pentru datasetul utilizator.",
        extra={"available_pixel_counts": pixel_counts},
    )

    try:
        if not has_pixel_level_data(dataset_id):
            raise ValueError(
                "Datasetul nu este pixel-level. Pentru ML sunt necesare coloanele "
                "date, roi, index, pixel_id, row, col, value."
            )

        indices = [x.upper() for x in parse_csv(args.indices)] if args.indices else get_dataset_indices(dataset_id)
        rois = [x.lower() for x in parse_csv(args.rois)] if args.rois else get_dataset_rois(dataset_id)

        if not indices:
            raise ValueError("Nu s-au găsit indici pentru dataset.")
        if not rois:
            raise ValueError("Nu s-au găsit ROI-uri pentru dataset.")

        generated = []

        for index_name in indices:
            for roi in rois:
                for pixel_count in pixel_counts:
                    print(
                        f"[INFO] Precompute user dataset={dataset_id} "
                        f"index={index_name} roi={roi} pixels={pixel_count}",
                        flush=True,
                    )

                    payload = build_pixel_ml_payload_from_dataset(
                        dataset_id=dataset_id,
                        index_name=index_name,
                        roi=roi,
                        pixel_count=int(pixel_count),
                    )

                    output_path = upload_user_ml_result_json(
                        dataset_id=dataset_id,
                        index_name=index_name,
                        roi=roi,
                        pixel_count=int(pixel_count),
                        payload=payload,
                    )

                    generated.append({
                        "index": index_name,
                        "roi": roi,
                        "pixels": int(pixel_count),
                        "path": output_path,
                    })

        update_dataset_status(
            dataset_id=dataset_id,
            status="completed",
            message="Preprocesarea ML a fost finalizată în Cloud Run.",
            extra={
                "available_indices": sorted(set(item["index"] for item in generated)),
                "available_rois": sorted(set(item["roi"] for item in generated)),
                "available_pixel_counts": pixel_counts,
                "generated_results": generated,
            },
        )

        print("[DONE] User dataset precompute completed.", flush=True)

    except Exception as exc:
        tb = traceback.format_exc()
        print(tb, flush=True)
        update_dataset_status(
            dataset_id=dataset_id,
            status="failed",
            message=f"Preprocesarea ML a eșuat: {exc}",
            extra={"traceback": tb[-4000:]},
        )
        raise


if __name__ == "__main__":
    main()
