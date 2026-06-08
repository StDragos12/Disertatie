import os

try:
    from google.cloud import run_v2
except Exception:
    run_v2 = None


PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "plucky-environs-416709")
REGION = os.getenv("CLOUD_RUN_REGION", "europe-west1")
USER_PRECOMPUTE_JOB_NAME = os.getenv(
    "USER_PRECOMPUTE_JOB_NAME",
    "ndvi-user-dataset-precompute",
)


def trigger_user_dataset_precompute(
    dataset_id: str,
    pixel_counts: str = "500,1000,2000,5000",
) -> str:
    """
    Pornește Cloud Run Job pentru precomputarea ML a unui dataset încărcat de utilizator.

    App Engine rămâne doar interfață web/upload.
    Cloud Run Job citește CSV-ul standardizat din Cloud Storage și scrie result.json.
    """
    if run_v2 is None:
        raise ImportError(
            "google-cloud-run nu este instalat. Adaugă google-cloud-run în requirements.txt."
        )

    dataset_id = str(dataset_id).strip()
    if not dataset_id:
        raise ValueError("dataset_id este obligatoriu pentru pornirea Cloud Run Job.")

    client = run_v2.JobsClient()

    job_resource_name = (
        f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{USER_PRECOMPUTE_JOB_NAME}"
    )

    request = run_v2.RunJobRequest(
        name=job_resource_name,
        overrides=run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    args=[
                        "--dataset-id",
                        dataset_id,
                        "--pixels",
                        str(pixel_counts),
                    ]
                )
            ]
        ),
    )

    operation = client.run_job(request=request)
    return operation.operation.name
