import os

from google.cloud import run_v2


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
    Pornește Cloud Run Job pentru preprocesarea asincronă a unui dataset încărcat.

    Important:
    Cloud Run override args înlocuiește args-urile definite la deploy.
    De aceea trebuie inclus explicit și scriptul Python:
    precompute_user_dataset.py
    """

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
                        "precompute_user_dataset.py",
                        "--dataset-id",
                        str(dataset_id),
                        "--pixels",
                        str(pixel_counts),
                    ]
                )
            ]
        ),
    )

    operation = client.run_job(request=request)
    return operation.operation.name