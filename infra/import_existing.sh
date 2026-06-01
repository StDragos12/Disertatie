#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="plucky-environs-416709"
REGION="europe-west1"
BUCKET_NAME="ndvi-spectral-arrays-plucky-environs-416709"
ARTIFACT_REPO="cloud-run-source-deploy"
CLOUD_RUN_JOB="ndvi-ml-precompute"
SERVICE_ACCOUNT="343145196185-compute@developer.gserviceaccount.com"

gcloud config set project "${PROJECT_ID}"

terraform init

terraform import \
  google_app_engine_application.app \
  "${PROJECT_ID}" || true

terraform import \
  google_storage_bucket.spectral_arrays \
  "${PROJECT_ID}/${BUCKET_NAME}" || true

terraform import \
  google_artifact_registry_repository.cloud_run_source_deploy \
  "projects/${PROJECT_ID}/locations/${REGION}/repositories/${ARTIFACT_REPO}" || true

terraform import \
  google_cloud_run_v2_job.ml_precompute \
  "projects/${PROJECT_ID}/locations/${REGION}/jobs/${CLOUD_RUN_JOB}" || true

terraform import \
  google_storage_bucket_iam_member.runtime_storage_object_admin \
  "b/${BUCKET_NAME} roles/storage.objectAdmin serviceAccount:${SERVICE_ACCOUNT}" || true

terraform plan