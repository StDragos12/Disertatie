variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "project_number" {
  description = "Google Cloud project number"
  type        = string
}

variable "region" {
  description = "Main region"
  type        = string
  default     = "europe-west1"
}

variable "app_engine_location" {
  description = "App Engine application location (used for App Engine and Cloud Run Job)."
  type        = string
  default     = "europe-west"
}

variable "bucket_name" {
  description = "Cloud Storage bucket"
  type        = string
}

variable "artifact_registry_repo_id" {
  description = "Artifact Registry repository"
  type        = string
  default     = "cloud-run-source-deploy"
}

variable "cloud_run_job_name" {
  description = "Cloud Run Job used for ML precomputation."
  type        = string
  default     = "ndvi-ml-precompute"
}

variable "runtime_service_account_email" {
  description = "Service account used by App Engine and Cloud Run Job."
  type        = string
}