output "project_id" {
  value = var.project_id
}

output "bucket_name" {
  value = google_storage_bucket.spectral_arrays.name
}

output "bucket_url" {
  value = "gs://${google_storage_bucket.spectral_arrays.name}"
}

output "app_engine_url" {
  value = "https://${var.project_id}.lm.r.appspot.com"
}

output "cloud_run_job_name" {
  value = google_cloud_run_v2_job.ml_precompute.name
}

output "cloud_run_job_region" {
  value = google_cloud_run_v2_job.ml_precompute.location
}

output "artifact_registry_repository" {
  value = google_artifact_registry_repository.cloud_run_source_deploy.name
}