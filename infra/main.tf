locals {
  runtime_service_account_member = "serviceAccount:${var.runtime_service_account_email}"

  required_services = toset([
    "appengine.googleapis.com",
    "run.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "artifactregistry.googleapis.com",
    "iam.googleapis.com",
    "iamcredentials.googleapis.com"
  ])
}

resource "google_project_service" "required_apis" {
  for_each = local.required_services

  project = var.project_id
  service = each.value

  disable_on_destroy = false
}

resource "google_app_engine_application" "app" {
  project     = var.project_id
  location_id = var.app_engine_location

  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      location_id
    ]
  }

  depends_on = [
    google_project_service.required_apis
  ]
}

resource "google_storage_bucket" "spectral_arrays" {
  project  = var.project_id
  name     = var.bucket_name
  location = "EUROPE-WEST1"

  uniform_bucket_level_access = false
  force_destroy               = false

  lifecycle {
    prevent_destroy = true

    ignore_changes = [
      uniform_bucket_level_access,
    ]
  }

  depends_on = [
    google_project_service.required_apis
  ]
}

resource "google_storage_bucket_iam_member" "runtime_storage_object_admin" {
  bucket = google_storage_bucket.spectral_arrays.name
  role   = "roles/storage.objectAdmin"
  member = local.runtime_service_account_member
}

resource "google_artifact_registry_repository" "cloud_run_source_deploy" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo_id
  description   = "Repository for Cloud Run source deployments."
  format        = "DOCKER"

  lifecycle {
    prevent_destroy = true
  }

  depends_on = [
    google_project_service.required_apis
  ]
}

resource "google_cloud_run_v2_job" "ml_precompute" {
  project  = var.project_id
  name     = var.cloud_run_job_name
  location = var.region

  template {
    template {
      service_account = var.runtime_service_account_email
      timeout         = "3600s"

      containers {
        image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo_id}/${var.cloud_run_job_name}:latest"

        command = ["python"]

        args = [
          "precompute_ml.py"
        ]

        env {
          name  = "GCS_BUCKET_NAME"
          value = var.bucket_name
        }

        resources {
          limits = {
            cpu    = "4"
            memory = "8Gi"
          }
        }
      }
    }
  }

  lifecycle {
    prevent_destroy = true

    ignore_changes = [
      template[0].template[0].containers[0].image,
      template[0].template[0].containers[0].args,
      template[0].template[0].containers[0].command
    ]
  }

  depends_on = [
    google_project_service.required_apis,
    google_storage_bucket_iam_member.runtime_storage_object_admin,
    google_artifact_registry_repository.cloud_run_source_deploy
  ]
}