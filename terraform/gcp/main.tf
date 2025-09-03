resource "google_vertex_ai_model" "housing_model" {
  display_name = "housing-model"
  description  = "Housing price prediction model"
  
  container_spec {
    image_uri = "gcr.io/${var.project_id}/housing-model:latest"
  }
}

resource "google_vertex_ai_endpoint" "housing_endpoint" {
  display_name = "housing-endpoint"
  location     = var.region
}
