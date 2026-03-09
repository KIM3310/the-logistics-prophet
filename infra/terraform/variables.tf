variable "project_id" {
  description = "GCP project id."
  type        = string
}

variable "region" {
  description = "Cloud Run region."
  type        = string
  default     = "asia-northeast3"
}

variable "service_name" {
  description = "Cloud Run service name."
  type        = string
  default     = "the-logistics-prophet"
}

variable "image" {
  description = "Container image to deploy."
  type        = string
}

variable "env" {
  description = "Environment variables for the service."
  type        = map(string)
  default     = {}
}
