variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "app_name" {
  description = "Application identifier"
  type        = string
  default     = "churnguard"
}

variable "container_port" {
  description = "Port exposed by FastAPI container"
  type        = number
  default     = 8000
}

variable "container_image" {
  description = "Docker image repository URI and tag"
  type        = string
  default     = "ghcr.io/aayankhan07/churnguard-api:latest"
}

variable "db_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "churnguard"
}

variable "db_user" {
  description = "PostgreSQL master username"
  type        = string
  default     = "churnguard_admin"
}

variable "db_password" {
  description = "PostgreSQL master password"
  type        = string
  sensitive   = true
}
