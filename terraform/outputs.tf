output "alb_dns_name" {
  description = "Public DNS name of the Load Balancer"
  value       = aws_lb.main.dns_name
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster"
  value       = aws_ecs_cluster.main.name
}

output "s3_model_bucket" {
  description = "S3 bucket for storing domain model artifacts"
  value       = aws_s3_bucket.model_artifacts.bucket
}

output "rds_endpoint" {
  description = "PostgreSQL RDS connection endpoint"
  value       = aws_db_instance.postgres.endpoint
}
