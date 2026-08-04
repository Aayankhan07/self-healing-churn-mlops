# ChurnGuard Enterprise Cloud Infrastructure (AWS ECS Fargate + RDS PostgreSQL + S3 + ALB)
# Infrastructure as Code (IaC) configuration using Terraform

terraform {
  required_version = ">= 1.3.0"
  backend "s3" {
    bucket         = "churnguard-tf-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "churnguard-tf-locks"
    encrypt        = true
  }
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# --- AWS Secrets Manager for Database Credentials ---
resource "random_password" "db_password" {
  length  = 16
  special = false
}

resource "aws_secretsmanager_secret" "db_credentials" {
  name        = "${var.app_name}-db-secret-${var.environment}"
  description = "Managed Database Master Credentials for ChurnGuard"
}

resource "aws_secretsmanager_secret_version" "db_credentials_val" {
  secret_id     = aws_secretsmanager_secret.db_credentials.id
  secret_string = jsonencode({
    username = var.db_user
    password = random_password.db_password.result
  })
}

# --- VPC & Networking ---
resource "aws_vhc_default" "default" {}

resource "aws_security_group" "alb_sg" {
  name        = "${var.app_name}-alb-sg-${var.environment}"
  description = "Allow inbound HTTP/HTTPS traffic to ALB"

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    ="-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs_sg" {
  name        = "${var.app_name}-ecs-sg-${var.environment}"
  description = "Allow traffic from ALB to ECS task container"

  ingress {
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "rds_sg" {
  name        = "${var.app_name}-rds-sg-${var.environment}"
  description = "Allow PostgreSQL traffic from ECS tasks"

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# --- S3 Bucket for Domain Model Artifacts ---
resource "aws_s3_bucket" "model_artifacts" {
  bucket        = "${var.app_name}-model-artifacts-${var.environment}"
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "model_artifacts_versioning" {
  bucket = aws_s3_bucket.model_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

# --- RDS PostgreSQL Database ---
resource "aws_db_instance" "postgres" {
  identifier             = "${var.app_name}-db-${var.environment}"
  allocated_storage      = 20
  max_allocated_storage  = 100
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = "db.t4g.micro"
  db_name                = var.db_name
  username               = var.db_user
  password               = random_password.db_password.result
  skip_final_snapshot    = true
  vpc_security_group_ids = [aws_security_group.rds_sg.id]
}

# --- CloudWatch Log Group ---
resource "aws_cloudwatch_log_group" "ecs_logs" {
  name              = "/ecs/${var.app_name}-${var.environment}"
  retention_in_days = 30
}

# --- ECS Fargate Cluster ---
resource "aws_ecs_cluster" "main" {
  name = "${var.app_name}-cluster-${var.environment}"
}

# --- IAM Role for ECS Task Execution ---
resource "aws_iam_role" "ecs_execution_role" {
  name = "${var.app_name}-ecs-execution-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# --- ECS Task Definition ---
resource "aws_ecs_task_definition" "app" {
  family                   = "${var.app_name}-task-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn

  container_definitions = jsonencode([
    {
      name      = var.app_name
      image     = var.container_image
      essential = true
      portMappings = [
        {
          containerPort = var.container_port
          hostPort      = var.container_port
        }
      ]
      environment = [
        { name = "ENV", value = var.environment },
        { name = "DATABASE_URL", value = "postgresql://${var.db_user}:${var.db_password}@${aws_db_instance.postgres.endpoint}/${var.db_name}" },
        { name = "S3_BUCKET_NAME", value = aws_s3_bucket.model_artifacts.bucket }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs_logs.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])
}

# --- Application Load Balancer (ALB) ---
resource "aws_lb" "main" {
  name               = "${var.app_name}-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb_sg.id]
  subnets            = data.aws_subnets.default.ids
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

data "aws_vpc" "default" {
  default = true
}

resource "aws_lb_target_group" "app" {
  name        = "${var.app_name}-tg-${var.environment}"
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = data.aws_vpc.default.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 5
    timeout             = 5
    interval            = 30
    matcher             = "200"
  }
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# --- ECS Service ---
resource "aws_ecs_service" "main" {
  name            = "${var.app_name}-service-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = 2
  launch_type     = "FARGATE"

  network_configuration {
    security_groups  = [aws_security_group.ecs_sg.id]
    subnets          = data.aws_subnets.default.ids
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = var.app_name
    container_port   = var.container_port
  }

  depends_on = [aws_lb_listener.http]
}
