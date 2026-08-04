"""Unit tests for performance utilities and Locust load test script structure."""

import os
from pathlib import Path
from src.onnx_exporter import export_xgboost_to_onnx, ONNXInferenceEngine
import pandas as pd


def test_locust_script_exists():
    locust_path = Path("scripts/locustfile.py")
    assert locust_path.exists()
    content = locust_path.read_text()
    assert "ChurnGuardUser" in content
    assert "predict_single" in content
    assert "predict_batch" in content


def test_benchmark_latency_script_exists():
    bench_path = Path("scripts/benchmark_latency.py")
    assert bench_path.exists()
    content = bench_path.read_text()
    assert "run_latency_benchmark" in content
    assert "latency_p95_ms" in content


def test_onnx_exporter_module(sample_dataframe):
    # Tests import and graceful fallback handling
    engine = ONNXInferenceEngine("non_existent_model.onnx")
    assert engine.session is None


def test_terraform_iac_files_exist():
    tf_dir = Path("terraform")
    assert (tf_dir / "main.tf").exists()
    assert (tf_dir / "variables.tf").exists()
    assert (tf_dir / "outputs.tf").exists()
    
    content = (tf_dir / "main.tf").read_text()
    assert "aws_ecs_service" in content
    assert "aws_db_instance" in content
    assert "aws_s3_bucket" in content


def test_prometheus_grafana_configs_exist():
    prom_path = Path("prometheus/prometheus.yml")
    assert prom_path.exists()
    assert "churnguard-api" in prom_path.read_text()

    graf_ds = Path("grafana/provisioning/datasources/prometheus.yml")
    assert graf_ds.exists()

    graf_dash = Path("grafana/dashboards/churnguard_observability.json")
    assert graf_dash.exists()
    assert "churnguard_shadow_divergence_delta" in graf_dash.read_text()
