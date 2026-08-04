"""
Automated Latency Benchmarking Utility for ChurnGuard API.
Measures p50, p90, p95, p99 latency percentiles and Requests Per Second (RPS).
Outputs structured benchmark report to reports/latency_benchmark.json.
"""

import sys
import time
import json
import argparse
import statistics
import logging
from pathlib import Path
import urllib.request
import urllib.parse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LatencyBenchmark")


def run_latency_benchmark(host: str = "http://localhost:8000", num_requests: int = 200, api_key: str = "dev-key-change-in-prod") -> dict:
    url = f"{host.rstrip('/')}/predict?domain=telecom"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "customerID": "BENCH-001",
        "tenure": 24,
        "MonthlyCharges": 65.0,
        "TotalCharges": 1560.0,
        "SeniorCitizen": 0,
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
    }
    data_bytes = json.dumps(payload).encode("utf-8")

    latencies_ms = []
    successes = 0
    failures = 0

    logger.info(f"Starting latency benchmark on {url} with {num_requests} requests...")
    t_start_total = time.time()

    for i in range(num_requests):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    lat_ms = (time.time() - t0) * 1000.0
                    latencies_ms.append(lat_ms)
                    successes += 1
                else:
                    failures += 1
        except Exception as e:
            failures += 1

    total_duration_s = time.time() - t_start_total
    rps = round(successes / total_duration_s, 2) if total_duration_s > 0 else 0.0

    if not latencies_ms:
        logger.error("All benchmark requests failed!")
        return {"status": "failed", "failures": failures}

    latencies_ms.sort()
    n = len(latencies_ms)

    def percentile(p):
        idx = int(round(p * n)) - 1
        return round(latencies_ms[max(0, min(idx, n - 1))], 2)

    metrics = {
        "total_requests": num_requests,
        "successful_requests": successes,
        "failed_requests": failures,
        "requests_per_second": rps,
        "latency_p50_ms": percentile(0.50),
        "latency_p90_ms": percentile(0.90),
        "latency_p95_ms": percentile(0.95),
        "latency_p99_ms": percentile(0.99),
        "min_latency_ms": round(latencies_ms[0], 2),
        "max_latency_ms": round(latencies_ms[-1], 2),
        "mean_latency_ms": round(statistics.mean(latencies_ms), 2),
    }

    logger.info(f"Benchmark Complete! RPS: {rps} req/sec | p50: {metrics['latency_p50_ms']}ms | p95: {metrics['latency_p95_ms']}ms | p99: {metrics['latency_p99_ms']}ms")

    Path("reports").mkdir(exist_ok=True)
    out_path = Path("reports/latency_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved benchmark report to {out_path}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency benchmark utility.")
    parser.add_argument("--host", type=str, default="http://localhost:8000", help="API host base URL")
    parser.add_argument("--requests", type=int, default=200, help="Number of test requests")
    parser.add_argument("--key", type=str, default="dev-key-change-in-prod", help="API Key")
    args = parser.parse_args()

    run_latency_benchmark(args.host, args.requests, args.key)
