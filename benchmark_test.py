import time
import csv
from backend.action_mapper import ActionDispatcher

dispatcher = ActionDispatcher(cooldown_seconds=0.5)

TEST_CASES = [
    ("BROWSER", True),
    ("MEDIA", True),
    ("FILES", True),
    ("NOTES", True),
    ("CAMERA", True),
    ("INVALID_TARGET", False),
    ("RANDOM_TEXT", False)
]

def run_benchmarks():
    print("=========================================")
    print("  EVA AUTOMATED SYSTEM BENCHMARK (TEST)  ")
    print("=========================================\n")

    results = []
    
    for target, should_succeed in TEST_CASES:
        t_start = time.perf_counter()
        res = dispatcher.execute_action(target, force=True)
        t_end = time.perf_counter()

        latency_ms = round((t_end - t_start) * 1000, 2)
        is_success = (res.get("status") == "success")
        passed = (is_success == should_succeed)

        results.append({
            "target": target,
            "expected_success": should_succeed,
            "actual_status": res.get("status"),
            "latency_ms": latency_ms,
            "passed": passed
        })

        print(f"Target: {target:<15} | Status: {res.get('status'):<10} | Latency: {latency_ms} ms | Test: {'PASS' if passed else 'FAIL'}")
        time.sleep(1.0)

    # Save to CSV for the report
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["target", "expected_success", "actual_status", "latency_ms", "passed"])
        writer.writeheader()
        writer.writerows(results)

    print("\n[SUCCESS] Benchmark completed. Metrics saved to 'benchmark_results.csv'.")

if __name__ == "__main__":
    run_benchmarks()