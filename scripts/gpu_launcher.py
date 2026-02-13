"""Generic multi-GPU job launcher.

Dispatches a list of shell commands across available GPUs using a thread pool.
Each job gets exclusive CUDA_VISIBLE_DEVICES assignment.

Usage:
    python scripts/gpu_launcher.py --jobs jobs.txt --log-dir logs/ [--gpus 0,1,2] [--max-parallel 4]
"""

import argparse
import os
import queue
import subprocess
import threading
import time
from pathlib import Path


def detect_gpus(gpu_ids: str | None = None) -> list[int]:
    """Detect available GPUs.

    If gpu_ids provided (e.g. "0,2,3"), restrict to those.
    Otherwise detect via nvidia-smi, fallback to torch.cuda.device_count().
    """
    if gpu_ids is not None:
        return [int(x.strip()) for x in gpu_ids.split(",") if x.strip()]

    # Try nvidia-smi first (no torch dependency needed)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(x.strip()) for x in result.stdout.strip().split("\n")]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback to torch
    try:
        import torch
        count = torch.cuda.device_count()
        if count > 0:
            return list(range(count))
    except ImportError:
        pass

    # No GPUs found — return [0] for CPU-only (CUDA_VISIBLE_DEVICES="" effectively)
    return [0]


def run_job(job_id: int, command: str, gpu_id: int, log_dir: Path) -> dict:
    """Run a single command with CUDA_VISIBLE_DEVICES=gpu_id.

    Captures stdout/stderr to log files.
    Returns dict with job_id, gpu_id, command, success, exit_code, duration_seconds.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"job_{job_id:04d}.stdout"
    stderr_path = log_dir / f"job_{job_id:04d}.stderr"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    start = time.time()
    try:
        with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
            proc = subprocess.run(
                command, shell=True, env=env,
                stdout=out, stderr=err,
            )
        exit_code = proc.returncode
    except Exception as e:
        exit_code = -1
        with open(stderr_path, "a") as err:
            err.write(f"\nLauncher error: {e}\n")

    duration = time.time() - start

    return {
        "job_id": job_id,
        "gpu_id": gpu_id,
        "command": command,
        "success": exit_code == 0,
        "exit_code": exit_code,
        "duration_seconds": round(duration, 1),
    }


class GPUJobScheduler:
    """Schedule jobs across a pool of GPUs using a thread pool."""

    def __init__(self, gpu_ids: list[int], max_parallel: int | None = None):
        self.gpu_ids = gpu_ids
        self.max_parallel = max_parallel or len(gpu_ids)

        self._gpu_pool: queue.Queue[int] = queue.Queue()
        for gid in gpu_ids:
            self._gpu_pool.put(gid)

        self._lock = threading.Lock()
        self._completed = 0
        self._total = 0

    def _run_one(self, job_id: int, command: str, log_dir: Path) -> dict:
        """Acquire a GPU, run job, release GPU."""
        gpu_id = self._gpu_pool.get()
        try:
            with self._lock:
                print(f"  [{self._completed}/{self._total} done] "
                      f"Starting job {job_id} on GPU {gpu_id}")
            result = run_job(job_id, command, gpu_id, log_dir)
            with self._lock:
                self._completed += 1
                status = "OK" if result["success"] else f"FAIL(rc={result['exit_code']})"
                print(f"  [{self._completed}/{self._total} done] "
                      f"Job {job_id} {status} ({result['duration_seconds']}s)")
            return result
        finally:
            self._gpu_pool.put(gpu_id)

    def run_all(self, jobs: list[str], log_dir: Path) -> list[dict]:
        """Run all jobs with GPU-aware scheduling.

        Returns list of result dicts.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._completed = 0
        self._total = len(jobs)
        log_dir.mkdir(parents=True, exist_ok=True)

        print(f"Launching {len(jobs)} jobs across {len(self.gpu_ids)} GPUs "
              f"(max_parallel={self.max_parallel})")

        results = [None] * len(jobs)
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {
                pool.submit(self._run_one, i, cmd, log_dir): i
                for i, cmd in enumerate(jobs)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results

    @staticmethod
    def print_summary(results: list[dict]):
        """Print a summary table of job results."""
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        failed = total - passed
        durations = [r["duration_seconds"] for r in results]
        avg_time = sum(durations) / total if total else 0
        total_time = sum(durations)

        print(f"\n{'=' * 60}")
        print(f"Job Summary")
        print(f"{'=' * 60}")
        print(f"  Total:   {total}")
        print(f"  Passed:  {passed}")
        print(f"  Failed:  {failed}")
        print(f"  Avg time: {avg_time:.1f}s")
        print(f"  Total time: {total_time:.1f}s (wall clock with parallelism)")

        if failed > 0:
            print(f"\nFailed jobs:")
            for r in results:
                if not r["success"]:
                    print(f"  Job {r['job_id']} (GPU {r['gpu_id']}): "
                          f"exit code {r['exit_code']}")
                    print(f"    Command: {r['command']}")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU job launcher")
    parser.add_argument("--jobs", required=True, help="File with one command per line")
    parser.add_argument("--log-dir", required=True, help="Directory for job logs")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU IDs (e.g. 0,1,2)")
    parser.add_argument("--max-parallel", type=int, default=None,
                        help="Max concurrent jobs (default: num GPUs)")
    args = parser.parse_args()

    gpu_ids = detect_gpus(args.gpus)
    print(f"GPUs: {gpu_ids}")

    jobs_path = Path(args.jobs)
    jobs = [line.strip() for line in jobs_path.read_text().splitlines() if line.strip()]
    if not jobs:
        print("No jobs to run.")
        return

    scheduler = GPUJobScheduler(gpu_ids, max_parallel=args.max_parallel)
    results = scheduler.run_all(jobs, Path(args.log_dir))
    scheduler.print_summary(results)

    # Exit with non-zero if any jobs failed
    if any(not r["success"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
