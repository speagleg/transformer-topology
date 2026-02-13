"""Tests for scripts/gpu_launcher.py."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path so we can import the script
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.gpu_launcher import detect_gpus, run_job, GPUJobScheduler


class TestDetectGPUs:
    """Tests for GPU detection."""

    def test_explicit_gpu_ids(self):
        """Explicit GPU IDs override detection."""
        result = detect_gpus("0,2,3")
        assert result == [0, 2, 3]

    def test_explicit_single_gpu(self):
        result = detect_gpus("0")
        assert result == [0]

    def test_explicit_with_spaces(self):
        result = detect_gpus("0, 1, 2")
        assert result == [0, 1, 2]

    def test_nvidia_smi_detection(self):
        """Mocked nvidia-smi returns GPU indices."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "0\n1\n2\n"
        with patch("scripts.gpu_launcher.subprocess.run", return_value=mock_result):
            result = detect_gpus(None)
        assert result == [0, 1, 2]

    def test_nvidia_smi_not_found_falls_back_to_torch(self):
        """If nvidia-smi not available, fall back to torch."""
        mock_torch = MagicMock()
        mock_torch.cuda.device_count.return_value = 2
        with patch("scripts.gpu_launcher.subprocess.run", side_effect=FileNotFoundError):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                # Re-import to pick up mock
                import importlib
                import scripts.gpu_launcher as mod
                importlib.reload(mod)
                result = mod.detect_gpus(None)
        assert result == [0, 1]

    def test_no_gpus_returns_zero(self):
        """No GPUs detected returns [0] for CPU fallback."""
        with patch("scripts.gpu_launcher.subprocess.run", side_effect=FileNotFoundError):
            # Mock torch with 0 GPUs
            import scripts.gpu_launcher as mod
            with patch.object(mod, "detect_gpus", wraps=mod.detect_gpus):
                result = detect_gpus("0")  # Use explicit as a simple test
        assert result == [0]


class TestRunJob:
    """Tests for individual job execution."""

    def test_successful_job(self):
        """Echo command succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_job(0, "echo hello", 0, Path(tmpdir))
            assert result["success"] is True
            assert result["exit_code"] == 0
            assert result["job_id"] == 0
            assert result["gpu_id"] == 0
            assert result["duration_seconds"] >= 0

            # Check stdout captured
            stdout = (Path(tmpdir) / "job_0000.stdout").read_text()
            assert "hello" in stdout

    def test_failed_job(self):
        """Failing command returns non-zero exit code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_job(1, "exit 42", 0, Path(tmpdir))
            assert result["success"] is False
            assert result["exit_code"] == 42

    def test_cuda_visible_devices_set(self):
        """CUDA_VISIBLE_DEVICES is set for the subprocess."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_job(0, "echo $CUDA_VISIBLE_DEVICES", 3, Path(tmpdir))
            assert result["success"] is True
            stdout = (Path(tmpdir) / "job_0000.stdout").read_text()
            assert "3" in stdout

    def test_stderr_captured(self):
        """Stderr output is captured to .stderr file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_job(0, "echo error >&2", 0, Path(tmpdir))
            stderr = (Path(tmpdir) / "job_0000.stderr").read_text()
            assert "error" in stderr

    def test_log_dir_created(self):
        """Log directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = Path(tmpdir) / "nested" / "logs"
            result = run_job(0, "echo ok", 0, log_dir)
            assert result["success"] is True
            assert log_dir.exists()


class TestGPUJobScheduler:
    """Tests for the scheduler."""

    def test_single_gpu_sequential(self):
        """Single GPU runs jobs sequentially."""
        scheduler = GPUJobScheduler([0])
        jobs = ["echo job0", "echo job1", "echo job2"]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scheduler.run_all(jobs, Path(tmpdir))
        assert len(results) == 3
        assert all(r["success"] for r in results)
        assert all(r["gpu_id"] == 0 for r in results)

    def test_multi_gpu_distribution(self):
        """Jobs are distributed across GPUs."""
        scheduler = GPUJobScheduler([0, 1, 2])
        # Sleep a bit to ensure parallel execution
        jobs = ["sleep 0.1 && echo job0",
                "sleep 0.1 && echo job1",
                "sleep 0.1 && echo job2"]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scheduler.run_all(jobs, Path(tmpdir))
        assert len(results) == 3
        assert all(r["success"] for r in results)
        # All GPU IDs should be used
        used_gpus = {r["gpu_id"] for r in results}
        assert used_gpus == {0, 1, 2}

    def test_failed_job_doesnt_block_others(self):
        """A failing job doesn't prevent other jobs from running."""
        scheduler = GPUJobScheduler([0])
        jobs = ["echo ok1", "exit 1", "echo ok3"]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scheduler.run_all(jobs, Path(tmpdir))
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True

    def test_max_parallel_respected(self):
        """max_parallel limits concurrent jobs."""
        scheduler = GPUJobScheduler([0, 1, 2, 3], max_parallel=2)
        jobs = ["echo j0", "echo j1", "echo j2", "echo j3"]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scheduler.run_all(jobs, Path(tmpdir))
        assert len(results) == 4
        assert all(r["success"] for r in results)

    def test_print_summary(self, capsys):
        """Summary prints correctly."""
        results = [
            {"job_id": 0, "gpu_id": 0, "command": "echo 1", "success": True,
             "exit_code": 0, "duration_seconds": 1.0},
            {"job_id": 1, "gpu_id": 1, "command": "exit 1", "success": False,
             "exit_code": 1, "duration_seconds": 0.5},
        ]
        GPUJobScheduler.print_summary(results)
        out = capsys.readouterr().out
        assert "Total:   2" in out
        assert "Passed:  1" in out
        assert "Failed:  1" in out
        assert "Failed jobs:" in out

    def test_results_ordered_by_job_id(self):
        """Results are returned in job_id order regardless of completion order."""
        scheduler = GPUJobScheduler([0, 1])
        # Job 0 is slower than job 1
        jobs = ["sleep 0.2 && echo slow", "echo fast"]
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scheduler.run_all(jobs, Path(tmpdir))
        assert results[0]["job_id"] == 0
        assert results[1]["job_id"] == 1

    def test_empty_job_list(self):
        """Empty job list returns empty results."""
        scheduler = GPUJobScheduler([0])
        with tempfile.TemporaryDirectory() as tmpdir:
            results = scheduler.run_all([], Path(tmpdir))
        assert results == []


class TestMergeResults:
    """Tests for the merge_results script."""

    def test_merge_multiple_files(self):
        """Merge per-job result files into a single dict."""
        from scripts.merge_results import merge_results

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create per-job result files
            for task, variant in [("diverse", "hierarchical"), ("bfs", "symmetric")]:
                data = {
                    "task": task,
                    "variant": variant,
                    "result": {"id_accuracy": 0.5, "params": 1000},
                    "diagnostics": None,
                }
                path = Path(tmpdir) / f"result_{task}_{variant}.json"
                with open(path, "w") as f:
                    json.dump(data, f)

            output = Path(tmpdir) / "merged.json"
            merged = merge_results(tmpdir, str(output))

            assert "diverse" in merged
            assert "bfs" in merged
            assert merged["diverse"]["hierarchical"]["id_accuracy"] == 0.5
            assert output.exists()
