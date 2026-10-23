"""Worker cancellation includes Windows venv interpreter children."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app_runtime import reap_worker, terminate_worker


class WorkerLifecycleTests(unittest.TestCase):
    def process(self):
        process = Mock(pid=12345)
        process.poll.return_value = None
        return process

    def test_windows_cancellation_targets_the_whole_process_tree(self):
        process = self.process()
        with patch("app_runtime.os.name", "nt"), patch("app_runtime.subprocess.run", return_value=Mock(returncode=0)) as run:
            reap_worker(process)
        self.assertEqual(run.call_args.args[0], ["taskkill", "/PID", "12345", "/T", "/F"])
        process.terminate.assert_not_called()
        process.wait.assert_called_once_with(timeout=8)

    def test_failed_tree_kill_falls_back_and_reaps(self):
        process = self.process()
        with patch("app_runtime.os.name", "nt"), patch("app_runtime.subprocess.run", side_effect=OSError("not found")):
            reap_worker(process)
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=8)

    def test_hung_worker_is_killed_after_bounded_wait(self):
        process = self.process()
        process.wait.side_effect = [subprocess.TimeoutExpired("worker", 8), 0]
        with patch("app_runtime.os.name", "posix"):
            reap_worker(process)
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_finished_worker_is_not_signalled(self):
        process = self.process()
        process.poll.return_value = 0
        with patch("app_runtime.subprocess.run") as run:
            terminate_worker(process)
        run.assert_not_called()
        process.terminate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
