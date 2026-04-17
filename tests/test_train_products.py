from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from code.train_products import (
    annotate_products_for_batch_skip,
    execute_training_session,
    split_resume_products,
    train_selected_products,
    write_run_outputs,
)


class TrainProductsRunnerTest(unittest.TestCase):
    def test_batch_training_continues_on_failures_and_skips_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "20240101_000000"

            def _executor(product_meta, config_path, run_dir_arg, force_rebuild):
                if product_meta["product_id"] == "B":
                    raise RuntimeError("boom")
                return {
                    "product_id": product_meta["product_id"],
                    "status": "success",
                    "product_dir": str(run_dir_arg / product_meta["product_id"]),
                }

            results, failures = train_selected_products(
                product_records=[
                    {"product_id": "A", "enabled": True},
                    {"product_id": "B", "enabled": True},
                    {"product_id": "C", "enabled": False},
                ],
                config_path=None,
                run_dir=run_dir,
                force_rebuild=False,
                executor=_executor,
            )

            status_map = {row["product_id"]: row["status"] for row in results}
            self.assertEqual(status_map["A"], "success")
            self.assertEqual(status_map["B"], "failed")
            self.assertEqual(status_map["C"], "skipped_disabled")
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["product_id"], "B")

    def test_write_run_outputs_emits_manifest_and_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "20240101_000001"
            results = [
                {"product_id": "A", "status": "success"},
                {"product_id": "B", "status": "failed"},
                {"product_id": "C", "status": "skipped_disabled"},
            ]
            failures = [{"product_id": "B", "error": "boom"}]

            manifest = write_run_outputs(
                run_dir=run_dir,
                run_id="20240101_000001",
                config_path=None,
                requested_products=["A", "B", "C"],
                results=results,
                failures=failures,
            )

            self.assertEqual(manifest["success_count"], 1)
            self.assertEqual(manifest["failure_count"], 1)
            self.assertEqual(manifest["skipped_count"], 1)
            self.assertEqual(manifest["pending_count"], 0)
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((run_dir / "run_summary.csv").exists())
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertEqual(json.loads((run_dir / "failed_products.json").read_text()), failures)
            self.assertTrue((run_dir / "insufficient_coverage_products.json").exists())

    def test_split_resume_products_keeps_only_successful_rows(self) -> None:
        retained_results, pending_records = split_resume_products(
            product_records=[
                {"product_id": "A", "enabled": True},
                {"product_id": "B", "enabled": True},
                {"product_id": "C", "enabled": False},
            ],
            existing_results=[
                {"product_id": "A", "status": "success"},
                {"product_id": "B", "status": "failed"},
                {"product_id": "C", "status": "skipped_disabled"},
            ],
        )

        self.assertEqual([row["product_id"] for row in retained_results], ["A"])
        self.assertEqual([row["product_id"] for row in pending_records], ["B", "C"])

    def test_annotate_products_for_batch_skip_marks_insufficient_coverage(self) -> None:
        annotated = annotate_products_for_batch_skip(
            [
                {"product_id": "A", "data_start": "2020-01-01 00:00:00", "data_end": "2026-02-01 00:00:00"},
                {"product_id": "B", "data_start": "2023-01-01 00:00:00", "data_end": "2026-02-01 00:00:00"},
                {"product_id": "C", "data_start": "2020-01-01 00:00:00", "data_end": "2025-06-30 00:00:00"},
            ],
            enforce_registry_coverage=True,
            required_data_start="2021-01-01",
            required_data_end="2026-01-01",
        )

        status_map = {row["product_id"]: row.get("_batch_skip_status") for row in annotated}
        self.assertIsNone(status_map["A"])
        self.assertEqual(status_map["B"], "skipped_insufficient_coverage")
        self.assertEqual(status_map["C"], "skipped_insufficient_coverage")
        self.assertIn("available_range=2023-01-01 to 2026-02-01", annotated[1]["_batch_skip_error"])
        self.assertIn("required_start=2021-01-01", annotated[1]["_batch_skip_error"])
        self.assertIn("required_end=2026-01-01", annotated[2]["_batch_skip_error"])

    def test_execute_training_session_resumes_only_missing_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "20240101_000002"
            executor_calls: list[str] = []

            def _executor(product_meta, config_path, run_dir_arg, force_rebuild):
                executor_calls.append(product_meta["product_id"])
                return {
                    "product_id": product_meta["product_id"],
                    "status": "success",
                    "sharpe": 1.25,
                    "total_return": 0.11,
                    "product_dir": str(run_dir_arg / product_meta["product_id"]),
                }

            results, failures, manifest = execute_training_session(
                product_records=[
                    {"product_id": "A", "enabled": True},
                    {"product_id": "B", "enabled": True},
                    {"product_id": "C", "enabled": False},
                ],
                config_path=None,
                run_dir=run_dir,
                run_id="20240101_000002",
                requested_products=["__all__"],
                existing_results=[{"product_id": "A", "status": "success", "product_dir": str(run_dir / "A")}],
                resume_from=str(run_dir),
                executor=_executor,
                logger=lambda message: None,
            )

            status_map = {row["product_id"]: row["status"] for row in results}
            self.assertEqual(executor_calls, ["B"])
            self.assertEqual(status_map["A"], "success")
            self.assertEqual(status_map["B"], "success")
            self.assertEqual(status_map["C"], "skipped_disabled")
            self.assertEqual(failures, [])
            self.assertEqual(manifest["success_count"], 2)
            self.assertEqual(manifest["skipped_count"], 1)
            self.assertEqual(manifest["pending_count"], 0)
            self.assertEqual(manifest["selected_products"], ["A", "B", "C"])
            self.assertEqual(json.loads((run_dir / "run_summary.json").read_text())[0]["product_id"], "A")

    def test_execute_training_session_skips_insufficient_coverage_products(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "runs" / "20240101_000003"
            executor_calls: list[str] = []

            def _executor(product_meta, config_path, run_dir_arg, force_rebuild):
                executor_calls.append(product_meta["product_id"])
                return {
                    "product_id": product_meta["product_id"],
                    "status": "success",
                    "product_dir": str(run_dir_arg / product_meta["product_id"]),
                }

            product_records = annotate_products_for_batch_skip(
                [
                    {"product_id": "A", "enabled": True, "data_start": "2020-01-01 00:00:00", "data_end": "2026-02-01 00:00:00"},
                    {"product_id": "B", "enabled": True, "data_start": "2023-01-01 00:00:00", "data_end": "2026-02-01 00:00:00"},
                ],
                enforce_registry_coverage=True,
                required_data_start="2021-01-01",
                required_data_end="2026-01-01",
            )

            results, failures, manifest = execute_training_session(
                product_records=product_records,
                config_path=None,
                run_dir=run_dir,
                run_id="20240101_000003",
                requested_products=["A", "B"],
                executor=_executor,
                logger=lambda message: None,
            )

            status_map = {row["product_id"]: row["status"] for row in results}
            self.assertEqual(executor_calls, ["A"])
            self.assertEqual(status_map["A"], "success")
            self.assertEqual(status_map["B"], "skipped_insufficient_coverage")
            self.assertEqual(failures, [])
            self.assertEqual(manifest["success_count"], 1)
            self.assertEqual(manifest["skipped_count"], 1)
            insufficient_rows = json.loads((run_dir / "insufficient_coverage_products.json").read_text())
            self.assertEqual(insufficient_rows[0]["product_id"], "B")
            self.assertEqual(insufficient_rows[0]["available_data_start"], "2023-01-01 00:00:00")
            self.assertEqual(insufficient_rows[0]["available_data_end"], "2026-02-01 00:00:00")


if __name__ == "__main__":
    unittest.main()
