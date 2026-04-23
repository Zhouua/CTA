from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pipeline.build_product_registry import build_product_registry


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class ProductRegistryBuilderTest(unittest.TestCase):
    def test_refresh_preserves_manual_fields_and_prefers_czce(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = root / "products"
            data_dir.mkdir()
            output_json = root / "product_registry.json"

            _write_csv(
                data_dir / "APZL.CZC.csv",
                [
                    "TDATE,CODE,MARKET,product",
                    "2024-01-01 09:00:00,APZL.CZC,CZC,AP",
                    "2024-01-02 09:00:00,APZL.CZC,CZC,AP",
                ],
            )
            _write_csv(
                data_dir / "APZL.CZCE.csv",
                [
                    "TDATE,CODE,MARKET,product",
                    "2024-01-01 09:00:00,APZL.CZCE,CZCE,AP",
                    "2024-01-03 09:00:00,APZL.CZCE,CZCE,AP",
                ],
            )
            _write_csv(
                data_dir / "AGZL.SHF.csv",
                [
                    "TDATE,CODE,MARKET,product",
                    "2024-01-01 09:00:00,AGZL.SHF,SHF,AG",
                    "2024-01-04 09:00:00,AGZL.SHF,SHF,AG",
                ],
            )
            _write_csv(
                data_dir / "BROKEN.csv",
                [
                    "DATE,CODE",
                    "2024-01-01 09:00:00,BROKEN",
                ],
            )

            output_json.write_text(
                json.dumps(
                    [
                        {
                            "product_id": "AP",
                            "category": "softs",
                            "mid_weekly_files": ["apple_inventory.csv"],
                            "enabled": False,
                            "aliases": ["OLD_AP.csv"],
                        }
                    ],
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            registry = build_product_registry(data_dir, output_json)
            self.assertEqual(len(registry), 2)

            ap = next(item for item in registry if item["product_id"] == "AP")
            self.assertEqual(ap["raw_data_file"], "APZL.CZCE.csv")
            self.assertEqual(ap["category"], "softs")
            self.assertEqual(ap["mid_weekly_files"], ["apple_inventory.csv"])
            self.assertFalse(ap["enabled"])
            self.assertIn("APZL.CZC.csv", ap["aliases"])
            self.assertIn("APZL.CZCE.csv", ap["aliases"])
            self.assertIn("OLD_AP.csv", ap["aliases"])

if __name__ == "__main__":
    unittest.main()
