import re
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_no_figure_number_script_names(self):
        numbered = []
        for path in PROJECT_ROOT.rglob("*.py"):
            if ".git" in path.parts:
                continue
            if re.search(r"fig\d+", path.name, flags=re.IGNORECASE):
                numbered.append(path.relative_to(PROJECT_ROOT).as_posix())

        self.assertEqual(numbered, [])

    def test_workflow_directories_exist(self):
        expected_dirs = [
            "scripts/field_generation",
            "scripts/modal_analysis",
            "scripts/propagation_studies",
            "scripts/reproduction",
            "legacy_code/PYTHON/chapter_03_free_space_and_spp",
            "legacy_code/PYTHON/chapter_04_turbulence",
            "legacy_code/PYTHON/core",
        ]

        missing = [
            rel_path
            for rel_path in expected_dirs
            if not (PROJECT_ROOT / rel_path).is_dir()
        ]
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
