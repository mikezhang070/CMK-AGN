"""Run manuscript Figure 1--11 scripts in paper order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPTS = [
    "fig01_trace_overall_architecture.py",
    "fig02_mdfan_graph_attention.py",
    "fig03_patient_level_mil.py",
    "fig04_primary_regression_endpoints.py",
    "fig05_fma_patient_diagnostics.py",
    "fig06_bi_patient_diagnostics.py",
    "fig07_clinical_tolerance_coverage.py",
    "fig08_secondary_benchmark_comparison.py",
    "fig09_ordinal_confusion_matrices.py",
    "fig10_regression_task_ablation.py",
    "fig11_ordinal_task_ablation.py",
]


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    output = root / "figures" / "rebuilt"
    for script in SCRIPTS:
        print(f"[paper-figures] {script}", flush=True)
        subprocess.run([sys.executable, str(here / script), "--root", str(root), "--output", str(output)], check=True)


if __name__ == "__main__":
    main()
