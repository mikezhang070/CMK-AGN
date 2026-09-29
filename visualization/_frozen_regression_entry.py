"""Adapters from the frozen regression builder to manuscript Figure 4--7 names."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

try:
    from . import _frozen_regression_figures as frozen
    from ._paper_common import parser
except ImportError:
    import _frozen_regression_figures as frozen
    from _paper_common import parser


SPECS = {
    "fig04": ("Fig01_Frozen_primary_endpoints", "Fig04_Primary_regression_endpoints"),
    "fig05": ("Fig02_FMA_patient_diagnostics", "Fig05_FMA_patient_diagnostics"),
    "fig06": ("Fig03_BI_patient_diagnostics", "Fig06_BI_patient_diagnostics"),
    "fig07": ("Fig04_Clinical_tolerance_coverage", "Fig07_Clinical_tolerance_coverage"),
}


def run(which: str, description: str) -> None:
    ap = parser(description)
    args = ap.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    frozen_root = root / "results"
    summary, patients = frozen.load_frozen_results(frozen_root)
    with tempfile.TemporaryDirectory(prefix="cmk_paper_figure_") as temp_name:
        temp = Path(temp_name)
        source_dir, final_dir = temp / "source_data", temp / "figures"
        if which == "fig04":
            frozen.fig01_endpoints(summary, source_dir, final_dir)
        elif which == "fig05":
            frozen._diagnostic_figure("FMA_UE", summary, patients["FMA_UE"], source_dir, final_dir, "02")
        elif which == "fig06":
            frozen._diagnostic_figure("BI", summary, patients["BI"], source_dir, final_dir, "03")
        elif which == "fig07":
            frozen.fig04_coverage(summary, patients, source_dir, final_dir)
        else:
            raise ValueError(which)

        source_stem, target_stem = SPECS[which]
        output.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf", ".svg"):
            shutil.copy2(final_dir / f"{source_stem}{suffix}", output / f"{target_stem}{suffix}")
        source_output = output / "source_data"
        source_output.mkdir(exist_ok=True)
        shutil.copy2(source_dir / f"{source_stem}.csv", source_output / f"{target_stem}.csv")
        print(f"rebuilt {target_stem} -> {output}")
