"""Paper Figure 6: Barthel Index patient-level diagnostics."""
try:
    from ._frozen_regression_entry import run
except ImportError:
    from _frozen_regression_entry import run

if __name__ == "__main__":
    run("fig06", "Paper Figure 6: BI patient diagnostics")
