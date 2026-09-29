"""Paper Figure 5: FMA patient-level diagnostics."""
try:
    from ._frozen_regression_entry import run
except ImportError:
    from _frozen_regression_entry import run

if __name__ == "__main__":
    run("fig05", "Paper Figure 5: FMA patient diagnostics")
