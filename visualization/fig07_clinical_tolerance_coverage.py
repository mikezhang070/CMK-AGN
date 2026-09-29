"""Paper Figure 7: clinical tolerance coverage sensitivity curves."""
try:
    from ._frozen_regression_entry import run
except ImportError:
    from _frozen_regression_entry import run

if __name__ == "__main__":
    run("fig07", "Paper Figure 7: clinical tolerance coverage")
