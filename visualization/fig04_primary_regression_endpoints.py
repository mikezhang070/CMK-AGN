"""Paper Figure 4: primary patient-independent regression endpoints."""
try:
    from ._frozen_regression_entry import run
except ImportError:
    from _frozen_regression_entry import run

if __name__ == "__main__":
    run("fig04", "Paper Figure 4: primary regression endpoints")
