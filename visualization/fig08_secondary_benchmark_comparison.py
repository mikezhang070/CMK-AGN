"""Paper Figure 8: verify/export the frozen secondary benchmark figure.

The release contains a provenance ledger for this panel, but not the complete
per-model numerical input table.  The exact submitted raster is retained and
verified instead of silently hard-coding values inferred from bar lengths.
"""
try:
    from ._frozen_asset_entry import run
except ImportError:
    from _frozen_asset_entry import run

if __name__ == "__main__":
    run("Fig08_Secondary_benchmark_comparison", "Paper Figure 8: secondary benchmark")
