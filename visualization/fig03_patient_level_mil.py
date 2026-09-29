"""Paper Figure 3: verify/export the frozen patient-level MIL artwork."""
try:
    from ._frozen_asset_entry import run
except ImportError:
    from _frozen_asset_entry import run

if __name__ == "__main__":
    run("Fig03_Patient_level_MIL", "Paper Figure 3: patient-level MIL")
