"""Paper Figure 1: verify/export the frozen TRACE architecture artwork.

The editable drawing source was not present in the server release.  This
script therefore preserves the exact raster embedded in the submitted DOCX;
it does not pretend to regenerate the architecture diagram.
"""
try:
    from ._frozen_asset_entry import run
except ImportError:
    from _frozen_asset_entry import run

if __name__ == "__main__":
    run("Fig01_TRACE_overall_architecture", "Paper Figure 1: TRACE architecture")
