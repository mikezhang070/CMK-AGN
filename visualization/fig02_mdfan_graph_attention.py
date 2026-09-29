"""Paper Figure 2: verify/export the frozen MDFan graph-attention artwork."""
try:
    from ._frozen_asset_entry import run
except ImportError:
    from _frozen_asset_entry import run

if __name__ == "__main__":
    run("Fig02_MDFan_graph_attention", "Paper Figure 2: MDFan graph attention")
