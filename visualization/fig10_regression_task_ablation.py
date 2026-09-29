"""Paper Figure 10: regression-task modality and component ablation."""
try:
    from ._ablation_radar import build
    from ._paper_common import parser
except ImportError:
    from _ablation_radar import build
    from _paper_common import parser

if __name__ == "__main__":
    args = parser(__doc__).parse_args()
    build(
        args.root.resolve(), args.output.resolve(),
        "Fig10_Regression_task_ablation.csv", "Fig10_Regression_task_ablation",
        [("FMA_UE", "FMA hand score"), ("BI", "Barthel Index")],
        ["mae", "rmse", "r2", "pearson_r", "spearman_r"],
        ["MAE", "RMSE", "$R^2$", "Pearson r", "Spearman $\\rho$"],
        {"mae", "rmse"},
        "Regression-task modality and component ablation",
        args.source_data_dir.resolve() if args.source_data_dir else None,
    )
