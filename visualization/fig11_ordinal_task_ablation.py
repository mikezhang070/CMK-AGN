"""Paper Figure 11: ordinal-task modality and component ablation."""
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
        "Fig11_Ordinal_task_ablation.csv", "Fig11_Ordinal_task_ablation",
        [("hand_tone", "Hand muscle tone"), ("hand_function", "Brunnstrom hand stage")],
        ["accuracy", "macro_f1", "cohen_kappa", "weighted_kappa"],
        ["Accuracy", "Macro-F1", "Cohen $\\kappa$", "Weighted $\\kappa$"],
        set(),
        "Ordinal-task modality and component ablation",
        args.source_data_dir.resolve() if args.source_data_dir else None,
    )
