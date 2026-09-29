"""Entry point used by manuscript figures whose editable drawing source is absent."""
from __future__ import annotations

try:
    from ._paper_common import copy_frozen_asset, parser
except ImportError:
    from _paper_common import copy_frozen_asset, parser


def run(stem: str, title: str) -> None:
    ap = parser(title)
    args = ap.parse_args()
    copy_frozen_asset(args.root.resolve(), args.output.resolve(), stem)
