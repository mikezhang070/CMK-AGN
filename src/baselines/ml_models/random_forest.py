"""Random Forest regressor / classifier.

@article{breiman2001random,
  author = {Breiman, L.}, title = {Random Forests},
  journal = {Machine Learning}, volume = {45}, number={1},
  year = {2001}, pages = {5--32}
}
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from baselines.ml_wrappers import MLBaseline
from baselines.registry import register


class RandomForestBaseline(MLBaseline):
    def __init__(self, spec, args):
        super().__init__(spec, args, name="random_forest")
        if spec.task_type == "regression":
            self.model = RandomForestRegressor(
                n_estimators=100, random_state=args.seed, n_jobs=1
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100, random_state=args.seed, n_jobs=1,
                class_weight="balanced",
            )


@register("random_forest")
def build(spec, args):
    return RandomForestBaseline(spec, args)
