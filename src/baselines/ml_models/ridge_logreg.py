"""Ridge regression / logistic regression.

@article{hoerl1970ridge,
  author = {Hoerl, A. E. and Kennard, R. W.},
  title  = {Ridge Regression: Biased Estimation for Nonorthogonal Problems},
  journal = {Technometrics}, volume={12}, number={1}, year={1970}, pages={55--67}
}
@book{hosmer2013applied,
  author = {Hosmer, D. W. and Lemeshow, S. and Sturdivant, R. X.},
  title  = {Applied Logistic Regression}, year={2013}, publisher={Wiley}
}
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression, Ridge

from baselines.ml_wrappers import MLBaseline
from baselines.registry import register


class RidgeLogregBaseline(MLBaseline):
    def __init__(self, spec, args):
        super().__init__(spec, args, name="ridge_logreg")
        if spec.task_type == "regression":
            self.model = Ridge(alpha=1.0, random_state=args.seed)
        else:
            self.model = LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=args.seed,
                multi_class="auto",
            )


@register("ridge_logreg")
def build(spec, args):
    return RidgeLogregBaseline(spec, args)
