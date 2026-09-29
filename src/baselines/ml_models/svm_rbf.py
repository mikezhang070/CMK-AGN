"""Support Vector Machine / Regression with RBF kernel.

@article{cortes1995support,
  author = {Cortes, C. and Vapnik, V.},
  title  = {Support-vector networks}, journal = {Machine Learning},
  volume = {20}, number={3}, year={1995}, pages={273--297}
}
@inproceedings{drucker1997svr,
  author = {Drucker, H. and Burges, C. J. C. and Kaufman, L. and
            Smola, A. and Vapnik, V.},
  title  = {Support Vector Regression Machines},
  booktitle = {NIPS}, year = {1997}
}
"""
from __future__ import annotations

from sklearn.svm import SVC, SVR

from baselines.ml_wrappers import MLBaseline
from baselines.registry import register


class SVMRBFBaseline(MLBaseline):
    def __init__(self, spec, args):
        super().__init__(spec, args, name="svm_rbf")
        if spec.task_type == "regression":
            self.model = SVR(kernel="rbf", C=1.0, gamma="scale")
        else:
            self.model = SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=args.seed,
            )


@register("svm_rbf")
def build(spec, args):
    return SVMRBFBaseline(spec, args)
