"""Traditional ML baselines (5 models). Each module self-registers."""
from baselines.ml_models import (  # noqa: F401
    mean_majority,
    ridge_logreg,
    svm_rbf,
    random_forest,
    xgb,
)
