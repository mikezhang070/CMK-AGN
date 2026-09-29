"""Deep-learning baselines (5 models). Each module self-registers."""
from baselines.dl_models import (  # noqa: F401
    mlp,
    cnn1d,
    eegnet,
    bilstm_attn,
    transformer_fusion,
)
