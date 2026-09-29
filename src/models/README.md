# CMK-AGN model implementations

This directory contains the predictive biosignal models used by the CMK-AGN
mainline:

- `adk_mdfan.py`: the core adaptive alignment and multimodal graph-attention
  implementation.
- `adk_mdfan_tri.py`: the tri-modal EEG/sEMG/IMU model used by the frozen
  patient-level experiments.

These models predict structured rehabilitation outcomes from multimodal
biosignals. They are separate from the downstream RehabFact language models,
which are implemented in `src/llm/`.

Frozen FMA and BI weights are stored under `checkpoints/`. Model selection is
performed only on outer-training validation data, and patient identity is the
evaluation unit. The fixed patient split is
`splits/split_patient_3fold.json`; trial-level random splitting is not part of
the frozen protocol.

Terminated exploratory SSL/contrastive follow-up experiments are not included
in this frozen release.
