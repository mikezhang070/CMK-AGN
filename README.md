![Logo](logo.png)

# CMK-AGN 

**简体中文** ｜ [English](#english-version)

<a id="中文版"></a>

本项目由珠海复旦创新研究院的医学人工智能科技创新中心研发团队开发。CMK-AGN 是一个三模态深度学习模型，将一次康复试次中同步采集的 **EEG · sEMG · IMU** 信号映射为四项相互独立的临床评分。各模态先经加权贝叶斯 DTW（EMG↔IMU）加线性重采样的 EEG 进行对齐，再由 CMK-AGN 主干网络融合；每个任务训练一个独立的输出头（不共享输出头，也不使用联合损失）。患者是唯一的评估单元，全部实验使用固定的患者级三折划分，严禁试次级随机划分。

本包是真实 29 例临床数据上的**冻结研究发布版**：包含最终源代码、18 个经审计的 FMA/BI checkpoint、冻结数值结果、论文正文 Figure 1–11 成品图、可运行的绘图代码，以及三个实验入口脚本。已终止的探索性实验分支不属于本包。

> **命名说明。** 模型对外名称为 **CMK-AGN**。出于向后兼容，Python 模块内部仍沿用旧标识符 `adk_mdfan`（例如 `src/models/adk_mdfan_tri.py`、类 `ADKMDFANTriBackbone`）。二者指的是同一套 CMK-AGN 架构。

---

## 临床任务

每个任务作为独立模型训练。

| 任务键 | 临床量表 | 类型 | 取值范围 / 类别 | 预设容差 |
|---|---|---|---|---|
| `FMA_UE` | FMA-UE 手部子分 | 回归 | 0 – 20（整数） | \|error\| ≤ 1.5 |
| `BI` | Barthel 指数 | 回归 | 0 – 100（步长 5） | \|error\| ≤ 10 |
| `hand_tone` | 手部 MAS（改良 Ashworth） | 6 类有序 | 0, 1, 1+, 2, 3, 4 | — |
| `hand_function` | Brunnstrom 手部分期 | 6 类有序 | 1 – 6 期 | — |

代码层的键名 `hand_tone` / `hand_function` 是为兼容 manifest 而保留的；临床上分别对应手部 MAS 与 Brunnstrom 手部分期。有序任务默认使用 CORN 序数头（`src/task_config.py`）。

---

## 冻结主线结果

由本包 18 个 checkpoint 支撑的患者级结果（`results/summary.csv`，n = 29）：

| 任务 | MAE | RMSE | R² | Spearman | 预设容差命中 |
|---|---:|---:|---:|---:|---:|
| FMA | 2.517 | 3.821 | 0.750 | 0.831 | 16/29（55.2%） |
| BI | 12.148 | 15.845 | 0.541 | 0.713 | 15/29（51.7%） |

- `results/FMA/`、`results/BI/`：主线冻结结果（患者级预测、逐折/逐种子指标、bootstrap 置信区间、选择参数）。
- `results/MAS/`、`results/Brunnstrom/`：论文补充的分类结果，**不随包发布权重**。
- `results/reference_comparison.csv`：历史 OOF 选择口径的对照参考，其原始患者级结果包未恢复，仅作溯源证据，不是第二个冻结结果。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

原始信号**未**包含在本包中（见下文 _数据_ 一节）。训练 manifest 须指向获得授权的 EEG / sEMG / IMU 文件，并通过环境变量传入：

```bash
export MANIFEST=/secure/authorized/samples_manifest_real_archive.csv
```

两个信号实验脚本共用固定患者级划分 `splits/split_patient_3fold.json`（仅含匿名 fold 成员关系）。

### 3. 冻结主线实验（FMA / BI）

```bash
MANIFEST=/secure/authorized/samples_manifest_real_archive.csv \
  bash scripts/run_frozen_experiment.sh
```

- 2 任务 × 3 种子（2024 / 42 / 1337）× 3 折，checkpoint 按外层训练验证集 MAE 选择。
- 输出写入 `runs/frozen_experiment/`，不会覆盖包内冻结的 `results/`、`figures/`、`checkpoints/`。
- `DRY_RUN=1` 只打印命令不训练；`DEVICE`、`EPOCHS`、`PATIENCE` 等可用环境变量覆盖。

### 4. 完整消融与多种子实验

```bash
MANIFEST=/secure/authorized/samples_manifest_real_archive.csv \
  bash scripts/run_ablation_multiseed_figures.sh
```

- 4 任务全模型 × 3 种子，6 种模态消融（EEG / EMG / IMU / EEG+EMG / EEG+IMU / EMG+IMU），3 种模块消融（去 MDFan / 去加权 DTW / 去三模态对齐），全部三折，共 48 次训练调用（144 个 fold 拟合）。
- 回归 checkpoint 按验证 MAE 选择；序数分类 checkpoint 按验证 weighted kappa 选择。
- 自动汇总消融表并重新生成 Figure 10 / 11 的源数据与 PNG/PDF/SVG，输出至 `runs/ablation_multiseed/`。
- `STAGE=train|aggregate|figures` 可单阶段重跑；`DRY_RUN=1 STAGE=train` 打印全部训练命令。

### 5. RehabFact 六基座模型对比实验

下游文本报告模块 RehabFact 在六种开源指令模型上做了患者不相交的三折对比（选择指标 `all.char_bleu4`，三折均值 ± 标准差；每折测试集为 5/4/4 名真实参考文本患者）：

| 排名 | 模型 ID | 基座模型 | char-BLEU4 | ROUGE-L F | 结果 |
|---:|---|---|---:|---:|---|
| 1 | `baichuan2_7b` | Baichuan2-7B-Chat | **47.29 ± 5.33** | **61.24 ± 3.72** | **最终选中** |
| 2 | `qwen3_8b` | Qwen3-8B | 42.49 ± 5.61 | 55.55 ± 5.56 | 未选中 |
| 3 | `gemma3_4b` | Gemma-3-4B-IT | 37.71 ± 5.83 | 52.67 ± 6.29 | 未选中 |
| 4 | `glm4_9b_chat` | GLM-4-9B-Chat | 27.54 ± 0.80 | 45.34 ± 1.81 | 未选中 |
| 5 | `llama31_8b` | Meta-Llama-3.1-8B-Instruct | 21.27 ± 2.92 | 36.87 ± 3.46 | 未选中 |
| 6 | `deepseek_r1_llama8b` | DeepSeek-R1-Distill-Llama-8B | 0.43 ± 0.06 | 1.46 ± 0.19 | 未选中 |

Baichuan2-7B-Chat 以 4.80 的 char-BLEU4 优势胜过亚军 Qwen3-8B。选中后，RehabFact 继续完成 `human_only`（仅医生文本训练）与 `human_plus_aux`（加入仅训练侧的规则辅助文本）的辅助文本消融（`results/RehabFact/aux_ablation_summary.csv`）：char-BLEU4 从 32.55 提升至 37.28，ROUGE-L F 从 44.29 提升至 48.15，事实正确率与安全性核查项保持 1.0。注意：模型选择阶段与辅助文本消融阶段是两批独立实验，数值不可直接横向比较。

```bash
# 重新运行固定六模型对比（写入 runs/，不覆盖冻结结果）：
CORPUS=/secure/authorized/patient_rehab_29subjects.json \
  bash scripts/run_rehabfact_six_llm_experiment.sh run

# 从既有评估树重建汇总：
RUN_ROOT=runs/rehabfact_six_model_comparison \
  bash scripts/run_rehabfact_six_llm_experiment.sh aggregate

# 从服务器原始 results/llm 导出历史评估证据（默认不含逐病例文本）：
SOURCE_RESULTS_ROOT=results/llm \
  bash scripts/run_rehabfact_six_llm_experiment.sh export-existing
```

六模型选择实验的完整数值证据随包发布：排行榜与逐折评估报告在 `results/RehabFact/six_model_comparison/evaluation/`（`summary.csv`、`mean.csv`、`best_model.json` 及各模型 `fold*.json/csv`），候选集、逐折 adapter 盘点与选择溯源在其上级目录。LoRA adapter 权重与逐病例预测文本不随包发布。`src/llm/README.md` 记录该模块的数据隔离与防泄漏约束。

---

## 目录结构

```text
CMK-AGN-Frozen-Release-20260815/
├── README.md                                # 本文件
├── RELEASE_MANIFEST.csv                      # 全部文件的 SHA-256 清单
├── requirements.txt
├── src/
│   ├── train.py                              # CMK-AGN 单任务训练器
│   ├── predict.py                            # 推理
│   ├── task_config.py                        # 四任务规格（唯一事实来源）
│   ├── clinical_model.py                     # 统一模型封装
│   ├── patient_splits.py                     # 患者级 K 折划分
│   ├── subject_aggregation.py                # 试次袋 → 患者级聚合
│   ├── aggregate_ablation.py                 # 跨折 / 跨消融汇总
│   ├── alignment/                            # 加权贝叶斯 DTW + 三模态对齐
│   ├── bjh_io/                               # EEG / EMG 加载器 + 缓存
│   ├── models/                               # CMK-AGN 主干（adk_mdfan_tri.py）
│   ├── baselines/                            # 深度学习 + 机器学习基线框架
│   ├── evaluation/                           # 容差评价
│   └── llm/                                  # RehabFact 下游模块（含六模型对比）
├── scripts/
│   ├── run_frozen_experiment.sh              # 冻结主线实验入口
│   ├── run_ablation_multiseed_figures.sh     # 消融 + 多种子 + Figure 10/11 入口
│   └── run_rehabfact_six_llm_experiment.sh   # 六基座模型对比 / 导出入口
├── checkpoints/
│   ├── FMA/seed{2024,42,1337}/fold{1,2,3}_best.pt
│   └── BI/seed{2024,42,1337}/fold{1,2,3}_best.pt    # 唯一发布的 18 个权重
├── results/                                  # FMA / BI / MAS / Brunnstrom / RehabFact
├── figures/                                  # 论文正文 Figure 1–11（27 个成品文件）
├── visualization/                            # 按论文图序的绘图代码 + 安全可发的源数据表
├── splits/                                   # 隐私缩减后的患者级划分
└── data/README.md                            # 数据获取与隐私边界说明
```

---

## 发布边界

- **权重**：仅 `checkpoints/FMA/` 与 `checkpoints/BI/` 下的 18 个冻结主线 checkpoint。每个 checkpoint 内嵌任务、折、训练/验证患者成员、模型配置、state dict 与验证指标。MAS、Brunnstrom、RehabFact LoRA、基座大模型与开发期探索性权重一律不随包发布。
- **图**：`figures/` 严格只含正文 Figure 1–11。Figure 4–7、9–11 可由包内数值源表重建；Figure 1–3 为冻结位图（可编辑绘图源未恢复）；Figure 8 为冻结渲染稿（回收的台账不是完整数值表）。
- **文档**：仅保留根 `README.md`、`data/README.md`、`src/llm/README.md`、`src/models/README.md` 四份 Markdown。
- **排除项**：原始 / 处理后信号、临床 manifest 原件、医生自由文本、逐病例提示词与生成报告、直接或可关联标识符、已终止的探索性实验分支、缓存文件。

---

## 数据（用于复现）

原始记录未纳入本包。29 例真实患者的 EEG / sEMG / IMU 信号、临床标签清单与医生参考文本受伦理、隐私与数据使用协议约束，只能在授权环境中使用；获取方式与边界见 `data/README.md`。由于原始数据缺席，本包自身无法重跑"checkpoint → 预测值"的逐字节等价验证。

- 公开 split 仅保留匿名 fold 成员关系（`splits/split_patient_3fold.json`），标签统计、临床字段与服务器数据路径已移除。
- 康复报告语料使用 `splits/llm_real_text13_patient_3fold.json`（13 例带医生参考文本的真实病例，患者级不相交三折）；规则辅助文本只进入训练侧。

## 隐私、再分发与限制

- 包内保留的去标识数值预测与图源数据仍属研究数据；公开再分发前必须确认伦理审批、知情同意、机构政策与数据使用授权。
- 本包未附带软件许可证（LICENSE）。在版权方添加明确许可证之前，请勿将其视为开源发布。
- 引用结果时请注意区分：`results/summary.csv` 是由本包 18 个 checkpoint 支撑的冻结主线结果；`results/reference_comparison.csv` 仅为历史口径对照。

---

<a id="english-version"></a>

# CMK-AGN 

[简体中文](#中文版) ｜ **English**

This project is developed by the Medical Artificial Intelligence Technology
Innovation Center at Zhuhai Fudan Innovation Institute. CMK-AGN is a
tri-modal deep learning model that maps **EEG · sEMG · IMU** signals,
acquired synchronously within a single rehabilitation trial, to four mutually
independent clinical scores. The modalities are first aligned with weighted
Bayesian DTW (EMG↔IMU) plus linearly resampled EEG, then fused by the CMK-AGN
backbone; each task is trained with its own independent output head (no shared
heads, no joint loss). The patient is the only evaluation unit: all
experiments use the fixed patient-level three-fold split, and trial-level
random splitting is strictly prohibited.

This package is the **frozen research release** on the real 29-subject
clinical cohort: final source code, 18 audited FMA/BI checkpoints, frozen
numeric results, the manuscript's Figure 1–11 outputs, runnable
figure-generation code, and three experiment entry scripts. Terminated
exploratory branches are not part of this package.

> **Naming note.** The public model name is **CMK-AGN**. For backward
> compatibility, the Python modules still carry the legacy identifier
> `adk_mdfan` (e.g. `src/models/adk_mdfan_tri.py`, class
> `ADKMDFANTriBackbone`). Both refer to the same CMK-AGN architecture.

---

## Clinical Tasks

Each task is trained as an independent model.

| Task key | Clinical scale | Type | Range / classes | Preset tolerance |
|---|---|---|---|---|
| `FMA_UE` | FMA-UE hand subscale | Regression | 0 – 20 (integer) | \|error\| ≤ 1.5 |
| `BI` | Barthel Index | Regression | 0 – 100 (step 5) | \|error\| ≤ 10 |
| `hand_tone` | Hand MAS (modified Ashworth) | 6-class ordinal | 0, 1, 1+, 2, 3, 4 | — |
| `hand_function` | Brunnstrom hand stage | 6-class ordinal | stages 1 – 6 | — |

The code-level keys `hand_tone` / `hand_function` are retained for manifest
compatibility; clinically they correspond to hand MAS and the Brunnstrom hand
stage, respectively. Ordinal tasks default to the CORN ordinal head
(`src/task_config.py`).

---

## Frozen Mainline Results

Patient-level results backed by the 18 packaged checkpoints
(`results/summary.csv`, n = 29):

| Task | MAE | RMSE | R² | Spearman | Preset-tolerance hits |
|---|---:|---:|---:|---:|---:|
| FMA | 2.517 | 3.821 | 0.750 | 0.831 | 16/29 (55.2%) |
| BI | 12.148 | 15.845 | 0.541 | 0.713 | 15/29 (51.7%) |

- `results/FMA/`, `results/BI/`: frozen mainline results (patient-level
  predictions, per-fold/per-seed metrics, bootstrap CIs, selection parameters).
- `results/MAS/`, `results/Brunnstrom/`: supplementary classification results
  from the manuscript; **weights are not distributed**.
- `results/reference_comparison.csv`: a historical OOF-selected reference
  reported for provenance only; its original patient-level bundle was not
  recovered, so it is not a second frozen result.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data setup

Raw signals are **not** included in this package (see _Data_ below). The
training manifest must point to authorized EEG / sEMG / IMU files and be
provided through an environment variable:

```bash
export MANIFEST=/secure/authorized/samples_manifest_real_archive.csv
```

Both signal-experiment scripts share the fixed patient-level split
`splits/split_patient_3fold.json` (anonymous fold membership only).

### 3. Frozen mainline experiment (FMA / BI)

```bash
MANIFEST=/secure/authorized/samples_manifest_real_archive.csv \
  bash scripts/run_frozen_experiment.sh
```

- 2 tasks × 3 seeds (2024 / 42 / 1337) × 3 folds; checkpoints are selected on
  outer-train validation MAE.
- Outputs go to `runs/frozen_experiment/`; the packaged frozen `results/`,
  `figures/`, and `checkpoints/` are never overwritten.
- `DRY_RUN=1` prints commands without training; `DEVICE`, `EPOCHS`,
  `PATIENCE`, etc. can be overridden via environment variables.

### 4. Complete ablation and multi-seed experiment

```bash
MANIFEST=/secure/authorized/samples_manifest_real_archive.csv \
  bash scripts/run_ablation_multiseed_figures.sh
```

- Full tri-modal model for 4 tasks × 3 seeds, six modality variants (EEG /
  EMG / IMU / EEG+EMG / EEG+IMU / EMG+IMU), three module variants (without
  MDFan / without weighted DTW / without tri-modal alignment), all three
  patient folds — 48 training invocations (144 fold fits) in total.
- Regression checkpoints use validation MAE; ordinal-classification
  checkpoints use validation weighted kappa.
- Aggregates ablation tables and regenerates Figure 10 / 11 source data and
  PNG/PDF/SVG outputs under `runs/ablation_multiseed/`.
- `STAGE=train|aggregate|figures` reruns one stage; `DRY_RUN=1 STAGE=train`
  prints all training commands.

### 5. RehabFact six-base-model comparison

The downstream reporting module RehabFact was compared across six
open-source instruction models in a patient-disjoint three-fold protocol
(selection metric `all.char_bleu4`, mean ± std over three folds; each test
fold contains 5/4/4 real human-reference patients):

| Rank | Model ID | Base model | char-BLEU4 | ROUGE-L F | Outcome |
|---:|---|---|---:|---:|---|
| 1 | `baichuan2_7b` | Baichuan2-7B-Chat | **47.29 ± 5.33** | **61.24 ± 3.72** | **Selected** |
| 2 | `qwen3_8b` | Qwen3-8B | 42.49 ± 5.61 | 55.55 ± 5.56 | Not selected |
| 3 | `gemma3_4b` | Gemma-3-4B-IT | 37.71 ± 5.83 | 52.67 ± 6.29 | Not selected |
| 4 | `glm4_9b_chat` | GLM-4-9B-Chat | 27.54 ± 0.80 | 45.34 ± 1.81 | Not selected |
| 5 | `llama31_8b` | Meta-Llama-3.1-8B-Instruct | 21.27 ± 2.92 | 36.87 ± 3.46 | Not selected |
| 6 | `deepseek_r1_llama8b` | DeepSeek-R1-Distill-Llama-8B | 0.43 ± 0.06 | 1.46 ± 0.19 | Not selected |

Baichuan2-7B-Chat won with a 4.80 char-BLEU4 margin over the runner-up
Qwen3-8B. After selection, RehabFact completed the auxiliary-text ablation
between `human_only` (trained on clinician text only) and `human_plus_aux`
(plus train-side rule-generated auxiliary text)
(`results/RehabFact/aux_ablation_summary.csv`): char-BLEU4 improved from
32.55 to 37.28 and ROUGE-L F from 44.29 to 48.15, with fact-correctness and
safety checks held at 1.0. Note: the model-selection stage and the
auxiliary-text ablation are two independent experiment batches; their
numbers are not directly comparable.

```bash
# Rerun the fixed six-model comparison (writes to runs/, never overwrites frozen results):
CORPUS=/secure/authorized/patient_rehab_29subjects.json \
  bash scripts/run_rehabfact_six_llm_experiment.sh run

# Rebuild summaries from an existing evaluation tree:
RUN_ROOT=runs/rehabfact_six_model_comparison \
  bash scripts/run_rehabfact_six_llm_experiment.sh aggregate

# Export the historical server-side results/llm evidence (per-case text excluded by default):
SOURCE_RESULTS_ROOT=results/llm \
  bash scripts/run_rehabfact_six_llm_experiment.sh export-existing
```

The complete numeric evidence of the selection experiment ships with this
package: the leaderboard and per-fold reports live under
`results/RehabFact/six_model_comparison/evaluation/` (`summary.csv`,
`mean.csv`, `best_model.json`, and per-model `fold*.json/csv`); the candidate
set, per-fold adapter inventory, and selection provenance sit one level up.
LoRA adapter weights and per-case prediction text are not distributed.
`src/llm/README.md` documents the module's data-isolation and leakage
controls.

---

## Directory Layout

```text
CMK-AGN-Frozen-Release-20260815/
├── README.md                                # this file
├── RELEASE_MANIFEST.csv                      # SHA-256 inventory of all files
├── requirements.txt
├── src/
│   ├── train.py                              # CMK-AGN single-task trainer
│   ├── predict.py                            # inference
│   ├── task_config.py                        # task specs (single source of truth)
│   ├── clinical_model.py                     # unified model wrapper
│   ├── patient_splits.py                     # patient-level K-fold splits
│   ├── subject_aggregation.py                # trial bag -> patient aggregation
│   ├── aggregate_ablation.py                 # cross-fold / cross-ablation aggregation
│   ├── alignment/                            # weighted Bayesian DTW + tri-modal alignment
│   ├── bjh_io/                               # EEG / EMG loaders + cache
│   ├── models/                               # CMK-AGN backbone (adk_mdfan_tri.py)
│   ├── baselines/                            # deep-learning + ML baseline framework
│   ├── evaluation/                           # tolerance metrics
│   └── llm/                                  # RehabFact downstream module (incl. six-model comparison)
├── scripts/
│   ├── run_frozen_experiment.sh              # frozen mainline experiment entry
│   ├── run_ablation_multiseed_figures.sh     # ablation + multi-seed + Figure 10/11 entry
│   └── run_rehabfact_six_llm_experiment.sh   # six-base-model comparison / export entry
├── checkpoints/
│   ├── FMA/seed{2024,42,1337}/fold{1,2,3}_best.pt
│   └── BI/seed{2024,42,1337}/fold{1,2,3}_best.pt    # the only 18 released weights
├── results/                                  # FMA / BI / MAS / Brunnstrom / RehabFact
├── figures/                                  # manuscript Figure 1-11 (27 output files)
├── visualization/                            # manuscript-ordered plotting code + distributable source tables
├── splits/                                   # privacy-reduced patient-level splits
└── data/README.md                            # data availability and privacy boundary
```

---

## Release Boundary

- **Weights**: only the 18 frozen mainline checkpoints under
  `checkpoints/FMA/` and `checkpoints/BI/`. Each checkpoint embeds its task,
  fold, train/validation subject membership, model configuration, state
  dictionary, and validation metrics. MAS, Brunnstrom, RehabFact LoRA,
  base-language-model, development, and exploratory weights are not
  distributed.
- **Figures**: `figures/` contains exactly the manuscript's Figure 1–11.
  Figures 4–7 and 9–11 are rebuildable from packaged numeric source tables;
  Figures 1–3 are frozen raster artworks (editable drawing sources were not
  recovered); Figure 8 is a frozen rendered comparison (the recovered ledger
  is not a complete numeric plotting table).
- **Documentation**: only four Markdown files are retained — root
  `README.md`, `data/README.md`, `src/llm/README.md`, `src/models/README.md`.
- **Exclusions**: raw/processed signals, the source clinical manifest,
  clinician-authored free text, per-case prompts and generated reports,
  direct or linkable identifiers, terminated exploratory branches, and
  caches.

---

## Data (for reproduction)

Raw records are not included in this package. The EEG / sEMG / IMU signals,
clinical label manifest, and clinician reference texts of the 29 real
subjects are governed by ethics, privacy, and data-use agreements and may
only be used inside authorized environments; availability and boundaries are
described in `data/README.md`. Because the raw data are absent, this package
alone cannot rerun a byte-level checkpoint-to-prediction equivalence check.

- The public split keeps anonymous fold membership only
  (`splits/split_patient_3fold.json`); label statistics, clinical fields, and
  server data paths were removed.
- The reporting corpus uses `splits/llm_real_text13_patient_3fold.json`
  (13 real cases with clinician-authored references, patient-disjoint three
  folds); rule-generated auxiliary text enters the training side only.

## Privacy, Redistribution, and Limitations

- The retained de-identified numeric predictions and figure-source values
  remain research data; ethics approval, informed consent, institutional
  policy, and data-use authorization must be confirmed before public
  redistribution.
- No software license accompanies this package. Do not treat it as
  open source until the copyright holder adds an explicit license.
- When citing results, distinguish carefully: `results/summary.csv` is the
  frozen mainline result backed by the 18 packaged checkpoints;
  `results/reference_comparison.csv` is a historical-protocol comparison
  only.
