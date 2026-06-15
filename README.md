# NARVAL — Nested Risk-Aware Preference Distillation via Teacher Value Shaping

Self-contained training package for the **NARVAL** objective from
*Nested Risk-Aware Preference Distillation via Teacher Value Shaping*
(EMNLP 2026 submission).

NARVAL combines three signals into a single Bradley-Terry style distillation
objective for a small student model:

1. **Information-geometric token weights** `w_t` — gradient of
   `KL(π_tch_dpo ‖ π_tch_ref)` with respect to token embeddings, normalized
   so completion-token weights sum to `T`.
2. **Nested CVaR risk shaping** `Φ_μ` — per-step CVaR over the vocabulary
   distribution of the log-ratio `log π_tch_dpo − log π_θ` under `π_tch_dpo`.
3. **Risk-corrected BT loss** — `L = −E[log σ(u − δ)]` with
   `u = β · Σ w · log(π_θ / π_tch_dpo)` differences between chosen / rejected,
   and `δ = β · (D_SeqRR(x, y_l) − D_SeqRR(x, y_w))`.

---

## Folder layout

```
narval/
├── README.md                       # this file
├── requirements.txt                # pip requirements
├── alignment/                      # alignment-handbook helpers (configs, data, tokenizer)
│   ├── __init__.py
│   ├── configs.py                  # H4ArgumentParser, ModelArguments, DataArguments
│   ├── data.py                     # apply_chat_template, get_datasets
│   ├── decontaminate.py
│   ├── model_utils.py              # get_tokenizer, get_quantization_config, ...
│   └── release.py
├── scripts/
│   └── run_narval.py               # NARVALTrainer + main()
├── recipes/
│   ├── accelerate_config/
│   │   └── deepspeed_zero3.yaml    # DeepSpeed ZeRO-3 config
│   └── configs/
│       └── narval.yaml             # NARVAL training hyperparams
└── run/
    └── run_narval.sh               # accelerate launcher
```

The `alignment/` package is copied locally so the script is independent of the
outer repo. `PYTHONPATH` is set automatically by `run_narval.sh` so that
`from alignment import ...` resolves to this folder.

---

## Prerequisites

NARVAL is the **final** stage of a three-stage pipeline. Before running NARVAL
you must have three checkpoints on disk:

| Role | Variable in `narval.yaml` | How to obtain |
|---|---|---|
| Student (SFT-initialized) | `model_name_or_path` | SFT the student backbone on the SFT dataset (e.g. Deita-10k). |
| DPO-trained teacher       | `teacher_dpo_path`    | SFT a larger backbone, then run DPO on a preference dataset (e.g. DPO-Mix-7K). |
| SFT reference teacher     | `teacher_ref_path`    | The same SFT teacher *before* DPO. Used only to derive token weights. |

The paper uses Llama-3.1-8B (DPO) → Llama-3.2-1B (student), and Qwen3-8B → Qwen3-1.7B / Qwen3-0.6B variants.

The original repo (one folder up) contains scripts for the SFT and DPO stages
(`scripts/run_sft.py`, `scripts/run_distill_dpo.py`). You can use those, or any
other SFT/DPO pipeline; NARVAL only consumes the resulting checkpoints.

---

## Installation

```bash
cd narval/
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The implementation targets the same versions used by the rest of the repo:

- `torch==2.5.1`
- `trl==0.12.0`
- `peft==0.13.0`
- `transformers` ≥ 4.46

Install Flash-Attention separately if you want to use it:

```bash
pip install flash-attn --no-build-isolation
```

---

## Configuration

Edit `recipes/configs/narval.yaml` before launching:

```yaml
model_name_or_path: model/student_sft        # student SFT checkpoint
teacher_dpo_path:   model/teacher_dpo        # DPO-trained teacher
teacher_ref_path:   model/teacher_sft        # SFT-only reference teacher

dataset_mixer:
  /path/to/preference_dataset: 1.0           # has (prompt, chosen, rejected)
dataset_splits: [train]

beta: 0.1                # DPO temperature
narval_mu: 0.97          # CVaR risk level (paper default)
narval_use_token_weight: true
narval_use_delta: true
narval_log_ratio_clip: 30.0

learning_rate: 5.0e-7
num_train_epochs: 5
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
max_length: 2048
max_prompt_length: 1800

output_dir: output/narval
```

Dataset schema: each row must contain `chosen` and `rejected` fields in
OpenAI chat format (same as `alignment-handbook` / DPO / TRL).

---

## Run training

```bash
bash run/run_narval.sh
```

The launcher uses DeepSpeed ZeRO-3 over 4 GPUs by default. Adjust by exporting
`CUDA_VISIBLE_DEVICES` or editing `recipes/accelerate_config/deepspeed_zero3.yaml`.

Logged metrics (to console and W&B if enabled):

| Metric | Meaning |
|---|---|
| `loss/narval` | NARVAL loss `−log σ(u − δ)` |
| `narval/u` | mean implicit reward gap (chosen − rejected) |
| `narval/delta` | mean risk-aware correction |
| `narval/reward_{chosen,rejected}` | per-branch implicit rewards |
| `narval/risk_{chosen,rejected}` | per-branch sequential risk `D_SeqRR` |
| `narval/accuracy` | fraction of pairs with `u > δ` |

---

## Ablations

The objective has two switches you can toggle for ablations matching Table 4
of the paper:

| Variant | `narval_use_token_weight` | `narval_use_delta` |
|---|---|---|
| **NARVAL (full)** | true  | true  |
| NARVAL w/o δ      | true  | false |
| NARVAL w/o weight | false | true  |
| NARVAL w/o both   | false | false |

`narval_mu` ∈ {0.95, 0.97, 0.98, 0.99} sweep matches Table 5 of the paper.

---

## Implementation notes

- Token importance is computed with a single backward pass through the DPO
  teacher's input embeddings; the resulting weights are detached before being
  used in the policy loss.
- Both teachers run frozen and in inference mode; with DeepSpeed ZeRO-3 they
  are sharded via `deepspeed.initialize(stage=3)` and reduced to stage 0 if
  the launcher uses a smaller ZeRO stage.
- The CVaR is computed exactly for a discrete distribution: sort by
  log-ratio descending, take the tail of mass `(1 − μ)` by walking down the
  cumulative probability, and return the probability-weighted mean of that
  tail divided by `(1 − μ)`.
- Log-ratios are clipped to `±narval_log_ratio_clip` (default 30) to prevent
  CVaR explosion when the student briefly assigns near-zero probability to a
  token the teacher likes.

---
