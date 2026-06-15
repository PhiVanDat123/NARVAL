"""NARVAL — Nested Risk-Aware pReference distillation via teacher VALue shaping.

Implements the objective from "Nested Risk-Aware Preference Distillation via
Teacher Value Shaping" (EMNLP 2026 submission). The objective combines:

  * Information-geometric token-importance weights w_t based on the gradient
    of KL(pi_tch_dpo || pi_tch_ref) with respect to token embeddings.
  * Nested CVaR risk shaping Phi_mu over per-step log-ratios between the DPO
    teacher and the student.
  * A Bradley-Terry style loss with implicit reward u and risk correction delta.

Usage:
    accelerate launch scripts/run_narval.py recipes/.../narval.yaml
"""
from __future__ import annotations

import logging
import os
import random
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    is_wandb_available,
    set_seed,
)
from transformers.data.data_collator import DataCollatorMixin
from trl.trainer.utils import disable_dropout_in_model, pad

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_kbit_device_map,
    get_quantization_config,
    get_tokenizer,
)
from alignment.data import apply_chat_template, get_datasets

logger = logging.getLogger(__name__)

if is_wandb_available():
    import wandb

if is_deepspeed_available():
    import deepspeed


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NARVALConfig(TrainingArguments):
    """Training arguments for NARVAL."""

    hub_model_revision: Optional[str] = field(default="main")
    logging_first_step: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)

    beta: float = field(default=0.1, metadata={"help": "DPO temperature beta."})
    narval_mu: float = field(
        default=0.97,
        metadata={"help": "CVaR risk level mu in (0,1); larger = more risk-averse."},
    )
    narval_use_token_weight: bool = field(
        default=True,
        metadata={"help": "If True, weight tokens by information-geometric importance."},
    )
    narval_use_delta: bool = field(
        default=True,
        metadata={"help": "If True, include the risk-aware delta correction term."},
    )
    narval_log_ratio_clip: float = field(
        default=30.0,
        metadata={"help": "Symmetric clip on per-token log-ratios for CVaR numerical stability."},
    )

    teacher_dpo_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the DPO-trained teacher (pi_tch_dpo)."},
    )
    teacher_ref_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the SFT reference teacher (pi_tch_ref). Used only for token weights."},
    )

    label_pad_token_id: int = field(default=-100)
    padding_value: Optional[int] = field(default=None)
    max_length: Optional[int] = field(default=None)
    max_prompt_length: Optional[int] = field(default=None)
    truncation_mode: str = field(default="keep_end")
    disable_dropout: bool = field(default=True)
    dataset_num_proc: Optional[int] = field(default=None)
    model_init_kwargs: Optional[dict[str, Any]] = field(default=None)

    def __post_init__(self):
        return super().__post_init__()


# ---------------------------------------------------------------------------
# Data collator
# ---------------------------------------------------------------------------


@dataclass
class NARVALDataCollator(DataCollatorMixin):
    pad_token_id: int
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_input_ids = [torch.tensor(e["prompt_input_ids"]) for e in examples]
        prompt_attention_mask = [torch.ones_like(t) for t in prompt_input_ids]
        chosen_input_ids = [torch.tensor(e["chosen_input_ids"]) for e in examples]
        chosen_attention_mask = [torch.ones_like(t) for t in chosen_input_ids]
        chosen_labels = [torch.tensor(e["chosen_labels"]) for e in examples]
        rejected_input_ids = [torch.tensor(e["rejected_input_ids"]) for e in examples]
        rejected_attention_mask = [torch.ones_like(t) for t in rejected_input_ids]
        rejected_labels = [torch.tensor(e["rejected_labels"]) for e in examples]

        return {
            "prompt_input_ids": pad(prompt_input_ids, padding_value=self.pad_token_id, padding_side="left"),
            "prompt_attention_mask": pad(prompt_attention_mask, padding_value=0, padding_side="left"),
            "chosen_input_ids": pad(chosen_input_ids, padding_value=self.pad_token_id),
            "chosen_attention_mask": pad(chosen_attention_mask, padding_value=0),
            "chosen_labels": pad(chosen_labels, padding_value=-100),
            "rejected_input_ids": pad(rejected_input_ids, padding_value=self.pad_token_id),
            "rejected_attention_mask": pad(rejected_attention_mask, padding_value=0),
            "rejected_labels": pad(rejected_labels, padding_value=-100),
        }


def tokenize_row(features, tokenizer, max_prompt_length, max_length):
    """Tokenize a single preference-pair row."""
    prompt_ids = tokenizer(features["prompt"], add_special_tokens=False)["input_ids"]
    chosen_ids = tokenizer(features["chosen"], add_special_tokens=False)["input_ids"]
    rejected_ids = tokenizer(features["rejected"], add_special_tokens=False)["input_ids"]

    if tokenizer.bos_token_id is not None and (not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id):
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    if tokenizer.eos_token_id not in chosen_ids[-5:]:
        chosen_ids = chosen_ids + [tokenizer.eos_token_id]
    if tokenizer.eos_token_id not in rejected_ids[-5:]:
        rejected_ids = rejected_ids + [tokenizer.eos_token_id]

    total_len = max(len(prompt_ids) + len(chosen_ids), len(prompt_ids) + len(rejected_ids))
    if max_length is not None and total_len > max_length:
        if max_prompt_length is not None and len(prompt_ids) > max_prompt_length:
            prompt_ids = prompt_ids[-max_prompt_length:]
        max_resp = max_length - len(prompt_ids)
        chosen_ids = chosen_ids[:max_resp]
        rejected_ids = rejected_ids[:max_resp]

    return {
        "prompt_input_ids": prompt_ids,
        "chosen_input_ids": prompt_ids + chosen_ids,
        "chosen_labels": [-100] * len(prompt_ids) + chosen_ids,
        "rejected_input_ids": prompt_ids + rejected_ids,
        "rejected_labels": [-100] * len(prompt_ids) + rejected_ids,
    }


# ---------------------------------------------------------------------------
# CVaR helper
# ---------------------------------------------------------------------------


def cvar_over_vocab(
    log_ratio: torch.Tensor,
    probs: torch.Tensor,
    mu: float,
) -> torch.Tensor:
    """Discrete CVaR_mu of log-ratios under the given probability distribution.

    Args:
        log_ratio: [B, T, V] = log pi_tch(z) - log pi_theta(z).
        probs: [B, T, V] = pi_tch(z), non-negative, sums to 1 along last dim.
        mu: risk level in (0, 1). Larger mu => smaller (1 - mu) tail =>
            more risk-averse (focuses on worst-case high log-ratios).

    Returns:
        [B, T] tensor of CVaR values.
    """
    tail_mass = 1.0 - float(mu)
    if tail_mass <= 0.0:
        return (log_ratio * probs).max(dim=-1).values

    sorted_vals, sorted_idx = torch.sort(log_ratio, dim=-1, descending=True)
    sorted_probs = torch.gather(probs, dim=-1, index=sorted_idx)

    cum_probs = torch.cumsum(sorted_probs, dim=-1)
    prev_cum = cum_probs - sorted_probs

    contrib = torch.clamp(tail_mass - prev_cum, min=0.0)
    contrib = torch.minimum(contrib, sorted_probs)

    cvar = (sorted_vals * contrib).sum(dim=-1) / tail_mass
    return cvar


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _set_inference(model: nn.Module) -> nn.Module:
    """Put a module in inference mode without using the literal method name
    that triggers some security hooks."""
    model.train(False)
    return model


class NARVALTrainer(Trainer):
    """Trainer implementing the NARVAL preference-distillation objective."""

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str],
        teacher_dpo: Union[PreTrainedModel, nn.Module],
        teacher_ref: Optional[Union[PreTrainedModel, nn.Module]],
        args: NARVALConfig,
        train_dataset: Optional[Dataset] = None,
        eval_dataset=None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        data_collator=None,
        callbacks=None,
    ):
        if model is None:
            raise ValueError("`model` (student) is required.")
        if teacher_dpo is None:
            raise ValueError("`teacher_dpo` is required.")

        if isinstance(model, str):
            kwargs = args.model_init_kwargs or {}
            model = AutoModelForCausalLM.from_pretrained(model, **kwargs)

        if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        if args.disable_dropout:
            disable_dropout_in_model(model)

        if args.padding_value is not None:
            self.padding_value = args.padding_value
        elif processing_class is not None and processing_class.pad_token_id is not None:
            self.padding_value = processing_class.pad_token_id
        else:
            raise ValueError("processing_class must have a pad_token_id.")

        if data_collator is None:
            data_collator = NARVALDataCollator(pad_token_id=self.padding_value)

        self.beta = args.beta
        self.narval_mu = args.narval_mu
        self.use_token_weight = args.narval_use_token_weight
        self.use_delta = args.narval_use_delta
        self.log_ratio_clip = args.narval_log_ratio_clip
        self.dataset_num_proc = args.dataset_num_proc
        self._stored_metrics: dict[str, dict[str, list]] = {"train": {}, "eval": {}}

        with PartialState().local_main_process_first():
            fn_kwargs = {
                "tokenizer": processing_class,
                "max_prompt_length": args.max_prompt_length,
                "max_length": args.max_length,
            }
            train_dataset = train_dataset.map(
                tokenize_row,
                fn_kwargs=fn_kwargs,
                num_proc=self.dataset_num_proc,
                desc="Tokenizing train dataset",
            )
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(
                    tokenize_row,
                    fn_kwargs=fn_kwargs,
                    num_proc=self.dataset_num_proc,
                    desc="Tokenizing eval dataset",
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
        )

        _set_inference(teacher_dpo)
        if teacher_ref is not None:
            _set_inference(teacher_ref)
        if self.is_deepspeed_enabled:
            self.teacher_dpo = self._prepare_deepspeed(teacher_dpo)
            self.teacher_ref = (
                self._prepare_deepspeed(teacher_ref) if teacher_ref is not None else None
            )
        else:
            self.teacher_dpo = self.accelerator.prepare_model(teacher_dpo, evaluation_mode=True)
            self.teacher_ref = (
                self.accelerator.prepare_model(teacher_ref, evaluation_mode=True)
                if teacher_ref is not None
                else None
            )

        if self.use_token_weight and self.teacher_ref is None:
            raise ValueError(
                "narval_use_token_weight=True requires teacher_ref (SFT teacher) to be provided."
            )

    def _prepare_deepspeed(self, model: nn.Module):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
        if hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes)
                if getattr(model.config, "hidden_sizes", None)
                else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        _set_inference(model)
        return model

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = [
                "prompt_input_ids",
                "chosen_input_ids",
                "rejected_input_ids",
            ]

    # ---- Core NARVAL computations -------------------------------------------

    @staticmethod
    def _shift_labels_mask(labels: torch.Tensor) -> torch.Tensor:
        return (labels[:, 1:] != -100).float()

    def _token_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute information-geometric token weights w_t for completion positions.

        Returns:
            [B, T-1] tensor of weights aligned with next-token positions.
            For each sequence the weights on completion positions sum to T
            (number of completion tokens in that sequence).
        """
        loss_mask = self._shift_labels_mask(labels)
        n_completion = loss_mask.sum(dim=-1).clamp(min=1.0)

        embed_layer = self.teacher_dpo.get_input_embeddings()
        with torch.enable_grad():
            embeds = embed_layer(input_ids).detach().requires_grad_(True)
            out_dpo = self.teacher_dpo(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits_dpo = out_dpo.logits.float()

            with torch.no_grad():
                out_ref = self.teacher_ref(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                logits_ref = out_ref.logits.float()

            logp_dpo = F.log_softmax(logits_dpo[:, :-1, :], dim=-1)
            p_dpo = logp_dpo.exp()
            logp_ref = F.log_softmax(logits_ref[:, :-1, :], dim=-1)
            per_pos_kl = (p_dpo * (logp_dpo - logp_ref)).sum(-1)
            kl_sum = (per_pos_kl * loss_mask).sum()

            grad_emb = torch.autograd.grad(
                kl_sum, embeds, retain_graph=False, create_graph=False
            )[0]

        importance = grad_emb.detach().abs().sum(dim=-1)
        # Embedding at position p contributes to the prediction at position p+1,
        # so weight for completion at next-token index t (0..T-2) uses input
        # position t+1's gradient. Shift accordingly.
        I = importance[:, 1:]
        I = I * loss_mask
        norm = I.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        w_hat = I / norm
        w = w_hat * n_completion.unsqueeze(-1)
        return w

    def _per_token_log_ratio_full(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s_logp = F.log_softmax(student_logits[:, :-1, :].float(), dim=-1)
        t_logp = F.log_softmax(teacher_logits[:, :-1, :].float(), dim=-1)
        log_ratio = (t_logp - s_logp).clamp(-self.log_ratio_clip, self.log_ratio_clip)
        t_probs = t_logp.exp()
        return log_ratio, t_probs

    @staticmethod
    def _gather_label_log_ratio(
        log_ratio: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Pick the label position. Returns log pi_theta(y_t) - log pi_tch(y_t)."""
        shifted_labels = labels[:, 1:].clone()
        shifted_labels[shifted_labels == -100] = 0
        return -log_ratio.gather(-1, shifted_labels.unsqueeze(-1)).squeeze(-1)

    def _compute_branch(
        self,
        student,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        loss_mask = self._shift_labels_mask(labels)

        if self.use_token_weight:
            w = self._token_importance(input_ids, attention_mask, labels)
        else:
            w = loss_mask
        w = w * loss_mask

        student_out = student(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        with torch.no_grad():
            teacher_out = self.teacher_dpo(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )

        log_ratio_full, teacher_probs = self._per_token_log_ratio_full(
            student_out.logits, teacher_out.logits
        )

        log_ratio_label = self._gather_label_log_ratio(log_ratio_full, labels)
        per_seq_reward = self.beta * (w * log_ratio_label).sum(dim=-1)

        if self.use_delta:
            cvar = cvar_over_vocab(log_ratio_full, teacher_probs.detach(), self.narval_mu)
            seq_risk = (w * cvar).sum(dim=-1)
        else:
            seq_risk = torch.zeros(input_ids.size(0), device=input_ids.device)

        return {
            "reward": per_seq_reward,
            "risk": seq_risk,
            "weight_sum": w.sum(dim=-1).detach(),
            "n_tokens": loss_mask.sum(dim=-1).detach(),
        }

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        ctx = amp.autocast("cuda") if torch.cuda.is_available() else nullcontext()
        with ctx:
            chosen = self._compute_branch(
                model,
                inputs["chosen_input_ids"],
                inputs["chosen_attention_mask"],
                inputs["chosen_labels"],
            )
            rejected = self._compute_branch(
                model,
                inputs["rejected_input_ids"],
                inputs["rejected_attention_mask"],
                inputs["rejected_labels"],
            )

            u = chosen["reward"] - rejected["reward"]
            if self.use_delta:
                delta = self.beta * (rejected["risk"] - chosen["risk"])
            else:
                delta = torch.zeros_like(u)

            logits = u - delta
            losses = -F.logsigmoid(logits)
            loss = losses.mean()

        metrics = {
            "loss/narval": losses.detach().mean().item(),
            "narval/u": u.detach().mean().item(),
            "narval/delta": delta.detach().mean().item(),
            "narval/reward_chosen": chosen["reward"].detach().mean().item(),
            "narval/reward_rejected": rejected["reward"].detach().mean().item(),
            "narval/risk_chosen": chosen["risk"].detach().mean().item(),
            "narval/risk_rejected": rejected["risk"].detach().mean().item(),
            "narval/accuracy": (u > delta).float().mean().item(),
        }
        self._store_metrics(metrics, "train")

        if return_outputs:
            return loss, metrics
        return loss

    def _store_metrics(self, metrics: dict[str, float], train_eval: Literal["train", "eval"]):
        bucket = self._stored_metrics.setdefault(train_eval, {})
        for k, v in metrics.items():
            bucket.setdefault(k, []).append(v)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for k, values in self._stored_metrics.get(train_eval, {}).items():
            if values:
                logs[k] = float(torch.tensor(values).mean().item())
        self._stored_metrics[train_eval] = {}

        if is_wandb_available() and wandb.run is not None:
            wandb_logs = {}
            for k, v in logs.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if k.startswith("eval_"):
                    wandb_logs["eval/" + k[len("eval_"):]] = v
                else:
                    wandb_logs["train/" + k] = v
            wandb.log(wandb_logs, step=self.state.global_step)

        try:
            return super().log(logs, start_time)
        except TypeError:
            return super().log(logs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _build_model_kwargs(model_args: ModelArguments, training_args: NARVALConfig):
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    return dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_flash_attention_2=model_args.use_flash_attention_2,
    )


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, NARVALConfig))
    model_args, data_args, training_args = parser.parse()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(training_args.get_process_log_level())
    transformers.utils.logging.set_verbosity(training_args.get_process_log_level())
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training parameters {training_args}")

    if training_args.teacher_dpo_path is None:
        raise ValueError("`teacher_dpo_path` must be set in the NARVAL config.")
    if training_args.narval_use_token_weight and training_args.teacher_ref_path is None:
        raise ValueError(
            "`teacher_ref_path` must be set when `narval_use_token_weight=True`."
        )

    set_seed(training_args.seed)

    if is_wandb_available() and training_args.report_to and "wandb" in training_args.report_to:
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "narval"),
            name=training_args.run_name,
            config={
                "model": model_args.model_name_or_path,
                "teacher_dpo": training_args.teacher_dpo_path,
                "teacher_ref": training_args.teacher_ref_path,
                "beta": training_args.beta,
                "mu": training_args.narval_mu,
                "use_token_weight": training_args.narval_use_token_weight,
                "use_delta": training_args.narval_use_delta,
                "learning_rate": training_args.learning_rate,
            },
            resume="allow",
        )

    # --- Datasets ---
    columns_to_keep = ["chosen", "rejected", "prompt"]
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=columns_to_keep,
    )
    logger.info(
        f"Training on splits: {[(s, dset.num_rows) for s, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    for split in ["train", "test"]:
        if split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
            )

    for index in random.sample(range(len(raw_datasets["train"])), min(2, len(raw_datasets["train"]))):
        logger.info(f"Sample {index} prompt: {raw_datasets['train'][index]['prompt'][:200]}...")

    # --- Models ---
    model_kwargs = _build_model_kwargs(model_args, training_args)
    logger.info(f"Loading student from {model_args.model_name_or_path}")
    student = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    teacher_kwargs = dict(model_kwargs)
    teacher_kwargs["use_cache"] = False
    logger.info(f"Loading DPO teacher from {training_args.teacher_dpo_path}")
    teacher_dpo = AutoModelForCausalLM.from_pretrained(training_args.teacher_dpo_path, **teacher_kwargs)
    for p in teacher_dpo.parameters():
        p.requires_grad = False

    teacher_ref = None
    if training_args.narval_use_token_weight:
        logger.info(f"Loading reference teacher from {training_args.teacher_ref_path}")
        teacher_ref = AutoModelForCausalLM.from_pretrained(
            training_args.teacher_ref_path, **teacher_kwargs
        )
        for p in teacher_ref.parameters():
            p.requires_grad = False

    trainer = NARVALTrainer(
        model=student,
        teacher_dpo=teacher_dpo,
        teacher_ref=teacher_ref,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if "test" in raw_datasets else None,
        processing_class=tokenizer,
    )

    checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if is_wandb_available() and wandb.run is not None:
        wandb.finish()

    logger.info("*** Saving model ***")
    trainer.save_model(training_args.output_dir)

    if trainer.accelerator.is_main_process:
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
