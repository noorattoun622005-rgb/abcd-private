"""Shared training pieces: Trainer settings, a checkpoint per pass, loss-curve plots, Hub upload."""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import TrainerCallback, TrainingArguments


class SaveEachEpoch(TrainerCallback):
    """Saves a checkpoint after every pass over the data (epoch-1/, epoch-2/, ...)."""
    def __init__(self, out_dir, tok):
        self.out_dir, self.tok = Path(out_dir), tok

    def on_epoch_end(self, args, state, control, model=None, **kw):
        d = self.out_dir / f"epoch-{round(state.epoch)}"
        model.save_pretrained(d); self.tok.save_pretrained(d)


def training_args(cfg, stage, out_dir, n_train):
    """Validation every eval_steps, keep the best checkpoint (lowest validation loss), cosine schedule."""
    c, max_steps = cfg[stage], cfg["max_steps"]
    steps_per_epoch = math.ceil(n_train / (c["batch_size"] * c["grad_accum"]))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * c["epochs"]
    return TrainingArguments(
        output_dir=str(out_dir), num_train_epochs=c["epochs"], max_steps=max_steps,
        per_device_train_batch_size=c["batch_size"], per_device_eval_batch_size=c["batch_size"],
        gradient_accumulation_steps=c["grad_accum"], learning_rate=c["lr"], weight_decay=c["weight_decay"],
        warmup_steps=int(c["warmup_ratio"] * total_steps), lr_scheduler_type="cosine", max_grad_norm=1.0,
        eval_strategy="steps", eval_steps=c["eval_steps"], save_strategy="steps", save_steps=c["eval_steps"],
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=max(5, c["eval_steps"] // 2), fp16=torch.cuda.is_available(), report_to="none",
        seed=cfg["seed"], data_seed=cfg["seed"], remove_unused_columns=False, dataloader_num_workers=0)


def plot_history(log_history, best_step, title, path):
    tr = [(x["step"], x["loss"]) for x in log_history if "loss" in x]
    va = [(x["step"], x["eval_loss"]) for x in log_history if "eval_loss" in x]
    fig, ax = plt.subplots(figsize=(6.5, 3.6), dpi=140)
    if tr: ax.plot(*zip(*tr), label="training loss", color="#2a6fdb", lw=1.6)
    if va: ax.plot(*zip(*va), label="validation loss", color="#e07a1f", lw=1.8, marker="o", ms=3)
    if best_step: ax.axvline(best_step, color="#777", ls="--", lw=1, label=f"best checkpoint (step {best_step})")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("cross-entropy loss"); ax.set_title(title, fontsize=10)
    ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def push_to_hub(model, tok, repo_id):
    """Upload a checkpoint to a private Hugging Face Hub repository (needs HF_TOKEN)."""
    model.push_to_hub(repo_id, private=True)
    tok.push_to_hub(repo_id, private=True)
    print(f"pushed to https://huggingface.co/{repo_id}")
