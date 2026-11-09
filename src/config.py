"""Settings (config.yaml), random seed, paths and device."""
import os
import random
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")      # one GPU: Kaggle sometimes gives two T4s
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import transformers
import yaml

ROOT = Path(__file__).resolve().parent.parent           # the repository folder


def load_config(path="config.yaml", quick=False):
    """Read config.yaml. quick=True: pipeline test on 5% of the data with a few training steps."""
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    cfg = yaml.safe_load(open(path))
    cfg["quick"] = quick
    cfg["max_steps"] = -1
    if quick:
        cfg["data"]["max_dialogues"] = 500
        cfg["cpt"].update(epochs=1, eval_steps=10)
        cfg["sft"].update(epochs=1, eval_steps=10)
        cfg["max_steps"] = 30
    work = Path(cfg["paths"]["work"])
    cfg["work"] = work if work.is_absolute() else ROOT / work
    for sub in ["checkpoints", "results/grading", "figures", "data/cpt"]:
        (cfg["work"] / sub).mkdir(parents=True, exist_ok=True)
    return cfg


def path(cfg, key):
    """A data file from config.yaml (relative to the repository)."""
    return ROOT / cfg["paths"][key]


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def hf_login():
    """HF_TOKEN from the environment or from Kaggle secrets (never written to the repository)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret("HF_TOKEN")
            os.environ["HF_TOKEN"] = token
        except Exception:
            pass
    return token
