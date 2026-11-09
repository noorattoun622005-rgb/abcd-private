"""Stage 2 — supervised fine-tuning on the SFT pairs, starting from the best continued-pretraining checkpoint.

    python train_sft.py            # full run (about 2 min on a T4)
    python train_sft.py --quick    # pipeline test
    python train_sft.py --push     # also upload the best checkpoint to the Hugging Face Hub (needs HF_TOKEN)

Loss on answer tokens only (prompt labels = -100), 10% of the pairs for validation (split by paraphrase group),
early stopping. Outputs: checkpoints/sft/epoch-k/, checkpoints/sft/best/, results/sft_info.json,
results/sft_test_overlap_audit.csv, figures/sft_loss.png.
"""
import argparse
import json
import time

from transformers import EarlyStoppingCallback, Trainer

from src.config import hf_login, load_config, set_seed
from src.data_prep import check_overlap, load_pairs, load_tests, split_pairs
from src.models import Collator, load_model, load_tokenizer, resolve
from src.template import encode_pair
from src.training import SaveEachEpoch, plot_history, push_to_hub, training_args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="pipeline test")
    ap.add_argument("--push", action="store_true", help="upload the best checkpoint to the Hugging Face Hub")
    args = ap.parse_args()

    cfg = load_config(args.config, quick=args.quick)
    set_seed(cfg["seed"])
    work = cfg["work"]

    pairs, tests = load_pairs(cfg), load_tests(cfg)
    check_overlap(pairs, tests, work / "results/sft_test_overlap_audit.csv")

    start = resolve(cfg, "cpt")                       # best continued-pretraining checkpoint
    tok = load_tokenizer(cfg["base_model"])
    train_pairs, val_pairs = split_pairs(pairs, cfg)
    ml = cfg["sft"]["max_len"]
    sft_train = [encode_pair(p, tok, ml) for p in train_pairs]
    sft_val = [encode_pair(p, tok, ml) for p in val_pairs]
    n_loss = sum(sum(l != -100 for l in e["labels"]) for e in sft_train); n_all = sum(len(e["labels"]) for e in sft_train)
    print(f"SFT: {len(sft_train)} train / {len(sft_val)} validation pairs; loss on {n_loss:,} of {n_all:,} tokens "
          f"({100 * n_loss / n_all:.0f}%) = answers + <eos>")

    ex = encode_pair(next(p for p in pairs if p["category"] == "scenario"), tok, ml)
    print("example (token -> in the loss?):")
    print(" ".join(f"[{tok.decode([t])!r}:{'LOSS' if l != -100 else '-100'}]" for t, l in zip(ex["input_ids"], ex["labels"]))[:1500])

    set_seed(cfg["seed"])
    model = load_model(start)
    out = work / "checkpoints/sft"
    trainer = Trainer(model=model, args=training_args(cfg, "sft", out / "trainer", len(sft_train)),
                      train_dataset=sft_train, eval_dataset=sft_val, data_collator=Collator(tok.pad_token_id),
                      callbacks=[EarlyStoppingCallback(cfg["sft"]["patience"], 0.001), SaveEachEpoch(out, tok)])
    t0 = time.time(); trainer.train()
    model = trainer.model
    model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")

    info = {"start_from": start, "pairs": len(pairs), "train": len(sft_train), "val": len(sft_val),
            "passes": trainer.state.epoch, "best_val_loss": trainer.state.best_metric,
            "best_step": getattr(trainer.state, "best_global_step", None), "minutes": (time.time() - t0) / 60,
            "log_history": trainer.state.log_history}
    json.dump(info, open(work / "results/sft_info.json", "w"), indent=1)
    plot_history(info["log_history"], info["best_step"],
                 f"SFT on {len(sft_train)} pairs (loss on answer tokens), lr {cfg['sft']['lr']}", work / "figures/sft_loss.png")
    print(f"{info['passes']:.2f} passes; best validation loss {info['best_val_loss']:.4f} at step {info['best_step']}")

    if args.push:
        hf_login()
        push_to_hub(model, tok, f"{cfg['hub_user']}/{cfg['models']['sft']['hub']}")


if __name__ == "__main__":
    main()
