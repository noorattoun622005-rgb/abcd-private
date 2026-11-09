"""Stage 1 — continued pretraining (next-token prediction on the cleaned ABCD corpus).

    python train_cpt.py            # full run (about 30 min on a T4)
    python train_cpt.py --quick    # pipeline test: 5% of the data, 30 steps
    python train_cpt.py --push     # also upload the best checkpoint to the Hugging Face Hub (needs HF_TOKEN)

Outputs: checkpoints/cpt/epoch-k/ (one per pass), checkpoints/cpt/best/, results/cpt_info.json,
results/cpt_samples.csv, figures/cpt_loss.png, held-out perplexity before -> after.
"""
import argparse
import json
import time

import pandas as pd
from transformers import EarlyStoppingCallback, Trainer

from src.config import hf_login, load_config, set_seed
from src.data_prep import build_corpus
from src.models import Collator, complete, load_model, load_tokenizer, pack, perplexity
from src.training import SaveEachEpoch, plot_history, push_to_hub, training_args

SAMPLE_PROMPTS = ["Gold members get", "Bronze members can return an item if",
                  "To initiate a refund, the agent asks the customer for",
                  "Customer support conversation.\nCustomer: hi, my promo code is not working.\nAgent:",
                  "The refund status can be"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true", help="pipeline test on 5%% of the data")
    ap.add_argument("--push", action="store_true", help="upload the best checkpoint to the Hugging Face Hub")
    args = ap.parse_args()

    cfg = load_config(args.config, quick=args.quick)
    set_seed(cfg["seed"])
    work, bs = cfg["work"], cfg["cpt"]["block_size"]

    tok = load_tokenizer(cfg["base_model"])
    train_texts, val_texts, held_texts = build_corpus(cfg, tok)
    train_blocks = pack(train_texts, tok, bs, keep_remainder=False)
    val_blocks = pack(val_texts, tok, bs, keep_remainder=True)
    held_blocks = pack(held_texts, tok, bs, keep_remainder=True)
    if cfg["quick"]:
        val_blocks, held_blocks = val_blocks[:40], held_blocks[:40]
    print(f"CPT data: {len(train_blocks)} training blocks x {bs} tokens = {len(train_blocks) * bs:,} tokens per pass; "
          f"{len(val_blocks)} validation blocks; held-out {sum(len(b['input_ids']) for b in held_blocks):,} tokens")

    model = load_model(cfg["base_model"])
    ppl_before = perplexity(model, held_blocks, tok.pad_token_id)
    print(f"held-out perplexity before: {ppl_before:.3f}")
    samples = pd.DataFrame({"prompt": SAMPLE_PROMPTS, "base": complete(model, tok, SAMPLE_PROMPTS)})

    set_seed(cfg["seed"])
    out = work / "checkpoints/cpt"
    trainer = Trainer(model=model, args=training_args(cfg, "cpt", out / "trainer", len(train_blocks)),
                      train_dataset=train_blocks, eval_dataset=val_blocks, data_collator=Collator(tok.pad_token_id),
                      callbacks=[EarlyStoppingCallback(cfg["cpt"]["patience"], 0.001), SaveEachEpoch(out, tok)])
    t0 = time.time(); trainer.train()
    model = trainer.model                              # best checkpoint (load_best_model_at_end=True)
    model.save_pretrained(out / "best"); tok.save_pretrained(out / "best")

    ppl_after = perplexity(model, held_blocks, tok.pad_token_id)
    info = {"passes": trainer.state.epoch, "best_val_loss": trainer.state.best_metric,
            "best_step": getattr(trainer.state, "best_global_step", None), "minutes": (time.time() - t0) / 60,
            "perplexity_before": ppl_before, "perplexity_after": ppl_after, "log_history": trainer.state.log_history}
    json.dump(info, open(work / "results/cpt_info.json", "w"), indent=1)
    plot_history(info["log_history"], info["best_step"],
                 f"Continued pretraining — {cfg['base_model'].split('/')[-1]}, lr {cfg['cpt']['lr']}", work / "figures/cpt_loss.png")
    samples["after CPT"] = complete(model, tok, SAMPLE_PROMPTS)
    samples.to_csv(work / "results/cpt_samples.csv", index=False)

    print(f"{info['passes']:.2f} passes; best validation loss {info['best_val_loss']:.4f} at step {info['best_step']}; "
          f"{info['minutes']:.1f} min")
    print(f"held-out perplexity: {ppl_before:.3f} -> {ppl_after:.3f} ({100 * (1 - ppl_after / ppl_before):.1f}% lower)")
    print(samples.to_string(index=False))

    if args.push:
        hf_login()
        push_to_hub(model, tok, f"{cfg['hub_user']}/{cfg['models']['cpt']['hub']}")


if __name__ == "__main__":
    main()
