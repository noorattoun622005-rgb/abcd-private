"""Runs the three models (base, continued-pretrained, SFT) on the 30 test questions through the same function
and scores them with one method. Writes results/results_<model>.csv, results/summary.json and SCOREBOARD.md.

    python score.py                         # all three models, all 30 questions, with held-out perplexity
    python score.py --no-ppl                # skip perplexity (faster)
    python score.py --ask SP-07 SP-25       # just show the three models' answers to some test questions
    python score.py --quick                 # first 5 questions only

Models come from config.yaml: local checkpoints/ if present, otherwise the Hugging Face Hub repos.
"""
import argparse
import json

import pandas as pd

from src.config import ROOT, hf_login, load_config, set_seed
from src.data_prep import heldout_texts, load_tests
from src.models import generate_answer, load_model, load_tokenizer, pack, perplexity, resolve
from src.scoring import is_refusal, load_human_grades, predicted_action, same_text, score_case, summarize
from src.template import parse_history

NAMES = {"base": "Small model, base", "cpt": "+ continued pretraining", "sft": "+ SFT"}
NOTES = {"base": "SmolLM2-135M, no training",
         "cpt": "3 passes, lr 1e-4, 512-token blocks, policy docs x5",
         "sft": "591 pairs (533/58), 2.5 passes, lr 5e-5, loss on answers only"}


def answer_all(name, model, tok, tests, cfg, grades):
    rows = []
    for _, c in tests.iterrows():
        g = generate_answer(model, tok, parse_history(c.conversation_history), c.customer_input, cfg["gen"])
        s, note = score_case(c, g["answer"])
        source = "exact" if c.answer_type == "exact" else "provisional"
        if c.answer_type != "exact" and (name, c.test_id) in grades:
            graded_answer, grade = grades[(name, c.test_id)]
            if same_text(g["answer"], graded_answer):
                s, source, note = grade, "human", "final grade of the two graders"
            else:
                note += " | NEW ANSWER: grade it by hand"
        rows.append({"test_id": c.test_id, "question": c.customer_input, "conversation_history": c.conversation_history,
                     "expected_action": c.expected_action, "expected_answer": c.expected_response,
                     "answer_type": c.answer_type, "model_answer": g["answer"], "predicted_action": predicted_action(g["answer"]),
                     "refused": is_refusal(g["answer"]), "score": s, "score_source": source, "note": note,
                     "latency_s": round(g["latency_s"], 3), "new_tokens": g["new_tokens"], "stopped": g["stopped"]})
        print(f"[{name}] {c.test_id} score={s:<4} ({source}) | {g['answer'][:90]!r}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="+", default=["base", "cpt", "sft"])
    ap.add_argument("--ask", nargs="+", metavar="TEST_ID", help="only show the answers to these test questions")
    ap.add_argument("--no-ppl", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, quick=args.quick)
    set_seed(cfg["seed"])
    hf_login()                                     # private Hub repos need HF_TOKEN
    work = cfg["work"]
    tests = load_tests(cfg)
    if args.ask:
        tests = tests[tests.test_id.isin(args.ask)]
    elif args.quick:
        tests = tests.head(5)
    grades = load_human_grades(ROOT / cfg["paths"]["human_grades"])
    tok = load_tokenizer(cfg["base_model"])
    held_blocks = None
    if not (args.no_ppl or args.ask):
        held_blocks = pack(heldout_texts(cfg), tok, cfg["cpt"]["block_size"], keep_remainder=True)
        if args.quick:
            held_blocks = held_blocks[:40]

    results, ppl = {}, {}
    for name in args.models:
        src = resolve(cfg, name)
        print(f"\n=== {NAMES[name]}  ({src})")
        model = load_model(src)
        results[name] = answer_all(name, model, tok, tests, cfg, grades)
        if held_blocks is not None:
            ppl[name] = perplexity(model, held_blocks, tok.pad_token_id)
            print(f"held-out perplexity: {ppl[name]:.3f}")
        del model

    if args.ask:                                   # side-by-side view, nothing saved
        for t in tests.test_id:
            print(f"\n{t}: {tests.set_index('test_id').loc[t, 'customer_input']}")
            for name, df in results.items():
                r = df.set_index("test_id").loc[t]
                print(f"  {NAMES[name]:<25} score {r.score:<4} | {r.model_answer[:200]}")
        return

    for name, df in results.items():
        df.to_csv(work / f"results/results_{name}.csv", index=False)
    rows = []
    for name, df in results.items():
        s = summarize(df)
        rows.append({"Version": NAMES[name], "Held-out perplexity": round(ppl[name], 2) if name in ppl else "-",
                     "Accuracy": f"{100 * s['accuracy']:.1f}%", "Refusal accuracy": f"{100 * s['refusal_accuracy']:.1f}%",
                     "Avg. latency (s)": round(s["latency_s"], 2),
                     "Notes": NOTES[name] + ("" if s["to_grade"] == 0 else f"; {s['to_grade']} new answers to grade by hand")})
    board = pd.DataFrame(rows)
    md = ["# Scoreboard", "",
          f"Same test set for every version: `tests/test_questions.csv` ({len(tests)} cases, "
          f"{int((tests.expected_action == 'ABSTAIN').sum())} should be refused). Greedy decoding, max "
          f"{cfg['gen']['max_new_tokens']} new tokens, identical prompt for all models (`score.py`). Accuracy = mean score "
          "(exact match for short answers, final grade of two human graders for longer ones). Refusal accuracy = share of "
          "should-refuse cases declined. Perplexity on the 10% held-out documents. Latency = mean generation time per "
          "answer on a T4 GPU.", "",
          "| " + " | ".join(board.columns) + " |", "|" + "---|" * len(board.columns)]
    md += ["| " + " | ".join(str(v) for v in r) + " |" for r in board.values]
    if not args.quick and set(results) == {"base", "cpt", "sft"}:
        (work / "SCOREBOARD.md").write_text("\n".join(md) + "\n")
    json.dump({k: {**summarize(results[k]), **({"perplexity": ppl[k]} if k in ppl else {})} for k in results},
              open(work / "results/summary.json", "w"), indent=2)
    print("\n" + board.to_string(index=False))


if __name__ == "__main__":
    main()
