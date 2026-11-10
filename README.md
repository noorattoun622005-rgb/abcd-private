# Phase 2 — Train a Small Model and Evaluate It (ABCD / AcmeCorp support)

**Team:** Noor Attoun · Shahd Al-Omari · Faten Jadou · Yaqeen Ziad
**Course:** Building LLM Systems — Team Project, Phase 2 · **Tag:** `phase-2`
**Report:** [`report/Phase2_Report.pdf`](report/Phase2_Report.pdf) (4 pages)

We trained **SmolLM2-135M** in two stages on the ABCD v1.1 customer-support data — continued pretraining on the
domain text, then supervised fine-tuning (SFT) on 591 question–answer pairs — and scored the base, CPT and SFT
models on our 30 test questions.

## Results

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
|---|---|---|---|---|---|
| Small model, base | 14.93 | 31.7% | 16.7% | 0.58 | SmolLM2-135M, no training |
| + continued pretraining | 3.28 | 8.3% | 0.0% | 0.63 | 3 passes, lr 1e-4, 512-token blocks, policy docs ×5 |
| + SFT | 4.52 | **38.3%** | 16.7% | 1.11 | 591 pairs (533/58), 2.5 passes, lr 5e-5, loss on answers only |

Accuracy: exact match for the 12 short answers, two human graders (1 / 0.5 / 0) for the 18 long answers.
First-round grader agreement **49/54 = 90.7%, Cohen's κ = 0.742**. Full table: [`SCOREBOARD.md`](SCOREBOARD.md).

## Team and roles

| Member | Role | Contribution |
|---|---|---|
| Noor Attoun | Team lead, training, grader A, report | Ran the notebook on Kaggle (CPT and SFT), chose the model and settings, graded as grader A, wrote the report and decision log |
| Shahd Al-Omari | SFT data, grader B, error analysis | Built and reviewed the 591 SFT pairs and the overlap check, graded as grader B, labelled the 19 wrong SFT answers |
| Faten Jadou | Corpus and test set | Prepared the CPT corpus (cleaning, PII placeholders, splits) and revised the test set v1 → v2 |
| Yaqeen Ziad | Results and repository | Built the scoreboard, tables and figures, organised the repository and README |

All four members hand-checked 15 SFT pairs each (`results/sft_hand_check_graded.xlsx`).

## Code

The pipeline exists in two equivalent forms, built from the same code, settings and seed:

* **Scripts** (the layout of the brief): `config.yaml` → `train_cpt.py` → `train_sft.py` → `score.py`, with the shared
  code in `src/`.
* **One notebook** with the outputs of our final run: [`notebooks/phase2.ipynb`](notebooks/phase2.ipynb), section by
  section (corpus, test set and SFT pairs, template and scoring, base model, CPT, SFT, scoreboard, human grading,
  error analysis).

| File | What it does |
|---|---|
| `config.yaml` | every setting: model names, data splits, training settings, generation settings, seed 42 |
| `train_cpt.py` | builds the cleaned corpus (held-out 10% by document), held-out perplexity before → after, continued pretraining with early stopping, a checkpoint per pass, loss curve |
| `train_sft.py` | overlap check against the test set, ABCD dialogue template, loss on answer tokens only, 10% validation, early stopping, loss curve |
| `score.py` | runs the base, CPT and SFT models on the 30 test questions through **one** function (`src/models.generate_answer`) with the same settings, scores them, writes `results/results_<model>.csv` and `SCOREBOARD.md` |
| `src/` | `config.py` (settings, seed), `data_prep.py` (cleaning, PII placeholders, splits, SFT pairs, overlap check), `template.py` (chat template, answer-only labels), `models.py` (loading, generation, perplexity), `training.py` (Trainer settings, checkpoints, plots), `scoring.py` (exact match, refusal check, human grades) |

**Scoring.** Short answers: exact match on the first word or number. Longer answers: the final grade of our two
graders (`results/grading/human_grades.csv`), applied only when the model's answer is identical to the graded one; a
new answer is scored by the provisional keyword rubric and flagged for grading.

## How to train and score

Setup: `pip install -r requirements.txt` (Python 3.10+, a GPU for training). On Kaggle (free T4 GPU) upload this
folder as a Dataset and run [`notebooks/run_on_kaggle.ipynb`](notebooks/run_on_kaggle.ipynb); with *Internet: On* and an
`HF_TOKEN` secret it runs these commands:

```bash
python train_cpt.py --push      # continued pretraining, ~30 min on a T4  (--quick: 5% of the data, 30 steps)
python train_sft.py --push      # SFT, ~2 min                            (--quick)
python score.py                 # all three models on the 30 test questions -> results/, SCOREBOARD.md
python score.py --ask SP-07 SP-25   # the same questions answered by the three models, side by side
```

The ABCD conversations are downloaded from the official GitHub release. A sample of the cleaned corpus is in
`data/cpt/` (`train_sample.jsonl`, and the full `validation.jsonl` and `heldout.jsonl`).

## Repository

```
report/Phase2_Report.pdf          the report (4 pages)
README.md, SCOREBOARD.md, config.yaml, requirements.txt, .env.example
train_cpt.py, train_sft.py, score.py, src/
notebooks/phase2.ipynb            the whole pipeline in one notebook, with outputs
notebooks/run_on_kaggle.ipynb     runs the three scripts on Kaggle (run_on_kaggle_output.ipynb = our run, with outputs)
tests/test_questions.csv          30 test cases (v2, used for scoring)
tests/test_questions_v1_original.csv, build_test_v2.py   v1 and the v1 -> v2 changes
data/guidelines.json, kb.json     ABCD v1.1 documents
data/cpt/                         cleaned corpus: train sample, validation and held-out splits
data/sft_pairs.jsonl              591 SFT pairs           data/policy_handbook.jsonl   42 handbook entries
data/grading/grader_A_round1.xlsx, grader_B_round1.xlsx   the two graders' first-round sheets
results/results_{base,cpt,sft}.csv     every answer of every model with its score (one file per model)
results/grading/                  agreement.json, disagreements.xlsx, final_grades.csv, human_grades.csv
results/sft_hand_check_graded.xlsx     60 SFT pairs checked by hand
results/sft_test_overlap_audit.csv     nearest SFT question for each test question
results/error_analysis_reviewed.xlsx, error_counts.json   causes of the 19 wrong SFT answers
results/cpt_info.json, sft_info.json, cpt_samples.csv, corpus_stats.csv
figures/                          cpt_loss.png, sft_loss.png (loss curves), error_causes.png
tools/                            build_sft_pairs.py, handbook.py (+ _v1_pairs.py, used by build_sft_pairs.py)
```

## Checkpoints and keys

Checkpoints are not in Git. The best continued-pretraining and SFT checkpoints are on the Hugging Face Hub (private
repositories; access on request): **[`noorattoun/abcd-smollm2-cpt`](https://huggingface.co/noorattoun/abcd-smollm2-cpt)**
and **[`noorattoun/abcd-smollm2-sft`](https://huggingface.co/noorattoun/abcd-smollm2-sft)**. `score.py` loads them from
there when `checkpoints/` is not present (set `HF_TOKEN`).

The Hub checkpoints come from our run of the scripts ([`notebooks/run_on_kaggle_output.ipynb`](notebooks/run_on_kaggle_output.ipynb)).
That run reproduced every number in the scoreboard (perplexity 14.93 / 3.28 / 4.52, accuracy 31.7% / 8.3% / 38.3%).
53 of the 54 hand-graded answers came out word for word the same; the SFT answer to SP-22 changed wording
("Order Received or In Transit" instead of "Out for Delivery or Delivered") and still invents an order status. No keys are committed: the only secret is `HF_TOKEN`
(see `.env.example`), kept in Kaggle secrets or an environment variable.
