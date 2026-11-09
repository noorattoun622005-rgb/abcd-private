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

## The notebook

**Everything is in one notebook: [`notebooks/phase2.ipynb`](notebooks/phase2.ipynb)** (saved with the outputs of
our final run). Sections follow the brief:

| § | Section | Brief requirement |
|---|---|---|
| 0 | Setup and configuration | fixed seed 42, all settings in one `CFG` |
| 1 | Continued-pretraining corpus | clean text, held-out 10% split by document, size in tokens |
| 2 | Test set and SFT pairs | 300–1,000 pairs, ≥ 50 checked by hand, no overlap with the test set |
| 3 | Prompt template, generation, scoring | one way to call every model, one scoring method |
| 4 | Base model | score the base model |
| 5 | Stage 1 — continued pretraining | perplexity before/after, loss curves, checkpoint per pass, early stopping |
| 6 | Stage 2 — SFT | fixed template, loss on answer tokens only, 10% validation |
| 7 | Scoreboard | one row per version, the same questions for all models |
| 8 | Human grading | blind sheets, two graders, agreement + κ, disagreements settled |
| 9 | Error analysis | every wrong SFT answer → one cause |
| 10–11 | Save | results archive and report numbers |

### How to run (Kaggle, free T4 GPU)

1. Upload this folder as a Kaggle **Dataset**.
2. New notebook → *File → Import Notebook* → `notebooks/phase2.ipynb`.
3. *Settings*: **Accelerator = GPU T4**, **Internet = On**; *Add Input* → the dataset.
4. *Run All* (about 45 minutes). `QUICK_TEST = True` in §0 runs a 5-minute smoke test instead.

The ABCD dialogues are downloaded from the official GitHub release by the notebook. Outputs go to
`/kaggle/working/` (`SCOREBOARD.md`, `results/`, `figures/`, `checkpoints/`, `phase2_results.zip`).

## Repository

```
report/Phase2_Report.pdf          the report (4 pages)
notebooks/phase2.ipynb            the whole pipeline, with outputs   (phase2_src.py = same code as a script)
SCOREBOARD.md                     final scoreboard
tests/test_questions.csv          30 test cases (v2, used for scoring)
tests/test_questions_v1_original.csv, build_test_v2.py   v1 and the v1 → v2 changes
data/guidelines.json, kb.json     ABCD v1.1 documents
data/sft_pairs.jsonl              591 SFT pairs           data/policy_handbook.jsonl   42 handbook entries
data/grading/grader_A_round1.xlsx, grader_B_round1.xlsx   the two graders' first-round sheets
results/results_{base,cpt,sft}.csv     every answer of every model with its score
results/grading/                  agreement.json, disagreements.xlsx, final_grades.csv (every grade and change)
results/sft_hand_check_graded.xlsx     60 SFT pairs checked by hand
results/sft_test_overlap_audit.csv     nearest SFT question for each test question
results/error_analysis_reviewed.xlsx, error_counts.json   causes of the 19 wrong SFT answers
results/summary.json, report_numbers.json, cpt_info.json, sft_info.json, corpus_stats.csv
figures/                          cpt_loss.png, sft_loss.png, error_causes.png
tools/                            build_sft_pairs.py, handbook.py, make_notebook.py
```

## Checkpoints and keys

Model checkpoints are not stored in the repository because of their size. Running the notebook reproduces them
exactly (seed 42; our second full run reproduced every one of the 54 graded answers), and they are available
from the team on request. No keys are committed; the only secret is `HF_TOKEN` (see `.env.example`), needed only
to push checkpoints to the Hugging Face Hub.
