# Scoreboard

Same test set for every version: `tests/test_questions.csv` (30 cases, 6 should be refused). Greedy decoding, max 128 new tokens, identical prompt for all models. Accuracy = mean score (exact match for short answers, 1/0.5/0 hand grades for longer ones). Refusal accuracy = share of should-refuse cases declined. Perplexity on the 10% held-out documents. Latency = mean generation time per answer on a T4 GPU.

| Version | Held-out perplexity | Accuracy | Refusal accuracy | Avg. latency (s) | Notes |
|---|---|---|---|---|---|
| Small model, base | 14.93 | 31.7% | 16.7% | 0.58 | `HuggingFaceTB/SmolLM2-135M`, no training; human-graded |
| + continued pretraining | 3.28 | 8.3% | 0.0% | 0.63 | 3.0 passes, lr 0.0001, 512-token blocks, policy docs x5; human-graded |
| + SFT | 4.52 | 38.3% | 16.7% | 1.11 | 591 pairs (533/58), 2.5 passes, lr 5e-05, loss on answers only; human-graded |
