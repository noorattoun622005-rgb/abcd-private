"""One scoring method for every model.

* Exact cases (yes/no or a number): first word (or first number) of the answer vs. gold_short.
* Refusal cases: 1 if the model declines, else 0.
* Longer answers: the two graders' final grade (results/grading/human_grades.csv), used only when the model's
  answer is identical to the answer that was graded. A new answer gets the provisional keyword rubric and is
  flagged for grading.
"""
import re

import pandas as pd

REFUSAL_RX = re.compile("|".join([
    r"i (do not|don't) have (that|this) information", r"outside what acmecorp", r"not in (our|the) (support )?guidelines",
    r"\b(cannot|can't|can not|unable to|not able to) (promise|guarantee|confirm|determine|look up|access|see|check|say)",
    r"\b(is|are) not (specified|documented|covered)", r"\b(does|do) not (specify|say|state|cover)"]), re.I)


def is_refusal(answer):
    return bool(REFUSAL_RX.search(answer or ""))


def predicted_action(answer):
    if is_refusal(answer):
        return "ABSTAIN"
    if "?" in answer and re.search(r"full name|account id|username|order id|email|address", answer, re.I):
        return "CLARIFY"
    return "ANSWER" if answer.strip() else "NONE"


def short_answer(answer, gold):
    body = re.sub(r"\s+", " ", answer.lower()).strip()
    if all(re.fullmatch(r"\d+(\.\d+)?", g.strip()) for g in gold.split("|")):
        m = re.search(r"\$?\s*(\d+(?:\.\d+)?)", body)
        return m.group(1) if m else ""
    m = re.match(r"[a-z']+", body)
    return m.group(0) if m else ""


def score_case(case, answer):
    """Automatic score: exact match, refusal check, or the provisional keyword rubric for longer answers."""
    if case["should_refuse"] == "yes":
        ok = is_refusal(answer)
        return float(ok), "declined" if ok else "should have declined"
    if case["answer_type"] == "exact":
        got = short_answer(answer, case["gold_short"])
        return float(got in [g.strip() for g in case["gold_short"].split("|")]), f"short answer '{got}' vs '{case['gold_short']}'"
    body = answer.lower()
    groups = [g for g in case["key_points"].split(";") if g.strip()]
    missing = [g for g in groups if not any(alt.strip() and alt.strip() in body for alt in g.split("|"))]
    cover = 1 - len(missing) / max(len(groups), 1)
    forbidden = [f for f in case["forbidden"].split("|") if f.strip() and f.strip() in body]
    s = 0.0 if forbidden else 1.0 if cover == 1 else 0.5 if cover >= 0.5 else 0.0
    note = "; ".join(filter(None, ["forbidden: " + ", ".join(forbidden) if forbidden else "",
                                   "missing: " + ", ".join(m.split("|")[0] for m in missing) if missing else ""]))
    return s, "provisional rubric" + (f" ({note})" if note else "")


def same_text(x, y):
    n = lambda s: re.sub(r"\s+", " ", str(s)).strip().lower()
    return n(x) == n(y)


def load_human_grades(path):
    """(model, test_id) -> (graded answer, final grade)."""
    try:
        g = pd.read_csv(path, keep_default_na=False)
    except FileNotFoundError:
        return {}
    return {(r.model, r.test_id): (r.graded_answer, float(r.final_grade)) for r in g.itertuples()}


def summarize(df):
    ref = df[df.expected_action == "ABSTAIN"]; ans = df[df.expected_action != "ABSTAIN"]
    return {"accuracy": df.score.mean(), "refusal_accuracy": (ref.score >= 1).mean() if len(ref) else float("nan"),
            "over_refusal": ans.refused.mean() if len(ans) else float("nan"),
            "stopped_cleanly": df.stopped.mean(), "latency_s": df.latency_s.mean(),
            "human_graded": int((df.score_source == "human").sum()),
            "to_grade": int((df.score_source == "provisional").sum())}
