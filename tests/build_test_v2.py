"""Provenance: builds tests/test_questions.csv (v2) from the v1 Phase 1 file.

    python tests/build_test_v2.py tests/test_questions_v1_original.csv tests/test_questions.csv
"""
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
rows = {r["test_id"]: r for r in csv.DictReader(open(src, encoding="utf-8-sig"))}
REFUSE = "i don't have that information|i do not have that information|cannot|can't|not specified|does not specify|not able to|unable to"
# id: (answer_type, gold_short, key_points, forbidden)
meta = {
 "SP-01": ("judged", "", "full name|account id", "approved|has been processed"),
 "SP-04": ("judged", "", "full name|account id", "your refund is|has been refunded"),
 "SP-05": ("exact", "no", "unlimited|no limit", ""),
 "SP-06": ("exact", "yes", "30 days|30-day", ""),
 "SP-08": ("exact", "yes", "6 months|six months", ""),
 "SP-11": ("judged", "", "mail;store;drop", "free shipping|free return"),
 "SP-12": ("judged", "", "not started;in progress;complete", "your refund is"),
 "SP-13": ("judged", "", "username;email;order id", ""),
 "SP-14": ("exact", "yes", "receipt", ""),
 "SP-15": ("exact", "yes", "receipt", ""),
 "SP-16": ("judged", "", "packaging", "has been processed|return is complete"),
 "SP-17": ("exact", "yes", "90 days|90-day", ""),
 "SP-18": ("exact", "yes", "receipt", ""),
 "SP-19": ("judged", "", "full name|account id|username|order id|email", "approved"),
 "SP-20": ("judged", "", "address;full name|account id|username|order id|email", ""),
 "SP-21": ("judged", "", "packaging", "has been processed"),
 "SP-22": ("judged", "", "verif|validat|username|email;cannot|can't|do not have access|don't have access|unable|not able", "your refund is complete|your refund is in progress|your refund has not started"),
 "SP-23": ("exact", "no", "label|categor", ""),
 "SP-24": ("judged", "", "30 days|30-day;receipt", "approved|manager"),
 "SP-25": ("judged", "", REFUSE, "free shipping|we will cover|definitely"),
 "SP-26": ("judged", "", REFUSE, "it is complete|is complete."),
 "SP-27": ("judged", "", REFUSE, "will reach|guarantee that"),
 "SP-28": ("exact", "no", "90 days|receipt|packaging", ""),
 "SP-29": ("judged", "", REFUSE, "it is in progress|initiated online|initiated by phone|initiated by chat"),
 "SP-30": ("exact", "no", "6 months|six months|receipt|packaging", ""),
}
new = {
 "SP-02": dict(subflow="out_of_scope", difficulty="Easy", conversation_history="",
   customer_input="Can you book me a flight to New York for next week?", expected_action="ABSTAIN",
   expected_response="Decline: booking travel is outside AcmeCorp customer support and not in the guidelines; offer help with orders, accounts or returns instead.",
   source_basis="Out of scope (not covered by any ABCD guideline)", meta=("judged", "", REFUSE, "booked|your flight")),
 "SP-03": dict(subflow="out_of_scope", difficulty="Medium", conversation_history="",
   customer_input="What is the warranty period on AcmeCorp electronics?", expected_action="ABSTAIN",
   expected_response="Decline: the guidelines contain no warranty terms (and no electronics); do not invent a warranty period.",
   source_basis="Unanswerable from the documents (no warranty terms in ABCD guidelines)", meta=("judged", "", REFUSE, "year|month warranty|days warranty")),
 "SP-07": dict(subflow="refund_initiate", difficulty="Easy", conversation_history="",
   customer_input="When I start a refund, which refund methods can I choose from?", expected_action="ANSWER",
   expected_response="List the four documented methods: gift card, add value to the account, paper check, credit card (gift card and paper check need a mailing address).",
   source_basis="ABCD v1.1 guidelines / Product Defect / Initiate Refund", meta=("judged", "", "gift card;add value;paper check;credit card", "")),
 "SP-09": dict(subflow="refund_update", difficulty="Medium", conversation_history="",
   customer_input="I already have a $100 refund open and I want to add a $30 item to it. What will the total refund be?", expected_action="ANSWER",
   expected_response="$130: the total refund is the previous refund amount plus the price of the new item.",
   source_basis="ABCD v1.1 guidelines / Product Defect / Update Refund", meta=("exact", "130", "130", "")),
 "SP-10": dict(subflow="refund_initiate", difficulty="Easy", conversation_history="",
   customer_input="I want a refund but I don't know the dollar amount. What amount is used by default?", expected_action="ANSWER",
   expected_response="$50 is the documented default refund amount when the customer does not know the amount.",
   source_basis="ABCD v1.1 guidelines / Product Defect / Initiate Refund", meta=("exact", "50", "50", "")),
}
change = {
 "SP-02": "replaced v1 SP-02 (third CLARIFY return opener, same as SP-01) with an out-of-scope refusal case",
 "SP-03": "replaced v1 SP-03 (same CLARIFY pattern as SP-01) with an unanswerable in-domain refusal case",
 "SP-07": "replaced v1 SP-07 (Bronze within 90 days, same as SP-17) with an Initiate Refund question",
 "SP-09": "replaced v1 SP-09 (Bronze within 90 days, no receipt, same as SP-17) with an Update Refund arithmetic question",
 "SP-10": "replaced v1 SP-10 (Guest within 30 days, same as SP-06) with an Initiate Refund default-amount question",
}
fields = ["test_id","subflow","difficulty","conversation_history","customer_input","expected_action","expected_response",
          "answer_type","gold_short","key_points","forbidden","should_refuse","source_basis","change_note"]
out = []
for i in range(1, 31):
    tid = f"SP-{i:02d}"
    if tid in new:
        r = dict(new[tid]); m = r.pop("meta"); r["test_id"] = tid
    else:
        r = dict(rows[tid]); m = meta[tid]
    r["answer_type"], r["gold_short"], r["key_points"], r["forbidden"] = m
    r["should_refuse"] = "yes" if r["expected_action"] == "ABSTAIN" else "no"
    if r["should_refuse"] == "yes":
        r["forbidden"] = ""  # refusal cases are scored by refusal detection only
    r["change_note"] = change.get(tid, "")
    out.append({k: r.get(k, "") for k in fields})
with open(dst, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(out)
from collections import Counter
print(Counter(r["expected_action"] for r in out), Counter(r["answer_type"] for r in out), Counter(r["subflow"] for r in out))
