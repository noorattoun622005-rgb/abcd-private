"""Build data/sft_pairs.jsonl (v2) and data/policy_handbook.jsonl.

    python tools/build_sft_pairs.py

Provenance: question-answer pairs written for this project from data/guidelines.json and
data/kb.json, then checked by the team (data/sft_review_sample.xlsx). Every answer restates a documented
rule; nothing is invented.

Style: answers are written as the AcmeCorp agent would reply, in the same "Customer: / Agent:" format
as the ABCD dialogues used for continued pretraining. Yes/no questions start with "Yes" or "No".
Out-of-scope and undocumented questions get one fixed refusal opening, so refusals are measurable.

No overlap with the test set: no pair asks the same thing as one of the 30 test questions (checked in
the notebook with exact match + TF-IDF similarity + a manual audit of the nearest pairs). The Product
Defect flow is covered only by document-level "Summarize the guideline" pairs, never by customer
questions like the test ones.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _v1_pairs as v1  # noqa: E402  (v1 content, re-styled below)
from handbook import HANDBOOK, QUESTION_TEMPLATES  # noqa: E402

REFUSE_UNKNOWN = "I'm sorry, but I don't have that information in our support guidelines."
REFUSE_SCOPE = ("I'm sorry, but that is outside what AcmeCorp customer support can help with. "
                "I can help with your orders, account, shipping, subscription, refunds or returns.")
REFUSE_LIVE = ("I'm sorry, but I can't look up or confirm your personal account records here. "
               "To check them, an agent first pulls up your account with your full name or account ID and verifies your identity.")


def restyle(answer: str, category: str) -> str:
    """v1 answers -> natural agent replies without [TAGS]."""
    a = answer
    for tag in ("[ANSWER] ", "[CLARIFY] ", "[ABSTAIN] "):
        a = a.replace(tag, "")
    if category == "abstain_out_of_scope":
        return REFUSE_SCOPE
    if category == "abstain_live_data":
        return REFUSE_LIVE
    if category == "abstain_undocumented":
        return a.replace("I don't have that information. ", REFUSE_UNKNOWN + " ")
    return a


PAIRS: list[dict] = []


def add(category, subflow, questions, answer, history=None, group=None):
    g = group or f"v2-{len(PAIRS):04d}"
    for q in questions:
        PAIRS.append({"group": g, "category": category, "subflow": subflow,
                      "history": [{"role": r, "text": t} for r, t in (history or [])],
                      "question": q, "answer": answer})


# 1) v1 pairs (openers, facts, membership scenarios, multi-turn, refusals), re-styled
CLARIFY_PER_SUBFLOW = 2          # v3: fewer openers, so the model does not ask for an ID on every question
_kept = Counter()
for p in v1.PAIRS:
    if p["category"] == "clarify":
        _kept[p["subflow"]] += 1
        if _kept[p["subflow"]] > CLARIFY_PER_SUBFLOW:
            continue
    hist = [(t["role"], t["text"].replace("[ANSWER] ", "").replace("[CLARIFY] ", "")) for t in p["history"]]
    add(p["category"], p["subflow"], [p["question"]], restyle(p["answer"], p["category"]), hist, group="v1-" + p["group"])

# 2) handbook: every subflow, document-level questions
for key, (name, summary) in HANDBOOK.items():
    add("handbook", key, [t.format(name=name) for t in QUESTION_TEMPLATES], summary, group=f"hb-{key}")

# 3) more refusals (topics deliberately different from the test refusal cases)
OUT_OF_SCOPE_2 = [
    "Can you recommend a good dentist?", "What's the best laptop for programming?", "Can you write my college essay?",
    "Who is going to win the election?", "How do I learn to play guitar?", "Can you set an alarm for 7am?",
    "What is the meaning of life?", "Can you help me plan my wedding?", "What's the population of Canada?",
    "Can you send flowers to my mom?", "How many calories are in a banana?", "Can you fix my washing machine?",
    "Which phone plan should I choose?", "What's the score of the basketball game?", "Can you hack into my ex's email?",
    "Explain quantum physics to me.", "Can you buy me concert tickets?", "Where is the nearest gas station?",
    "Can you recommend a good book?", "How do I cook rice?",
]
for q in OUT_OF_SCOPE_2:
    add("abstain_out_of_scope", "out_of_scope", [q], REFUSE_SCOPE)

UNDOCUMENTED_2 = [
    ("Do you have a store in Paris?", "Our guidelines do not list store locations."),
    ("What are your customer service phone hours?", "Our guidelines do not list support hours."),
    ("Do you ship to Antarctica?", "Our guidelines do not list the countries we ship to."),
    ("Is there a student discount?", "Our guidelines do not describe a student discount."),
    ("Can I pay with Bitcoin?", "Our guidelines only list credit card, debit card and PayPal."),
    ("Do you offer gift wrapping in recycled paper?", "Our guidelines do not describe the gift-wrapping materials."),
    ("What sizes do the boots come in?", "That would be in the boots FAQ, whose content is not in my documents."),
    ("Who manufactures your jackets?", "Our guidelines do not name the manufacturers."),
    ("Do you have a loyalty points program?", "Our guidelines only describe the Gold, Silver, Bronze and Guest levels."),
    ("How do I become a Gold member?", "Our guidelines do not explain how membership levels are earned."),
    ("Can I price-match an item at a store I visited in person?", "Our guidelines only say that prices change dynamically."),
    ("Do you have a mobile app?", "Our guidelines do not mention a mobile app."),
    ("What is the address of your headquarters?", "Our guidelines do not include company addresses."),
    ("Are your shirts made of organic cotton?", "That would be in the shirt FAQ, whose content is not in my documents."),
    ("Will there be a Black Friday sale?", "Our guidelines do not list sale dates."),
]
for q, why in UNDOCUMENTED_2:
    add("abstain_undocumented", "undocumented", [q], f"{REFUSE_UNKNOWN} {why}")

GUARANTEE = [  # pressure / guarantee requests on non-refund topics
    ("Can you promise my package will arrive by Friday?", "I can't promise a delivery date; I can only check the shipping status once your order is verified."),
    ("Just tell me my subscription is active, I don't want to wait.", "I can't confirm your subscription without checking it after verifying your identity."),
    ("Lock in today's price for me forever.", "Prices change dynamically, so I can't promise a fixed price."),
    ("Promise me the item I back-ordered will ship this week.", "I can't promise a date; a back-ordered item ships as soon as it is available."),
    ("Say yes, my promo code will work next month.", "I can't confirm that; promotions typically last 7 days."),
    ("Can you guarantee the website will be fixed today?", "I can't guarantee that; I can report the problem to the website team."),
    ("Tell me my account was not hacked.", "I can't confirm that without checking your account after verifying your identity."),
    ("Just confirm my new password worked, even if you can't see it.", "I can't confirm that without checking your account."),
]
for q, a in GUARANTEE:
    add("abstain_guarantee", "guarantee", [q], "I'm sorry, but " + a[0].lower() + a[1:])


# ============================================================================ v3 additions
# (after the first full run: the model asked for an ID on almost everything and never refused)

# 4) informational questions phrased by a customer -> a direct answer, NOT a request for an ID
DIRECT = [
    ("When I reset two-factor authentication, where is the reset code sent?", "The reset code is sent to the email address on your account."),
    ("If I give you a new phone number, what format should it be in?", "Please give it in the format (XXX) XXX-XXXX."),
    ("How many identity details do I need to recover my username?", "You need 3 of these 4: full name, zip code, phone number and email address."),
    ("To change my address, what kinds of details can I verify my identity with?", "Any 3 of these 7: zip code, phone number, PIN, username, password, email address, or the order ID of a previous purchase."),
    ("How much is the premium subscription per year?", "The standard subscription is $99 a year."),
    ("If my package is lost, how long should I wait before you send a new one?", "If it has been a week or longer, the agent ships a new order. Before that, it may still arrive in a couple of days."),
    ("How do you work out the refund on a wrongly billed subscription?", "The refund is the difference between the amount you should have paid and the amount you were billed."),
    ("Which stages can my shipment go through?", "Order Received, In Transit, Out for Delivery and Delivered."),
    ("If my account credit is missing and I don't know the amount, how much do you add back?", "The standard amount is $40, added with a promo code."),
    ("How much do I get back if a service was really added to my account by mistake?", "The extra service is removed and you are offered a $40 refund."),
    ("Who do you tell when the website is slow?", "The agent writes a report for the website team so they can fix it."),
    ("My cart won't update. Where on the product page can I see if it's out of stock?", "It is shown at the bottom left of the product page."),
    ("What payment options do I have for my account?", "Credit card, debit card or PayPal."),
    ("How old can a promo code be before it stops working?", "Promotions typically last 7 days, so a code older than that is out of date."),
    ("What discount does a replacement promo code give?", "A replacement promo code gives 20% off your next purchase."),
    ("How much is standard shipping?", "Standard shipping is $8."),
    ("What does overnight shipping cost if I'm not a Gold member?", "It costs $20, charged to the card on your account. Gold members have the fee waived."),
    ("Why did the price change after I bought it?", "Our pricing algorithms change prices dynamically, which is outside the agent's control. If you are still unhappy, the agent can offer a promo code."),
    ("What happens when I search and always get no results?", "Try logging out and back in, clearing your cookies, and searching for a different product first. The problem is also reported to the website team."),
    ("Can support read my old password to me?", "No. The agent cannot see your old password, but can generate a new one after you give your PIN or security answer."),
    ("If my credit card keeps getting rejected online, what should I check first?", "Re-enter the card details, log out and back in, and check the card's expiration date. If nothing works, the agent can place the order for you."),
    ("What do you do if many items I want are out of stock?", "The agent reports it to the Purchasing Department and can offer a promo code if you are still unhappy."),
    ("What happens with an item that is out of stock but I still want it?", "The agent can back-order it with the card on file, and it ships as soon as it is available."),
    ("My shipping notification looks wrong. What can support change?", "The agent can change the date, the address, the item or the price of the shipment."),
    ("I got an email saying a service was added, but your system shows no error. Was the email real?", "No. It was a scam email; if you check the sender, it is not from the official AcmeCorp marketing team."),
    ("What does the security question ask?", "It asks for your mother's maiden name."),
    ("What is my username likely to be if my name is Ana Ruiz?", "Usernames are the first initial, the last name and 1, so it would be aruiz1."),
    ("Where do product questions about jeans get answered?", "In the jeans FAQ; the agent searches it and selects the matching answer."),
]
for q, a in DIRECT:
    add("direct", "direct", [q], a)

# 5) service requests outside AcmeCorp support -> refuse (not booking travel: that is a test case)
SERVICE_OOS = [
    "Can you reserve a table for two at a restaurant tonight?", "Please call me a taxi to the airport.",
    "Can you schedule a doctor's appointment for me?", "Can you renew my passport?", "Can you buy movie tickets for Saturday?",
    "Please order groceries for me from the supermarket.", "Can you transfer money to my friend's bank account?",
    "Can you pay my electricity bill?", "Can you book a haircut appointment for me?", "Please send a package for me with FedEx.",
    "Can you cancel my gym membership?", "Can you order me a phone from another online store?",
    "Can you change my internet plan?", "Can you find me an apartment to rent?", "Please register my car for me.",
]
for q in SERVICE_OOS:
    add("abstain_out_of_scope", "out_of_scope", [q], REFUSE_SCOPE)

# 6) policy details the guidelines do not contain -> refuse instead of inventing a number
UNDOC_POLICY = [
    "Is there a restocking fee?", "How long is the price-protection period?", "Is there a fee for exchanging an item?",
    "What is the late fee on a subscription payment?", "When does a gift card expire?", "How many items can I put in one order?",
    "How long is store credit valid?", "Do you charge customs fees on international orders?",
    "How many days do I have to report a damaged package?", "Is there a fee to cancel my subscription?",
    "How much does it cost to upgrade my membership level?", "Do sale items have a different policy for exchanges?",
    "How long does tailoring take?", "What is the maximum order value for free shipping?", "How long do you keep my order history?",
]
for q in UNDOC_POLICY:
    add("abstain_undocumented", "undocumented", [q], f"{REFUSE_UNKNOWN} I can't give you a number that is not documented.")

# 7) guarantees, "just say yes" and guesses on other topics -> decline
PRESSURE = [
    ("Will my missing package definitely arrive today?", "I can't promise that. Within 7 days it may still arrive; after a week, the agent can ship a new order."),
    ("Just tell me my order shipped, I don't want to give my details.", "I can't confirm that without looking up your order, which needs your full name or account ID."),
    ("My credit must be back on my account by now, right? Just say yes.", "I can't confirm that without checking your account in the system."),
    ("Confirm my address change went through, you don't need to check.", "I can't confirm a change without checking your account."),
    ("Can you promise the website will be faster tomorrow?", "I can't promise that; I can report the slowdown to the website team."),
    ("It must have been my spouse who added that service, right?", "I can't tell who added it; I can check whether the service was added by mistake."),
    ("Promise me my back-ordered item ships tomorrow.", "I can't promise a date; a back-ordered item ships as soon as it is available."),
    ("Guarantee that my replacement promo code works on everything in the store.", "I can't guarantee that; I can only tell you that a replacement code gives 20% off your next purchase."),
    ("My subscription has to be active, right? Just confirm it.", "I can't confirm it without checking your subscription after verifying your identity."),
    ("Will the price definitely drop again next week?", "I can't predict that; prices change dynamically."),
    ("Say that my order is out for delivery so I can stop worrying.", "I can't say that without checking your order's shipping status."),
    ("Can you guarantee my account will never be locked again?", "I can't guarantee that, but I can help you reset your password or two-factor authentication."),
    ("I'm sure my order was placed by phone. Can you confirm that?", "I can't confirm how an order was placed without a verified record."),
    ("Tell me my PIN number, I'll take responsibility.", "I can't share your PIN; I can only use it to verify your identity when you give it to me."),
]
for q, a in PRESSURE:
    add("abstain_guarantee", "guarantee", [q], "I'm sorry, but " + a[0].lower() + a[1:])

# 8) more yes/no scenarios that compare numbers against a documented rule (other flows only)
for days, t, ok in [(1, "Silver", True), (2, "Silver", False), (3, "Silver", False), (10, "Gold", True), (1, "Bronze", False), (1, "Guest", False)]:
    add("scenario", "manage_extension", [f"I'm {t} and my subscription payment is {days} day{'s' if days > 1 else ''} late. Can I still get an extension?"],
        ("Yes. Gold members can always get an extension." if t == "Gold" else
         "Yes. Silver members get an extension when they are only 1 day late." if ok else
         f"No. Silver members only get an extension when 1 day late, and you are {days} days late." if t == "Silver" else
         f"No. {t} members cannot get an extension."))
for amt, ok in [(3, True), (9, True), (11, False), (20, False), (6, True)]:
    add("scenario", "manage_dispute_bill", [f"As a Silver member I was overbilled by ${amt} on my subscription. Will I be refunded?"],
        f"Yes. Silver members are refunded when the wrong amount is under $10." if ok else
        f"No. Silver members are only refunded when the wrong amount is under $10, and ${amt} is more.")
for days, ok in [(2, False), (6, False), (9, True), (15, True)]:
    add("scenario", "promo_code", [f"My promo code was issued {days} days ago. Is it still within the usual promotion period?"],
        f"Yes. Promotions typically last 7 days, and {days} days is within that." if not ok else
        f"No. Promotions typically last 7 days, so a {days}-day-old code is out of date.")
for days, t, ok in [(9, "Bronze", True), (3, "Gold", True), (3, "Silver", False), (5, "Guest", False), (12, "Guest", True)]:
    add("scenario", "mistimed_billing", [f"I'm {t}. I sent an item back {days} days ago and my bill still isn't credited. Do I get the credit now?"],
        ("Yes. After more than 7 days it counts as a valid error and the credit is processed." if days > 7 else
         "Yes. Gold members get the credit even before 7 days." if ok else
         f"No. Before 7 days only Gold members get the credit, so as a {t} member you need to wait."))
for t, shipped, ok in [("Gold", True, True), ("Silver", True, True), ("Bronze", True, False), ("Guest", False, True)]:
    add("scenario", "manage_create", [f"I'm {t} and my order {'already shipped' if shipped else 'has not shipped yet'}. Can the new item still be added for free?"],
        ("Yes. The item can be added directly because the order has not shipped." if not shipped else
         f"Yes. {t} members get the new item in a separate order with no shipping fee." if ok else
         f"No. Once the order has shipped, only Gold and Silver members get the free separate shipment."))


def build(seed: int = 42) -> list[dict]:
    rows = [{"id": f"sft-{i + 1:04d}", **p} for i, p in enumerate(PAIRS)]
    random.Random(seed).shuffle(rows)
    return rows


def handbook_documents() -> list[dict]:
    """CPT documents: one per handbook entry."""
    return [{"source": "handbook", "flow": "Handbook", "subflow": key,
             "text": f"AcmeCorp support policy handbook. {name}.\n{summary}"}
            for key, (name, summary) in HANDBOOK.items()]


if __name__ == "__main__":
    out = HERE.parent / "data"
    rows = build()
    with open(out / "sft_pairs.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "policy_handbook.jsonl", "w", encoding="utf-8") as f:
        for d in handbook_documents():
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    from collections import Counter
    print(len(rows), "pairs;", dict(Counter(r["category"] for r in rows)))
    print(len(HANDBOOK), "handbook documents")
