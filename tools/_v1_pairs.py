"""Source of the SFT question-answer pairs (data/sft_pairs.jsonl).

How the pairs were made
-----------------------
Written for this project directly from data/guidelines.json and
data/kb.json, then checked by the team (see data/sft_review_sample.csv). Every answer
restates a rule that is in the guidelines; nothing is invented.

Design rules
------------
* Same fixed answer format everywhere. An answer starts with one tag:
    [ANSWER]  a direct answer (yes/no questions start with "Yes." or "No.")
    [CLARIFY] the next piece of information the agent needs (usually full name or account ID)
    [ABSTAIN] "I don't have that information..." for out-of-scope or undocumented questions
* No overlap with the test set: the 30 test questions are about the Product Defect flow
  (return eligibility by membership level, return methods, the refund-status procedure,
  refund methods and amounts) plus two out-of-scope requests. The SFT pairs therefore
  cover the OTHER nine flows only, and the out-of-scope refusals use different topics.
  The Product Defect facts must come from continued pretraining (and, in Phase 3, RAG).
  `python -m src.overlap` checks this automatically.
* Paraphrases of one fact share a `group`, and the train/validation split is by group.
"""
from __future__ import annotations

import random

PAIRS: list[dict] = []
_GID = [0]


def add(category: str, subflow: str, questions: list[str], answer: str, history: list[tuple] | None = None):
    _GID[0] += 1
    g = f"g{_GID[0]:03d}"
    hist = [{"role": r, "text": t} for r, t in (history or [])]
    for q in questions:
        PAIRS.append({"group": g, "category": category, "subflow": subflow,
                      "history": hist, "question": q, "answer": answer})


ASK_ID = "could you please give me your full name or account ID?"

# =============================================================================== 1. CLARIFY
# Conversation openers: the guidelines start almost every flow with [Pull up Account].
CLARIFY = {
    "recover_username": (["I forgot my username.", "I can't remember the username for my account.",
                          "Can you help me find my username?", "What was my username again? I lost it.",
                          "I need my username, please."],
                         "I can help you recover your username. To pull up your account, " + ASK_ID),
    "recover_password": (["I forgot my password.", "I need to reset my password, I can't log in.",
                          "My password isn't working and I forgot it.",
                          "Help, I can't remember my account password."],
                         "I can help you get a new password. First, " + ASK_ID),
    "reset_2fa": (["I need to reset my two-factor authentication.", "My two-factor code isn't working anymore.",
                   "I lost access to my 2FA, can you reset it?", "How do I reset two-step verification on my account?",
                   "The two-factor authentication on my account is broken."],
                  "I can help reset your two-factor authentication. To start, " + ASK_ID),
    "status_service_added": (["There's a service on my account that I never added.",
                              "Someone added an extra service to my plan without asking me.",
                              "Why is there a new service on my account?",
                              "I see a subscription service I didn't sign up for."],
                             "I'm sorry about that, let me look into it. To pull up your account, " + ASK_ID),
    "status_service_removed": (["An important service was removed from my plan by mistake.",
                                "My service disappeared from my account.",
                                "You removed a service from my account that I still need.",
                                "Why was a service taken off my plan?"],
                               "Sorry for the trouble, I'll check that. First, " + ASK_ID),
    "status_shipping_question": (["Does my account get free international shipping?",
                                  "Can I ship to another country for free with my account?",
                                  "Is out-of-country shipping included with my account?"],
                                 "Let me check your account for you. To pull it up, " + ASK_ID),
    "status_credit_missing": (["Some credit is missing from my account.", "My account credit disappeared.",
                               "I had store credit and now it's gone.", "Where did the credit on my account go?"],
                              "I'm sorry for the inconvenience, let me look into the missing credit. First, " + ASK_ID),
    "manage_change_address": (["I need to change the address on my account.", "I moved and want to update my address.",
                               "Can you update my home address?", "Please change my account address."],
                              "I can update your address. To pull up your account, " + ASK_ID +
                              " You will also need to tell me your current street address."),
    "manage_change_name": (["I want to change the name on my account.", "I got married and need to update my name.",
                            "Can you correct the name on my profile?", "My name is spelled wrong on my account."],
                           "I can help change the name on your account. First, " + ASK_ID +
                           " I will also need your current first and last name."),
    "manage_change_phone": (["I need to update my phone number.", "I have a new phone number, can you change it?",
                             "Please change the phone number on file.", "My phone number on the account is old."],
                            "I can update your phone number. To pull up your account, " + ASK_ID +
                            " I will also need your current phone number."),
    "manage_payment_method": (["I want to change the payment method on my account.",
                               "Can I update the card saved on my account?",
                               "I'd like to switch my account to PayPal.", "Please change my saved payment method."],
                              "I can help change your payment method. First, " + ASK_ID),
    "status_mystery_fee": (["There's an extra fee on my order that I don't understand.",
                            "I was charged a fee I didn't expect.", "Why is my order total higher than it should be?",
                            "There is a mystery charge on my order."],
                           "I'm sorry about the unexpected charge, let me look into it. First, " + ASK_ID),
    "status_delivery_time": (["My delivery is scheduled for the morning but I need it in the evening.",
                              "Can I move my delivery time to the evening?",
                              "The delivery time on my order is wrong."],
                             "I can look into the delivery time. To pull up your account, " + ASK_ID),
    "status_payment_method": (["The wrong payment method was used for my order.",
                               "I want to change how I paid for my order.",
                               "My order was charged to the wrong card."],
                              "Let me help with the payment method on your order. First, " + ASK_ID),
    "status_quantity": (["My order confirmation shows two items but I only ordered one.",
                         "The email confirmation has the wrong quantity.",
                         "The quantity on my order email doesn't match what I ordered."],
                        "Sorry about the confusion, let me check the order. To pull up your account, " + ASK_ID),
    "manage_upgrade": (["I want to upgrade my order to overnight shipping.", "Can I get my order delivered tomorrow instead?",
                        "Please upgrade my shipping to overnight.", "Is it possible to speed up my delivery?"],
                       "I can check whether your shipping can be upgraded. First, " + ASK_ID),
    "manage_downgrade": (["I want to downgrade to next-week delivery to save on shipping.",
                          "Can I switch to slower shipping to pay less?",
                          "I don't need fast shipping, can you change it to next week?"],
                         "I can look into changing your delivery. To pull up your account, " + ASK_ID),
    "manage_create": (["I want to add another item to my existing order.", "Can I add something to an order I already placed?",
                       "Please add one more item to my order."],
                      "I can help add an item to your order. First, " + ASK_ID),
    "manage_cancel": (["I accidentally ordered two of the same item and only want one.",
                       "I ordered a duplicate by mistake, can you cancel one?",
                       "Please remove the extra item I ordered by accident."],
                      "I can help with the duplicate item. To pull up your account, " + ASK_ID),
    "bad_price_competitor": (["I found the same item cheaper at another store.",
                              "A competitor has a better price than you for what I bought.",
                              "Why is your price higher than other shops?"],
                             "I understand, let me look into it. To start, " + ASK_ID),
    "bad_price_yesterday": (["The item I bought was cheaper yesterday.", "The price dropped right after I ordered.",
                             "I paid more than yesterday's price."],
                            "I understand how frustrating that is. To start, " + ASK_ID),
    "out_of_stock_general": (["Everything I want is out of stock.", "Why are so many items out of stock?",
                              "Your store never has anything in stock."],
                             "I'm sorry to hear that. First, " + ASK_ID),
    "out_of_stock_one_item": (["The one item I want is out of stock.", "A specific product I need is sold out.",
                               "The item in my order is out of stock."],
                              "Sorry about that, let me see what I can do. To pull up your account, " + ASK_ID),
    "promo_code_invalid": (["My promo code says it is invalid.", "The discount code isn't working.",
                            "My coupon code gets rejected at checkout.", "Why won't my promo code work?"],
                           "I can look into the promo code. First, " + ASK_ID),
    "mistimed_billing": (["I was billed for something I never bought.", "There's a charge on my bill for an order I didn't make.",
                          "I sent an item back but my bill still hasn't been credited."],
                         "I'm sorry about the billing problem, let me look into it. First, " + ASK_ID),
    "shipping_status": (["I got an email about my shipment that seems wrong.",
                         "The shipping notification I received doesn't look right.",
                         "I want to check on my shipment."],
                        "I can check your shipment. To pull up your account, " + ASK_ID),
    "manage_shipping": (["I need to change the shipping address on my order.", "I ordered the wrong item and want to change my order.",
                         "Can I update where my package is being shipped?"],
                        "I can help update your shipment. First, " + ASK_ID),
    "missing_item": (["My package never arrived.", "I didn't receive my item.", "The order says delivered but I don't have it."],
                     "I'm sorry your item hasn't arrived. To pull up your account, " + ASK_ID),
    "shipping_cost": (["The shipping cost on my order is too high.", "Can you waive my shipping fee?",
                       "I want my shipping fee refunded."],
                      "I can look into the shipping fee. First, " + ASK_ID),
    "subscription_status": (["Is my premium subscription still active?", "Did my subscription payment go through?",
                             "How much do I owe on my subscription?", "When is my subscription renewal date?"],
                            "I can check your subscription. To pull up your account, " + ASK_ID),
    "manage_pay_bill": (["I want to pay my subscription bill.", "I'd like to renew my premium subscription.",
                         "Can I pay for my subscription now?"],
                        "I can help you pay your subscription. First, " + ASK_ID),
    "manage_extension": (["I can't pay my subscription on time, can I get an extension?",
                          "I need more time to pay for my premium membership.",
                          "Could you extend my subscription payment deadline?"],
                         "Let me check what options you have. To pull up your account, " + ASK_ID),
    "manage_dispute_bill": (["I was billed twice for my subscription.", "My subscription bill is wrong.",
                             "I think I was overcharged for my subscription."],
                            "I'm sorry about the billing issue, let me look into it. First, " + ASK_ID),
}
for sub, (qs, ans) in CLARIFY.items():
    add("clarify", sub, qs, "[CLARIFY] " + ans)

# =============================================================================== 2. FACT QA
F = lambda sub, qs, a: add("fact", sub, qs, "[ANSWER] " + a)  # noqa: E731

# ---- Account Access
F("recover_username", ["What information do I need to recover my username?",
                       "How do you verify me before giving me my username?",
                       "Which details are needed for username recovery?"],
  "The agent pulls up your account with your full name or account ID, then verifies your identity with 3 of these 4 items: full name, zip code, phone number and email address.")
F("recover_username", ["How are usernames formed?", "What format does a username follow?",
                       "If my name is John Smith, what would my username be?"],
  "A username is the first letter of the first name, the last name and the number 1. For example, John Smith becomes jsmith1.")
F("recover_username", ["What would the username be for Wendy Chesterfield?"],
  "It would be wchesterfield1: the first letter of the first name, the last name, then 1.")
F("recover_password", ["Can you tell me my current password?", "Can the agent look up my old password?",
                       "Will support read my password to me?"],
  "No. The agent cannot get your existing password, but can generate a new one for you.")
F("recover_password", ["What do I need to get a new password?", "How does the password reset work?",
                       "Which details are needed to make a new password?"],
  "After pulling up your account, the agent checks your username, then needs either your PIN number or the answer to your security question to generate a new password.")
F("recover_password", ["What if I don't know my username when resetting my password?"],
  "Then the agent first follows the Recover Username steps to find it, and continues with the password reset.")
F("reset_2fa", ["How does a two-factor reset work?", "How do you reset two-factor authentication?",
                "What happens when I ask to reset 2FA?"],
  "The agent sends a special reset code to your email address. You need to give the email on the account; then the agent also sends a link about security best practices.")
F("reset_2fa", ["What if I don't have access to my email for the two-factor reset?",
                "Can I reset 2FA without my email address?"],
  "Yes. Instead of the email address you can give your PIN number or answer the security question, and the reset code is sent to the email address saved on file.")
F("reset_2fa", ["What is the security question for my account?"],
  "The security question used in the guidelines is: What is your mother's maiden name?")

# ---- Troubleshoot Site
F("credit_card", ["My credit card keeps getting rejected on the website. What can I try?",
                  "The site says my credit card is invalid. How do I fix it?",
                  "Why does my card get declined at checkout and what should I do?"],
  "Try entering the card information again, log out of your account and log back in, and check the card's expiration date in case it is too old. If nothing works, the agent can place the purchase for you.")
F("credit_card", ["What does the agent need to make a purchase for me?",
                  "If you place the order for me, what information do you need?"],
  "The agent needs your credit card number, its expiration date, and the brand and type of the item, such as Guess Jeans.")
F("shopping_cart", ["Items won't stay in my shopping cart. What should I do?",
                    "My cart is not updating when I add something.",
                    "Why can't I add an item to my cart?"],
  "Refresh the page and add the item again, log out and log back in, and check whether the item is out of stock; that is shown at the bottom left of the product page. If nothing works, the agent can make the purchase for you.")
F("shopping_cart", ["Where can I see if a product is out of stock?"],
  "The stock status is listed at the bottom left-hand side of the product page.")
F("search_results", ["The search always says no results found. How do I fix it?",
                     "Search isn't working on the website.", "Every search I do returns nothing."],
  "Log out and log back in, clear your browser cookies, and try searching for a different product such as a Prada Handbag before searching for your item again. The agent will also report the problem to the website team.")
F("search_results", ["How do I clear my cookies?"],
  "Open the settings option in your browser and select More Details, where the cookies can be reset.")
F("slow_speed", ["The website is really slow. What can I do?", "The site is loading too slowly to use.",
                 "Why is acmecorp.com so slow for me?"],
  "Try visiting another website first: if every site is slow, the problem is your internet connection. Also close other browser tabs and programs running in the background, and log out and back in. The agent will report the slow speed to the website team.")
F("slow_speed", ["If all websites are slow, whose problem is it?"],
  "Then it is the customer's internet connection, not a problem with the AcmeCorp website.")

# ---- Manage Account
F("status_service_added", ["What happens if a service was added to my account by mistake?",
                           "If an unknown service was really added to my account, what will you do?"],
  "If the system confirms the error, the agent removes the extra service from your account and offers a refund of $40.")
F("status_service_added", ["How much is the refund for a service added by mistake?"],
  "The refund for a wrongly added service is $40.")
F("status_service_added", ["I got an email saying a service was added, but your system shows no error. What does that mean?",
                           "What if the email about an added service was wrong?"],
  "It means the email was a scam: if you check the sender, it is not from the official AcmeCorp marketing team.")
F("status_service_added", ["My spouse told me a service was added, but the system shows nothing. Should I worry?",
                           "What if I heard about the extra service in the news but the system shows no error?"],
  "No. You must have misheard or misunderstood; there is nothing to worry about and there are no extra charges.")
F("status_service_removed", ["What happens if my service really was removed by mistake?",
                             "If a service was wrongly removed, how do you fix it?"],
  "The agent reinstates the lost service on your account and then checks with you that it is working again.")
F("status_service_removed", ["My service was reinstated but I still don't see it. Why?"],
  "It may take a few minutes for the reinstated service to appear on your account.")
F("status_credit_missing", ["Where can account credit come from?", "What are the possible sources of account credit?"],
  "The guidelines list three origins for credits: a subscription refund, a previous purchase, or a promotional package.")
F("status_credit_missing", ["If my credit is really missing, how do you add it back?",
                            "What do you do when credit is missing from my account?"],
  "The agent asks how much credit to add (the standard amount is $40 if you don't know) and sends you a promo code for that amount.")
F("status_credit_missing", ["What is the standard amount of credit added back?"],
  "The standard amount is $40.")
F("status_credit_missing", ["The system says my credit is not missing, but I can't see it. Why?"],
  "Then it is a display problem in the customer interface. Log out and log back in; the credit shows up on the agent's side.")
F("manage_change_address", ["What do I need to verify my identity to change my address?",
                            "How many identity details do I need to update my address?"],
  "You must give 3 of these 7 items: zip code, telephone number, PIN number, username, password, email address, or the order ID of a previous purchase.")
F("manage_change_address", ["Do you need my full address when I change it?",
                            "What part of my address do you need for the change?"],
  "Only the street address is needed, both for the current and the new address, for example 7221 Florshim Drive.")
F("manage_change_name", ["What do I need to change the name on my account?", "How do I update my account name?"],
  "You must provide your current first and last name, and verify your identity with 3 of the 7 items: zip code, telephone number, PIN, username, password, email address or a previous order ID. Then give the new name.")
F("manage_change_phone", ["What format should my phone number be in?", "How should I write my new phone number?"],
  "Phone numbers use the format (XXX) XXX-XXXX.")
F("manage_change_phone", ["What do I need to change my phone number?"],
  "Your current phone number, 3 of the 7 identity items (zip code, telephone number, PIN, username, password, email address or a previous order ID), and the new number in the format (XXX) XXX-XXXX.")
F("manage_payment_method", ["Which payment methods can I have on my account?", "What payment options are accepted?",
                            "What are the valid payment methods?"],
  "The valid payment methods are credit card, debit card and PayPal.")
F("manage_payment_method", ["Can I pay with cash or bank transfer?"],
  "No. The documented payment methods are only credit card, debit card and PayPal.")

# ---- Order Issue
F("status_payment_method", ["Can I change the payment method on an order that is in transit?",
                            "My order was received but not shipped. Can I still change the payment method?"],
  "Yes. If the order status is Order Received or In Transit, the payment method can be changed immediately.")
F("status_payment_method", ["My order is out for delivery. Can I still change its payment method?",
                            "The order was already delivered. What happens if I change the payment method?"],
  "No, not for this order. When the item is Out for Delivery or Delivered, the new payment method only applies to future orders.")
F("shipping_status", ["What are the possible shipping statuses?", "Which shipping status options exist?",
                      "What stages can a shipment be in?"],
  "There are four shipping statuses: Order Received, In Transit, Out for Delivery and Delivered.")
F("status_quantity", ["My email shows two items but I was charged correctly for one. What happened?",
                      "What if the system shows the confirmation email was wrong about the quantity?"],
  "Then the email was wrong: you were only charged for one item, and nothing else needs to be done.")
F("status_quantity", ["The extra item is a company mistake and my order is still at Order Received. What happens?"],
  "The agent offers the refund right away and removes the unwanted item from your order.")
F("status_quantity", ["The extra item was a company mistake but it is already in transit. What do I do?"],
  "You need to wait until the item arrives and then call customer support back to start sending it back.")
F("manage_upgrade", ["How much does it cost to upgrade to overnight shipping?",
                     "What is the fee for overnight shipping?"],
  "Gold members have the fee waived. Guest, Bronze and Silver members pay $20, charged to the credit card on their account.")
F("manage_upgrade", ["My order is already in transit. Can I still upgrade to overnight?",
                     "Can I upgrade shipping after the item has shipped?"],
  "No. Once the item is In Transit, Out for Delivery or Delivered, the date cannot be changed, but it will likely arrive today or tomorrow anyway.")
F("manage_downgrade", ["Can I delay my delivery to next week if it hasn't shipped yet?"],
  "Yes. If the order has not shipped yet, the shipping date can be moved to next week with no problem.")
F("manage_downgrade", ["My order already shipped. Can I get anything back for choosing slower delivery?"],
  "Gold, Silver and Bronze members get a credit for the shipping fee added back to their account. Guest members cannot get anything once the item has shipped.")
F("manage_create", ["Can I add an item to my order before it ships?",
                    "My order is at Order Received. Can I add another item?"],
  "Yes. While the order is at Order Received, the agent can add the item directly by entering its brand and type, for example Calvin Klein Shirt.")
F("manage_create", ["My order already shipped. Can Gold or Silver members still add an item?"],
  "Yes. Gold and Silver members get the new item shipped as soon as possible in a separate order with no shipping fee.")
F("manage_cancel", ["I ordered a duplicate item and the order hasn't shipped. What happens?"],
  "The agent offers the refund immediately and removes the unwanted item from the order.")
F("manage_cancel", ["My duplicate item already shipped and I am a Guest. What can I do?"],
  "As a Guest you must wait until the item arrives and then call customer support back to start sending it back.")
F("manage_cancel", ["My duplicate item already shipped and I am a Bronze member. Can I get my money back now?"],
  "Yes. Bronze, Silver and Gold members can be offered a refund immediately; after the item arrives, call support back to get the shipping label for sending it back.")
F("status_mystery_fee", ["How does support decide whether an extra fee was the company's mistake?"],
  "The agent checks the system (Ask the Oracle). If it was a company mistake, the fee is removed. If not, it depends on your membership level.")
F("status_delivery_time", ["What should I do if I just want to confirm my delivery date?"],
  "Confirming a date belongs to the Shipping Issue status flow; changing the delivery time from morning to evening is the Order Issue delivery-time flow.")
F("order_issue", ["What do I need to give so you can look up my order?", "Which details verify my identity for an order?"],
  "Your full name, account ID and order ID, which the agent enters into Verify Identity.")

# ---- Purchase Dispute
F("bad_price", ["Why did the price of my item change?", "Why are prices different from day to day?"],
  "Prices can change and that is outside the agent's control: the pricing algorithms often change the price of an item dynamically.")
F("bad_price", ["I'm still unhappy about the price difference. Can you do anything?",
                "Is there any compensation for a price drop if I keep complaining?"],
  "Yes. If you are still unhappy, the agent can offer you a discount with a promo code.")
F("bad_price", ["What reasons can a price dispute have?"],
  "Two reasons are recorded: a competitor has a better price, or the price was lower yesterday.")
F("out_of_stock_general", ["What do you do when many items are out of stock?",
                           "Will anyone be told that items are out of stock?"],
  "The agent writes up a report for the Purchasing Department so they can do a better job, and can offer a promo code if you are still unhappy.")
F("out_of_stock_one_item", ["Can you back-order an item that is out of stock?",
                            "What happens if you back-order an item for me?"],
  "Yes. The agent can back-order it using the same credit card on file, and the item ships immediately when it becomes available.")
F("promo_code_invalid", ["How long do promotions last?", "How long is a promo code valid?",
                         "When do promo codes stop working?"],
  "Promotions typically last 7 days.")
F("promo_code_invalid", ["What is the difference between an invalid and an out-of-date promo code?"],
  "A code issued more than 7 days ago is out of date; a newer code that does not work is invalid. Both are handled the same way.")
F("promo_code_invalid", ["How much is the replacement promo code worth?", "What discount does the new promo code give?"],
  "The replacement promo code is for 20% off your next purchase.")
F("promo_code_invalid", ["Who can get a new promo code if the old one was used wrongly?",
                         "Which members qualify for a replacement promo code?"],
  "Gold, Silver and Bronze members all qualify for a new code. Guests do not, so the agent cannot help them in that case.")
F("mistimed_billing", ["I sent my item back 10 days ago and still have no credit. Will you process it?",
                       "It has been more than a week since I sent the item back and the credit is missing. What happens?"],
  "Yes. If you have waited more than 7 days, it counts as a valid error and the agent starts processing the credit.")
F("mistimed_billing", ["I sent my item back 3 days ago and the credit isn't there yet. Can you give it now?"],
  "Only for Gold members. If it has been less than 7 days, Gold members still get the credit; other members have to wait.")
F("mistimed_billing", ["I was billed for something I never bought and the system says it was my error. Can I still get credit?"],
  "Only if you are a Gold member. Gold members get the credit anyway; other members cannot be helped on this issue.")

# ---- Shipping Issue
F("shipping_status", ["If my shipping email was wrong, how can support fix my order?",
                      "What changes can support make when my shipment has a problem?"],
  "The agent can change the date, change the address, change the item or change the price.")
F("manage_shipping", ["What kinds of shipment changes can I request?"],
  "You can change the shipping address or change the order if you ordered the wrong item.")
F("missing_item", ["My package has been missing for 4 days. What should I do?",
                   "It's only been a few days and my item hasn't arrived. Is it lost?"],
  "Not yet. Within 7 days, a package is sometimes marked as delivered by mistake and can still take a couple of days to arrive.")
F("missing_item", ["My item has been missing for over a week. What will you do?",
                   "It has been 10 days and my package never came."],
  "After a week or longer, the agent ships a new order. You will need to give your full address and the brand and type of the product.")
F("shipping_cost", ["Can my shipping fee be waived if the order is still in transit?",
                    "My order was just received. Can you waive the shipping fee?"],
  "Yes. If the status is Order Received or In Transit, the agent can waive the shipping fee.")
F("shipping_cost", ["My order was already delivered. Can I get the shipping fee back?",
                    "The package is out for delivery. Can I get a shipping refund?"],
  "Only part of it. When the order is Out for Delivery or Delivered, the agent refunds the standard shipping amount of $8.")
F("shipping_cost", ["How much is standard shipping?"],
  "Standard shipping is $8.00.")

# ---- Subscription Inquiry
F("subscription_status", ["What can you tell me about my subscription after checking it?",
                          "What does a subscription status check show?"],
  "It shows whether the subscription is active or cancelled, the amount due and the due date. The agent can also send you an account login link so you can check it yourself.")
F("manage_pay_bill", ["How much is the standard premium subscription?", "What does the subscription cost per year?"],
  "The standard subscription is $99 a year.")
F("manage_pay_bill", ["I don't have my credit card number. Can I still pay my subscription?"],
  "Yes. If you don't have a card number, the agent uses the credit card already on your account.")
F("manage_extension", ["Who can get a subscription payment extension?", "Which members are allowed an extension?"],
  "Gold members can always get an extension. Silver members get one only if they are 1 day late. Bronze and Guest members cannot get an extension.")
F("manage_extension", ["If I can't get an extension, can I talk to a manager?"],
  "Yes. The agent takes your phone number and notifies the manager, who will call you back later.")
F("manage_dispute_bill", ["Who gets a refund when a subscription bill is wrong?",
                          "How is a disputed subscription bill handled for each membership level?"],
  "Gold members always get what they ask for. Silver members get a refund if the wrong amount is less than $10. For Bronze and Guest members it depends on what the system check says.")
F("manage_dispute_bill", ["I should have been charged $50 but the bill was $60. How much is refunded?"],
  "$10, the difference between the correct amount and the billed amount.")
F("manage_dispute_bill", ["I should have paid $40 but was billed $55. What is the refund?"],
  "$15, the difference between the correct amount and the billed amount.")

# ---- Storewide / single-item query structure (the FAQ article texts themselves are not in our documents)
F("storewide_query", ["What kinds of storewide questions can support answer?",
                      "Which categories does the storewide FAQ have?"],
  "The storewide FAQ has four categories: pricing, timing, membership and policies.")
F("single_item_query", ["Which clothing items have their own FAQ?", "What product FAQs are there?"],
  "There are FAQs for four items: jeans, shirts, boots and jackets.")
F("membership", ["What membership levels are there?", "What are the AcmeCorp membership tiers?",
                 "Which member levels exist?"],
  "There are four levels: Gold, Silver, Bronze and Guest.")
F("pricing", ["What pricing questions are covered in the FAQ?"],
  "The price of gift wrapping, the price of stitching your name, the price of overnight shipping, and whether a shopping cart qualifies for free shipping.")
F("policy", ["Which policies are covered in the storewide FAQ?"],
  "Canceling subscriptions, being late on a subscription payment, the refund policy and the return policy.")

# =============================================================================== 3. tier scenarios (yes/no)
TIERS = ["Gold", "Silver", "Bronze", "Guest"]
for t in TIERS:  # mystery fee (customer error)
    ok = t in ("Gold", "Silver")
    add("scenario", "status_mystery_fee",
        [f"I'm a {t} member and the system says the extra fee on my order was my error. Can it still be removed?"],
        "[ANSWER] " + ("Yes. Gold and Silver members always get extra fees removed." if ok else
                       f"No. For {t} members a customer error means the extra fee stays."))
for t in TIERS:  # overnight upgrade fee
    add("scenario", "manage_upgrade", [f"I'm a {t} member. Is the overnight shipping fee waived for me?"],
        "[ANSWER] " + ("Yes. Gold members have the overnight shipping fee waived." if t == "Gold" else
                       f"No. {t} members pay $20 for overnight shipping, charged to the card on the account."))
for t in TIERS:  # downgrade after shipping
    ok = t != "Guest"
    add("scenario", "manage_downgrade", [f"I'm {t} and my order already shipped. Can I get the shipping fee credited back?"],
        "[ANSWER] " + (f"Yes. {t} members get a credit for the shipping fee added back to their account." if ok else
                       "No. For Guest members nothing can be done once the order has shipped."))
for t in TIERS:  # promo replacement
    ok = t != "Guest"
    add("scenario", "promo_code_invalid", [f"I'm a {t} member and the system says my promo code mistake was mine. Can I still get a new code?"],
        "[ANSWER] " + (f"Yes. {t} members qualify for a new promo code for 20% off the next purchase." if ok else
                       "No. Guests do not qualify for a replacement code, so there is nothing the agent can do."))
for t in TIERS:  # extension, 1 day late
    ok = t in ("Gold", "Silver")
    add("scenario", "manage_extension", [f"I'm a {t} member and my subscription payment is 1 day late. Can I get an extension?"],
        "[ANSWER] " + ("Yes. Gold members can always get an extension." if t == "Gold" else
                       "Yes. Silver members get an extension when they are only 1 day late." if ok else
                       f"No. {t} members cannot get an extension."))
for t in TIERS:  # extension, 5 days late
    ok = t == "Gold"
    add("scenario", "manage_extension", [f"I'm {t} and I'm 5 days late on my subscription payment. Is an extension possible?"],
        "[ANSWER] " + ("Yes. Gold members can always get an extension." if ok else
                       "No. Silver members only get an extension if they are 1 day late." if t == "Silver" else
                       f"No. {t} members cannot get an extension."))
for t in TIERS:  # billing credit after < 7 days
    ok = t == "Gold"
    add("scenario", "mistimed_billing", [f"I'm a {t} member. I sent an item back 4 days ago and the credit is missing. Can you process it now?"],
        "[ANSWER] " + ("Yes. Gold members get the credit even before 7 days have passed." if ok else
                       f"No. Before 7 days only Gold members get the credit; as a {t} member you need to wait."))
for t in TIERS:  # duplicate item already shipped
    ok = t != "Guest"
    add("scenario", "manage_cancel", [f"I'm a {t} member. My duplicate item is already out for delivery. Can I get my money back now?"],
        "[ANSWER] " + (f"Yes. {t} members can be offered a refund immediately, and then call back for a shipping label after the item arrives." if ok else
                       "No. Guests must wait until the item arrives and then call support back to start sending it back."))
for t in TIERS:  # add item after shipping
    ok = t in ("Gold", "Silver")
    add("scenario", "manage_create", [f"I'm a {t} member and my order is already in transit. Can you ship the extra item separately for free?"],
        "[ANSWER] " + (f"Yes. {t} members get the new item shipped in a separate order with no shipping fee." if ok else
                       f"No. The free separate shipment is special treatment for Gold and Silver members only."))
for amt, t, ok in [(5, "Silver", True), (15, "Silver", False), (25, "Gold", True), (8, "Silver", True), (12, "Silver", False)]:
    add("scenario", "manage_dispute_bill",
        [f"I'm a {t} member and my subscription bill is ${amt} too high. Do I get a refund?"],
        "[ANSWER] " + ("Yes. Gold members always get what they ask for." if t == "Gold" else
                       f"Yes. Silver members get a refund when the wrong amount is less than $10." if ok else
                       f"No. Silver members only get a refund when the wrong amount is less than $10, and ${amt} is not."))
for days, ok in [(2, False), (5, False), (8, True), (12, True)]:
    add("scenario", "missing_item", [f"My package has been missing for {days} days. Will you send a new one?"],
        "[ANSWER] " + (f"Yes. After a week or longer the agent ships a new order to your full address." if ok else
                       f"Not yet. After {days} days the package may have been marked delivered by mistake and can still take a couple of days."))
for days, ok in [(3, False), (10, False), (6, False), (14, False), (8, False)]:
    pass  # (kept intentionally empty: promo-code age handled in multi-turn below)
for status, ok in [("Order Received", True), ("In Transit", True), ("Out for Delivery", False), ("Delivered", False)]:
    add("scenario", "shipping_cost", [f"My order status is {status}. Can you waive the whole shipping fee?"],
        "[ANSWER] " + ("Yes. The shipping fee can be waived while the order is Order Received or In Transit." if ok else
                       f"No. At {status} the agent can only refund part of it, the standard $8 shipping amount."))
for status, ok in [("Order Received", True), ("In Transit", False), ("Out for Delivery", False), ("Delivered", False)]:
    add("scenario", "manage_upgrade", [f"My order shows {status}. Can it still be upgraded to overnight shipping?"],
        "[ANSWER] " + ("Yes. The order has not shipped yet, so the shipping can be upgraded." if ok else
                       f"No. At {status} the item has already shipped and the date cannot be changed, but it will likely arrive today or tomorrow."))

# =============================================================================== 4. multi-turn
M = lambda sub, hist, qs, a: add("multiturn", sub, qs, a, hist)  # noqa: E731
M("promo_code_invalid", [("user", "My promo code isn't working."), ("assistant", "When was the promo code issued?")],
  ["It was issued 10 days ago. Why won't it work?"],
  "[ANSWER] Promotions typically last 7 days, so a code from 10 days ago is out of date. If you are a Gold, Silver or Bronze member, the agent can generate a new code for you.")
M("promo_code_invalid", [("user", "My discount code is rejected."), ("assistant", "How long ago did you get the code?")],
  ["Just 2 days ago. What now?"],
  "[CLARIFY] A 2-day-old code is still within the usual 7 days, so I will check the system for an error. First, " + ASK_ID)
M("promo_code_invalid", [("user", "My promo code is invalid."), ("assistant", "What is your membership level?")],
  ["I'm a Guest. Can you give me another code?"],
  "[ANSWER] No. If the system shows the code was used incorrectly, Guests do not qualify for a replacement code. I'm sorry I can't do more.")
M("promo_code_invalid", [("user", "My coupon doesn't work."), ("assistant", "What is your membership level?")],
  ["I'm Silver."],
  "[ANSWER] Silver members qualify, so even if the old code is not valid the agent will generate a new promo code for you, worth 20% off your next purchase.")
M("manage_extension", [("user", "I need more time to pay my subscription."), ("assistant", "What is your membership level?")],
  ["I'm Bronze. Can I get an extension?"],
  "[ANSWER] No. Bronze members cannot get an extension. I can take your phone number and ask a manager to call you back.")
M("manage_extension", [("user", "I can't pay my subscription yet."), ("assistant", "What is your membership level, and how late is the payment?")],
  ["Silver, and I'm one day late."],
  "[ANSWER] Yes. Silver members get an extension when they are only 1 day late, so the agent can extend your subscription.")
M("manage_extension", [("user", "I need a payment extension."), ("assistant", "What is your membership level?")],
  ["Gold."],
  "[ANSWER] Gold members can always get an extension, so your subscription can be extended.")
M("manage_dispute_bill", [("user", "My subscription bill is wrong."), ("assistant", "What should you have been charged, and what were you billed?")],
  ["I should have paid $30 but I was charged $45."],
  "[ANSWER] The refund amount would be $15, the difference between the two. Whether you get it depends on your membership level, so what is your membership level?")
M("manage_dispute_bill", [("user", "I was overcharged on my subscription."), ("assistant", "What is your membership level?")],
  ["Bronze. Do I get my money back?"],
  "[ANSWER] For Bronze members it depends on the system check: if the system confirms the error you get the refund, and if not, you don't.")
M("missing_item", [("user", "My order hasn't arrived."), ("assistant", "How long have you been waiting?")],
  ["About 9 days now."],
  "[ANSWER] Since you have waited more than a week, the agent will ship a new order. Please give your full address and the brand and type of the product.")
M("missing_item", [("user", "Where is my package?"), ("assistant", "How many days have you been waiting?")],
  ["Three days."],
  "[ANSWER] Within 7 days the package may have been marked as delivered by mistake and can still take a couple of days to arrive, so please wait a little longer.")
M("shipping_cost", [("user", "Your shipping fee is too high."), ("assistant", "What is the current shipping status of your order?")],
  ["It says In Transit."],
  "[ANSWER] Because the order is In Transit, the agent can waive the shipping fee on it.")
M("shipping_cost", [("user", "I want my shipping fee back."), ("assistant", "What is the current shipping status of your order?")],
  ["It was delivered yesterday."],
  "[ANSWER] For a delivered order the agent can refund part of the shipping fee: the standard shipping amount of $8.")
M("manage_upgrade", [("user", "I want overnight shipping."), ("assistant", "What is the shipping status of your order?")],
  ["Order Received."],
  "[CLARIFY] The order has not shipped yet, so it can be upgraded. What is your membership level? Gold members have the fee waived; others pay $20.")
M("manage_upgrade", [("user", "Please upgrade me to overnight delivery."), ("assistant", "What is your membership level?")],
  ["I'm Silver. How much will it cost?"],
  "[ANSWER] Silver members pay $20 for the overnight upgrade, charged to the credit card on the account.")
M("manage_downgrade", [("user", "I want slower shipping to save money."), ("assistant", "Has your order shipped yet?")],
  ["Yes, it's out for delivery. I'm a Guest."],
  "[ANSWER] I'm sorry, for Guest members there is nothing that can be done once the order has shipped.")
M("manage_downgrade", [("user", "Can I switch to next-week delivery?"), ("assistant", "Has your order shipped yet?")],
  ["No, it's still at Order Received."],
  "[ANSWER] Yes. Since it has not shipped, the delivery can be moved to next week with no problem.")
M("status_payment_method", [("user", "I want to change the payment method on my order."), ("assistant", "What is the shipping status of the order?")],
  ["It's out for delivery."],
  "[ANSWER] Because the item is Out for Delivery, the new payment method will only apply to future orders. Would you still like to change it?")
M("status_payment_method", [("user", "My order used the wrong card."), ("assistant", "What is the shipping status?")],
  ["Order Received."],
  "[ANSWER] Then the payment method can be changed immediately to credit card, debit card or PayPal. Which one would you like?")
M("status_mystery_fee", [("user", "There's a strange fee on my order."), ("assistant", "The system shows it was not a company error. What is your membership level?")],
  ["I'm Bronze."],
  "[ANSWER] I'm sorry, for Bronze members a customer error means the extra fee stays, so it cannot be removed.")
M("status_mystery_fee", [("user", "Why was I charged an extra fee?"), ("assistant", "The system shows it was not a company error. What is your membership level?")],
  ["Gold."],
  "[ANSWER] Gold members always get extra fees removed, so the agent will remove the fee. How much were you overcharged?")
M("mistimed_billing", [("user", "I sent my item back but haven't received credit on my bill."), ("assistant", "How many days have you been waiting?")],
  ["Twelve days."],
  "[ANSWER] After more than 7 days this counts as a valid error, so the agent will start processing the credit.")
M("mistimed_billing", [("user", "My bill still shows a charge for an item I sent back."), ("assistant", "How many days have you been waiting?")],
  ["Two days. I'm a Silver member."],
  "[ANSWER] Before 7 days only Gold members get the credit, so as a Silver member you need to wait a little longer.")
M("status_credit_missing", [("user", "Credit is missing from my account."), ("assistant", "Where did the credit come from?")],
  ["From a promotional package. I don't know how much it was."],
  "[ANSWER] If the system confirms the credit is missing, the agent adds the standard amount of $40 and sends you a promo code for it.")
M("status_service_added", [("user", "A service was added to my account."), ("assistant", "How did you hear about it?")],
  ["I got an email about it."],
  "[ANSWER] I'll check the system. If it shows no error, the email was a scam that did not come from the official AcmeCorp marketing team. If it is a real error, the service is removed and you get a $40 refund.")
M("recover_username", [("user", "I forgot my username."), ("assistant", "Could you give me your full name or account ID?")],
  ["My name is Maria Lopez."],
  "[CLARIFY] Thank you. To verify your identity, please give me 3 of these 4 items: your full name, zip code, phone number or email address.")
M("recover_password", [("user", "I can't remember my password."), ("assistant", "What is your username?")],
  ["It's mlopez1. Can you tell me my old password?"],
  "[ANSWER] I can't get your existing password, but I can generate a new one. Please give me your PIN number or the answer to your security question.")
M("reset_2fa", [("user", "I need to reset two-factor authentication."), ("assistant", "What email address is on the account?")],
  ["I don't have access to that email anymore."],
  "[CLARIFY] That's okay. Please give me your PIN number instead, or answer the security question: What is your mother's maiden name?")
M("manage_change_address", [("user", "I need to update my address."), ("assistant", "What is your current street address?")],
  ["12 Oak Street. Is that all you need?"],
  "[CLARIFY] Thanks. I also need to verify your identity with 3 of these 7 items: zip code, telephone number, PIN, username, password, email address or the order ID of a previous purchase.")
M("manage_change_phone", [("user", "I want to change my phone number."), ("assistant", "What is your current phone number?")],
  ["555 123 4567."],
  "[CLARIFY] Thank you. Please verify your identity with 3 of the 7 items (zip code, telephone number, PIN, username, password, email address or a previous order ID), then give me the new number in the format (XXX) XXX-XXXX.")
M("credit_card", [("user", "My credit card keeps getting rejected."), ("assistant", "Please try entering the card information again.")],
  ["I tried, it still doesn't work."],
  "[ANSWER] Please log out of your account and log back in, and check the card's expiration date in case it is too old. If it still fails, I can place the purchase for you.")
M("slow_speed", [("user", "The website is very slow."), ("assistant", "Could you try visiting another website?")],
  ["Other websites are slow too."],
  "[ANSWER] If every website is slow, the problem is your internet connection rather than the AcmeCorp website.")
M("search_results", [("user", "Search shows no results."), ("assistant", "Please try clearing your cookies.")],
  ["How do I do that?"],
  "[ANSWER] Open your browser's settings option and select More Details; the cookies can be reset there.")
M("out_of_stock_one_item", [("user", "The item I want is out of stock."), ("assistant", "I've reported it to the Purchasing Department.")],
  ["That doesn't help me. Can you do anything else?"],
  "[ANSWER] I can back-order the item for you with the same credit card on file, and it will ship immediately when it becomes available.")
M("bad_price", [("user", "Another store sells this cheaper."), ("assistant", "Prices change dynamically and that is outside our control.")],
  ["I'm still not happy about it."],
  "[ANSWER] I understand. Since you are still unhappy, I can offer you a discount with a promo code.")
M("manage_pay_bill", [("user", "I want to pay my subscription."), ("assistant", "How much would you like to pay?")],
  ["I'm not sure how much I owe."],
  "[ANSWER] The agent can check the amount due with Subscription Status and tell you. The standard subscription is $99 a year.")

# =============================================================================== 5. ABSTAIN
REFUSE = "[ABSTAIN] I don't have that information. "
OUT_OF_SCOPE = [
    "What's the weather going to be like tomorrow?", "Who won the football game last night?",
    "Can you recommend a good restaurant near me?", "Write me a poem about the ocean.",
    "What is the capital of Australia?", "Can you help me with my math homework?",
    "How do I fix a Python error in my code?", "What medicine should I take for a headache?",
    "Can you give me legal advice about my lease?", "Which stocks should I invest in?",
    "Can you order me a pizza?", "What is the exchange rate between dollars and euros?",
    "How tall is Mount Everest?", "Translate this sentence into French for me.",
    "Can you reserve a hotel room for me?", "What's a good movie to watch tonight?",
    "How do I change the oil in my car?", "Who is the CEO of AcmeCorp?",
    "What is AcmeCorp's annual revenue?", "How many employees does AcmeCorp have?",
    "When was AcmeCorp founded?", "Can you call my bank for me?",
    "Can you help me file my taxes?", "What is the best phone to buy this year?",
    "Can you rent me a car for the weekend?", "Tell me a joke about cats.",
    "What time is it in Tokyo right now?", "How do I bake chocolate chip cookies?",
]
for q in OUT_OF_SCOPE:
    add("abstain_out_of_scope", "out_of_scope", [q],
        REFUSE + "That is outside what AcmeCorp customer support covers. I can help with your orders, account, shipping, subscription or purchases.")

# FAQ article contents are not part of our documents (only the list of questions is).
FAQ_UNKNOWN = [
    ("boots", "How do I remove a paint stain from my boots?"), ("boots", "Are your boots waterproof?"),
    ("boots", "How wide are the boots?"), ("boots", "What color are the laces on the boots?"),
    ("boots", "How long does it take to wear in the boots?"), ("boots", "How do I get gum off my boots?"),
    ("shirt", "How should I wash this shirt?"), ("shirt", "Does the shirt shrink?"),
    ("shirt", "What material is the shirt made of?"), ("shirt", "How long are the sleeves on the shirt?"),
    ("shirt", "Do the shirts come with brown or black buttons?"),
    ("jeans", "How do I get a grass stain out of my jeans?"), ("jeans", "How much does it cost to get jeans tailored?"),
    ("jeans", "What is the leg length of the jeans?"), ("jeans", "Do you have jeans with ripped holes?"),
    ("jacket", "How do I remove a wine stain from my jacket?"), ("jacket", "Is the jacket dry-clean only?"),
    ("jacket", "How do I detach the hood of the jacket?"), ("jacket", "Is the jacket warm enough for winter?"),
    ("pricing", "How much does gift wrapping cost?"), ("pricing", "What is the price of stitching my name on an item?"),
    ("pricing", "Does my shopping cart qualify for free shipping?"),
    ("timing", "When does the spring collection come out?"), ("timing", "What time does the local store open?"),
    ("timing", "When does the annual sale begin?"),
    ("membership", "How do I qualify for premium membership?"), ("membership", "How long does a membership last?"),
    ("policy", "What is the policy for canceling a subscription?"),
]
for sub, q in FAQ_UNKNOWN:
    add("abstain_undocumented", sub, [q],
        REFUSE + f"The guidelines only say this is answered in the {sub} FAQ, and that article's content is not in my documents.")

LIVE_DATA = [
    "What is my account ID?", "What is the PIN number on my account?", "What is my current account balance?",
    "Which items are in my last order?", "What address do you have on file for me?",
    "What is the order ID of my last purchase?", "Which credit card is saved on my account?",
    "How many orders have I placed this year?", "What is the email address on my account?",
    "What membership level am I?",
]
for q in LIVE_DATA:
    add("abstain_live_data", "live_data", [q],
        REFUSE + "I can't look up your personal account records here. An agent would first pull up your account with your full name or account ID and verify your identity.")


def build(seed: int = 42) -> list[dict]:
    rows = []
    for i, p in enumerate(PAIRS):
        rows.append({"id": f"sft-{i + 1:04d}", **p, "source": "written from data/guidelines.json; hand check: data/sft_review_sample.csv"})
    random.Random(seed).shuffle(rows)
    return rows


def write_review_sample(rows: list[dict], path: str, n: int = 60, seed: int = 42):
    """A stratified random sample for the hand check (the brief asks for at least 50)."""
    import csv

    from .common import resolve

    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    rng = random.Random(seed)
    sample = []
    for cat, items in sorted(by_cat.items()):
        k = max(2, round(n * len(items) / len(rows)))
        sample += rng.sample(items, min(k, len(items)))
    sample = sorted(sample[:max(n, 50)], key=lambda r: r["id"])
    with open(resolve(path), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "category", "subflow", "history", "question", "answer",
                                          "ok", "reviewer", "note"])
        w.writeheader()
        for r in sample:
            w.writerow({"id": r["id"], "category": r["category"], "subflow": r["subflow"],
                        "history": " || ".join(f"{t['role']}: {t['text']}" for t in r["history"]),
                        "question": r["question"], "answer": r["answer"], "ok": "", "reviewer": "", "note": ""})
    return len(sample)


if __name__ == "__main__":
    from .common import load_config, write_jsonl

    cfg = load_config()
    rows = build(cfg["seed"])
    write_jsonl(cfg["sft"]["pairs"], rows)
    n = write_review_sample(rows, "data/sft_review_sample.csv", seed=cfg["seed"])
    print(f"Wrote {len(rows)} pairs to {cfg['sft']['pairs']} and {n} pairs to data/sft_review_sample.csv for the hand check")
