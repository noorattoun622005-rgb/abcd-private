"""AcmeCorp support handbook: one plain-language summary per subflow of data/guidelines.json.

Written for this project directly from the guidelines and checked against them.
It is used twice:
  * as extra documents in the continued-pretraining corpus (train split only), so the policies
    appear in clean prose and not only inside role-play instructions;
  * as SFT pairs of the form "Summarise the support guideline for <subflow>" -> summary,
    so every one of the 55 subflows is covered by the SFT data.
The role-play instructions in the guidelines that tell agents to invent things ("make up a reason",
"pick a refund status") are deliberately NOT restated.
"""

ASK_ID = "the agent first asks for the customer's full name or account ID to pull up the account"

HANDBOOK = {
    # ------------------------------------------------------------------ Account Access
    "recover_username": ("Recover Username",
        f"To recover a username, {ASK_ID}, then verifies identity with 3 of these 4 items: full name, zip code, "
        "phone number and email address. Usernames are the first letter of the first name, the last name and the "
        "number 1, for example John Smith becomes jsmith1."),
    "recover_password": ("Recover Password",
        f"For a forgotten password, {ASK_ID} and the username (if the username is unknown, the Recover Username "
        "steps come first). The agent cannot see the old password but can generate a new one after the customer "
        "gives either the PIN number or the answer to the security question."),
    "reset_2fa": ("Reset Two-Factor Authentication",
        f"To reset two-factor authentication, {ASK_ID}. A reset code is sent to the customer's email address. If the "
        "customer cannot use the email, they give their PIN number or answer the security question (mother's maiden "
        "name) and the code goes to the email on file. The agent also sends a link about security best practices."),
    # ------------------------------------------------------------------ Manage Account
    "status_service_added": ("Service Added (unknown service on the account)",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, records how the customer heard about it (email, spouse or news) and checks "
        "the system. If there is no error: an email was a scam not sent by the official AcmeCorp marketing team, and "
        "a spouse or the news was misheard or misunderstood, with no extra charges. If the error is real, the service "
        "is removed and a $40 refund is offered."),
    "status_service_removed": ("Service Removed (service lost by mistake)",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, records how the customer heard about it and checks the system. If the "
        "customer is right, the service is reinstated and the agent checks that it works; it may take a few minutes "
        "to appear. If not, the customer misheard or the email was a scam."),
    "status_shipping_question": ("Free International Shipping Question",
        f"For questions about free shipping to or from another country, {ASK_ID}, checks the account in the system "
        "and sends the customer a link to the subscription FAQ page."),
    "status_credit_missing": ("Missing Credit",
        f"{ASK_ID[0].upper() + ASK_ID[1:]} and asks where the credit came from: a subscription refund, a previous "
        "purchase or a promotional package. If the system shows the credit is not missing, it is a display problem "
        "and the customer should log out and back in. If it is missing, the agent adds it with a promo code; the "
        "standard amount is $40 when the customer does not know."),
    "manage_change_address": ("Change Address",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, asks for the current street address (street part only), verifies identity "
        "with 3 of 7 items (zip code, phone number, PIN, username, password, email address, order ID of a previous "
        "purchase) and then enters the new street address."),
    "manage_change_name": ("Change Name",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, asks for the current first and last name, verifies identity with 3 of 7 "
        "items (zip code, phone number, PIN, username, password, email address, previous order ID) and updates the "
        "account with the new name."),
    "manage_change_phone": ("Change Phone Number",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, asks for the current phone number in the format (XXX) XXX-XXXX, verifies "
        "identity with 3 of 7 items and then saves the new number in the same format."),
    "manage_payment_method": ("Change Account Payment Method",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, asks for the current payment method, verifies identity with 3 of 7 items "
        "and sets the new method. The valid payment methods are credit card, debit card and PayPal."),
    # ------------------------------------------------------------------ Order Issue
    "status_mystery_fee": ("Unexpected Fee on an Order",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies the order with full name, account ID and order ID, and checks "
        "whether the company made an error. A company error means the fee is removed. Otherwise Gold and Silver "
        "members still always get the fee removed, while for Bronze and Guest members the fee stays."),
    "status_delivery_time": ("Delivery Time Change",
        f"When a customer wants the delivery moved from morning to evening, {ASK_ID}, verifies the order (full name, "
        "account ID, order ID), checks the system for an error and the shipping status, then updates the order. "
        "Simply confirming a date belongs to the shipping status flow."),
    "status_payment_method": ("Wrong Payment Method on an Order",
        f"{ASK_ID[0].upper() + ASK_ID[1:]} and verifies the order. If the order is Order Received or In Transit, the "
        "payment method can be changed immediately to credit card, debit card or PayPal. If it is Out for Delivery "
        "or Delivered, the new method only applies to future orders."),
    "status_quantity": ("Wrong Quantity in the Confirmation Email",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies the order and checks the system. If the email was wrong, the "
        "customer was only charged for one item. If it was a company mistake and the order is Order Received, the "
        "refund is offered and the extra item removed; once shipped, the customer waits for delivery and calls back."),
    "manage_upgrade": ("Upgrade to Overnight Shipping",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies the order and asks for the shipping status. Only an order at "
        "Order Received can be upgraded; once shipped it will likely arrive today or tomorrow anyway. Gold members "
        "have the fee waived; Silver, Bronze and Guest members pay $20 on the card on the account."),
    "manage_downgrade": ("Downgrade to Next-Week Delivery",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies the order and checks the shipping status. If not shipped, the "
        "delivery can move to next week. If already shipped, Gold, Silver and Bronze members get the shipping fee "
        "credited back, and Guest members cannot be helped."),
    "manage_create": ("Add an Item to an Existing Order",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies the order and checks the shipping status. At Order Received the "
        "item (brand and type, e.g. Guess Jeans) is added directly. If already shipped, Gold and Silver members get "
        "the new item in a separate order with no shipping fee."),
    "manage_cancel": ("Accidental Duplicate Item",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies the order and checks the shipping status. At Order Received the "
        "refund is offered and the duplicate removed. If shipped, Bronze, Silver and Gold members get a refund right "
        "away and call back later for a shipping label; Guests wait for delivery and call back."),
    # ------------------------------------------------------------------ Product Defect
    "refund_initiate": ("Initiate Refund",
        "The agent asks for the refund reason and for the customer's full name or account ID to pull up the account. "
        "The purchase is validated with username, email address "
        "and order ID. The customer picks a refund method: gift card, add value to the account, paper check or credit "
        "card; a gift card or paper check needs the full mailing address, the others the account ID. The agent then "
        "offers the refund amount, which defaults to $50 if the customer does not know it."),
    "refund_update": ("Update Refund (add an item to an existing refund)",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, validates the purchase with username, email address and order ID, and "
        "records the brand and type of the new item. The new total refund is the previous refund amount plus the "
        "price of the new item."),
    "refund_status": ("Refund Status",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, then validates the purchase by collecting the username, then the email "
        "address, then the order ID. Only a verified record can show the status, which is one of: not started, in "
        "progress or complete. If the customer is unhappy with the status, the agent escalates it to a manager."),
    "return_item": ("Returns (stained item, wrong color or wrong size)",
        f"All three return reasons follow the same steps. {ASK_ID[0].upper() + ASK_ID[1:]} and validates the purchase "
        "(username, email address, order ID). Eligibility depends on membership: Gold members get unlimited returns; "
        "Silver within 6 months, or with a receipt, or in the original packaging; Bronze within 90 days, or with a "
        "receipt, or in the original packaging; Guest within 30 days or with a receipt. If eligible, the agent takes "
        "the full address for the shipping label and the customer chooses by mail, in store or drop-off center."),
    # ------------------------------------------------------------------ Purchase Dispute
    "bad_price": ("Bad Price (competitor cheaper or lower price yesterday)",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, records whether the reason is a competitor or yesterday's price, and "
        "verifies the order ID. The agent explains that pricing algorithms change prices dynamically, outside the "
        "agent's control. If the customer keeps pushing, the agent offers a promo code."),
    "out_of_stock_general": ("Many Items Out of Stock",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, writes a report for the Purchasing Department so they can do a better "
        "job, and offers a promo code if the customer is still unhappy."),
    "out_of_stock_one_item": ("One Item Out of Stock",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, records the item, reports it to the Purchasing Department and can "
        "back-order it with the card on file; it ships as soon as it is available."),
    "promo_code": ("Promo Code Invalid or Out of Date",
        f"{ASK_ID[0].upper() + ASK_ID[1:]} and checks the code. Promotions typically last 7 days, so an older code is "
        "out of date. If the code should work, the problem is fixed; if the customer used it wrongly, Gold, Silver "
        "and Bronze members get a new code for 20% off their next purchase, while Guests cannot be helped."),
    "mistimed_billing": ("Mistimed Billing (sent back or never bought)",
        f"{ASK_ID[0].upper() + ASK_ID[1:]} and validates the purchase. For an item already sent back, waiting more "
        "than 7 days is a valid error and the credit is processed; before 7 days only Gold members get it. For an "
        "item never bought, if the system says it was the customer's error, only Gold members get the credit."),
    # ------------------------------------------------------------------ Shipping Issue
    "status": ("Shipping Status Question",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies the order and checks the system. The shipping statuses are Order "
        "Received, In Transit, Out for Delivery and Delivered. To fix a problem the agent can change the date, the "
        "address, the item or the price."),
    "manage": ("Change a Shipment (address or wrong item)",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, checks the shipping status, validates the purchase and then updates the "
        "order with the new shipping address or the correct item."),
    "missing": ("Missing Package",
        f"{ASK_ID[0].upper() + ASK_ID[1:]} and validates the purchase. Within 7 days a package can be marked "
        "delivered by mistake and may still arrive in a couple of days. After a week or more the agent ships a new "
        "order using the full address and the brand and type of the product."),
    "cost": ("Shipping Cost Complaint",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, validates the purchase and checks the shipping status. At Order Received "
        "or In Transit the shipping fee can be waived. At Out for Delivery or Delivered only the standard shipping "
        "amount of $8 is refunded."),
    # ------------------------------------------------------------------ Single-item and storewide FAQ
    "single_item_faq": ("Product Questions (boots, shirts, jeans, jackets)",
        "Questions about a specific product are answered from that product's FAQ page: there are FAQs for boots, "
        "shirts, jeans and jackets. The agent searches the FAQ and selects the matching answer. The article texts "
        "themselves are not part of our guideline documents."),
    "storewide_faq": ("Storewide Questions (pricing, membership, timing, policies)",
        "General store questions are answered from the storewide FAQ, which has four categories: pricing, membership, "
        "timing and policies. The agent searches the FAQ and selects the matching answer; the article texts are not "
        "part of our guideline documents."),
    # ------------------------------------------------------------------ Subscription Inquiry
    "subscription_status": ("Subscription Status, Amount Due and Due Date",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies identity and checks the subscription: active or cancelled, the "
        "amount due and the due date. The agent can also send a link so the customer can log in and check it."),
    "manage_pay_bill": ("Pay a Subscription Bill",
        f"{ASK_ID[0].upper() + ASK_ID[1:]}, verifies identity, checks the amount due and takes the payment. The "
        "standard subscription is $99 a year; without a card number the card on the account is used."),
    "manage_extension": ("Subscription Payment Extension",
        f"{ASK_ID[0].upper() + ASK_ID[1:]} and checks the membership level. Gold members can always get an extension, "
        "Silver members only when 1 day late, and Bronze and Guest members cannot; the agent can then pass their "
        "phone number to a manager who calls back."),
    "manage_dispute_bill": ("Dispute a Subscription Bill",
        f"{ASK_ID[0].upper() + ASK_ID[1:]} and checks the membership level. Gold members always get what they ask for, "
        "Silver members get a refund when the wrong amount is under $10, and for Bronze and Guest members it depends "
        "on the system check. The refund is the difference between the correct and the billed amount."),
    # ------------------------------------------------------------------ Troubleshoot Site
    "credit_card": ("Credit Card Rejected on the Website",
        "The customer can re-enter the card details, log out and back in, and check the card's expiration date. If "
        "nothing works, the agent can place the purchase using the card number, expiration date and the item's brand "
        "and type."),
    "shopping_cart": ("Shopping Cart Not Updating",
        "The customer can refresh the page and add the item again, log out and back in, and check whether the item is "
        "out of stock (shown at the bottom left of the product page). If nothing works, the agent can place the "
        "purchase for them."),
    "search_results": ("Search Not Working",
        "The customer can log out and back in, clear the browser cookies (browser settings, More Details) and try a "
        "different search such as Prada Handbag before the original item. The agent reports the problem to the "
        "website team and can place the purchase if needed."),
    "slow_speed": ("Website Too Slow",
        "The agent reports the slowdown to the website team. The customer can log out and back in, close other tabs "
        "and programs, and try another website: if every site is slow, it is their internet connection, not "
        "AcmeCorp."),
}

QUESTION_TEMPLATES = [
    "Summarize the AcmeCorp support guideline for: {name}.",
    "How does AcmeCorp handle this case: {name}?",
]
