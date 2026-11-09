"""The one chat template used by every model and every stage (the ABCD dialogue format):

    Customer support conversation.
    Customer: <earlier customer turn>          (only for multi-turn cases)
    Agent: <earlier agent turn>
    Customer: <question>
    Agent: <answer><eos>

The SFT target is the answer plus the end-of-sequence token; prompt tokens get the label -100,
so the loss is computed on the answer only.
"""
HEADER = "Customer support conversation."
STOP_STRINGS = ["\nCustomer:", "\nAgent:", "\nAction:"]


def parse_history(raw):
    """'Customer: ...\\nAssistant: ...' (test CSV) -> [{'role': 'user'|'assistant', 'text': ...}]"""
    turns = []
    for line in str(raw or "").splitlines():
        line = line.strip()
        if line.lower().startswith("customer:"):
            turns.append({"role": "user", "text": line.split(":", 1)[1].strip()})
        elif line.lower().startswith(("assistant:", "agent:")):
            turns.append({"role": "assistant", "text": line.split(":", 1)[1].strip()})
        elif line and turns:
            turns[-1]["text"] += " " + line
    return turns


def build_prompt(history, question):
    lines = [HEADER] + [("Customer: " if t["role"] == "user" else "Agent: ") + t["text"] for t in history]
    return "\n".join(lines + [f"Customer: {question}", "Agent:"])


def encode_pair(p, tok, max_len):
    """prompt + answer + <eos>; labels are -100 on the prompt (loss on answer tokens only)."""
    prompt = build_prompt(p["history"], p["question"])
    p_ids = tok(prompt)["input_ids"]
    a_ids = tok(" " + p["answer"].strip(), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    p_ids = p_ids[-(max_len - len(a_ids)):]                      # keep the end of a long prompt
    return {"input_ids": p_ids + a_ids, "labels": [-100] * len(p_ids) + a_ids}
