"""Model loading, the one generation function used for every model, packing and perplexity."""
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from .config import DEVICE, ROOT
from .template import STOP_STRINGS, build_prompt


def resolve(cfg, name):
    """Model name in config.yaml -> local checkpoint folder if it exists, else the Hugging Face Hub repo."""
    m = cfg["models"][name]
    if isinstance(m, str):
        return m
    local = Path(m["local"]) if Path(m["local"]).is_absolute() else cfg["work"] / m["local"]
    if local.exists():
        return str(local)
    if (ROOT / m["local"]).exists():
        return str(ROOT / m["local"])
    return f"{cfg['hub_user']}/{m['hub']}"


def load_tokenizer(name_or_path):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(name_or_path):
    m = AutoModelForCausalLM.from_pretrained(name_or_path)
    return m.float().to(DEVICE)        # train and evaluate in float32 (fp16 autocast during training)


class StopOnStrings(StoppingCriteria):
    def __init__(self, tok, start):
        self.tok, self.start = tok, start

    def __call__(self, input_ids, scores, **kw):
        return any(s in self.tok.decode(input_ids[0, self.start:]) for s in STOP_STRINGS)


def clean_answer(text):
    for s in STOP_STRINGS:
        text = text.split(s)[0]
    return text.strip()


@torch.no_grad()
def generate_answer(model, tok, history, question, gen):
    """THE function used to call every model (base, continued-pretrained, SFT): greedy, same stop rule."""
    ids = tok(build_prompt(history, question), return_tensors="pt")["input_ids"].to(model.device)
    t0 = time.perf_counter()
    out = model.generate(ids, attention_mask=torch.ones_like(ids), do_sample=False, num_beams=1,
                         max_new_tokens=gen["max_new_tokens"], repetition_penalty=gen["repetition_penalty"],
                         eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                         stopping_criteria=StoppingCriteriaList([StopOnStrings(tok, ids.shape[1])]))
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    new = out[0, ids.shape[1]:]
    raw = tok.decode(new, skip_special_tokens=True)
    stopped = (len(new) > 0 and int(new[-1]) == tok.eos_token_id) or any(s in raw for s in STOP_STRINGS)
    return {"answer": clean_answer(raw), "raw": raw, "latency_s": time.perf_counter() - t0,
            "new_tokens": int(len(new)), "stopped": bool(stopped)}


@torch.no_grad()
def complete(model, tok, prompts, n=50):
    """Plain greedy continuation (before/after samples)."""
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt")["input_ids"].to(model.device)
        out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


def pack(texts, tok, block_size, keep_remainder):
    """Concatenate documents separated by <eos> and cut into fixed-size blocks (labels = inputs)."""
    stream = []
    for t in texts:
        stream += tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    blocks = [stream[i:i + block_size] for i in range(0, len(stream), block_size)]
    if blocks and len(blocks[-1]) < block_size and not keep_remainder:
        blocks = blocks[:-1]
    return [{"input_ids": b, "labels": list(b)} for b in blocks if len(b) > 1]


class Collator:
    """Pads a batch; padded positions get label -100 so they never count in the loss."""
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id); labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"]); labels[i, :L] = torch.tensor(b["labels"]); mask[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


@torch.no_grad()
def perplexity(model, blocks, pad_id, batch_size=8):
    """exp(mean next-token loss) over all held-out tokens (token-weighted)."""
    model.eval(); collate = Collator(pad_id); total, count = 0.0, 0
    for i in range(0, len(blocks), batch_size):
        b = {k: v.to(model.device) for k, v in collate(blocks[i:i + batch_size]).items()}
        logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1].float()
        labels = b["labels"][:, 1:]
        total += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                                   ignore_index=-100, reduction="sum").item()
        count += (labels != -100).sum().item()
    return math.exp(total / count)
