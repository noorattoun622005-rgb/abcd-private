"""Data preparation: the continued-pretraining corpus (cleaning, PII placeholders, split by document)
and the SFT pairs (loading, overlap check against the test set, train/validation split)."""
import gzip
import json
import random
import re
import shutil
import unicodedata
import urllib.request

import pandas as pd

from .config import path

# ---------------------------------------------------------------- cleaning
_ASCII = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
          "→": "->", "…": "...", " ": " "}


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    for k, v in _ASCII.items():
        text = text.replace(k, v)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def guideline_documents(guidelines):
    """One document per subflow guideline (55)."""
    docs = []
    for flow, fv in guidelines.items():
        for sub, sv in fv["subflows"].items():
            lines = ["AcmeCorp customer support guidelines.", f"Flow: {flow} ({fv['description']}). Subflow: {sub}."]
            instr = [i.strip() + ("" if i.strip()[-1:] in ".:!?" else ".") for i in sv.get("instructions", []) if i.strip()]
            if instr:
                lines.append("Instructions: " + " ".join(instr))
            for n, a in enumerate(sv["actions"], 1):
                text = a["text"].strip() if a["text"].strip().lower() not in ("x", "") else f"Use [{a['button']}]."
                lines.append(f"Step {n} - {a['button']} ({a['type']}): {text}")
                lines += ["- " + s.replace("<or>", "or").replace(" > ", ": ").replace("> ", "").strip() for s in a["subtext"]]
            docs.append({"source": "guideline", "flow": flow, "subflow": sub, "text": clean_text("\n".join(lines))})
    return docs


def kb_document(kb):
    """The knowledge base of action sequences as one document."""
    lines = ["AcmeCorp support knowledge base: the actions an agent takes, in order, for each subflow."]
    lines += [f"{s.replace('_', ' ')}: " + ", then ".join(a.replace("-", " ") for a in acts) + "." for s, acts in kb.items()]
    return {"source": "kb", "flow": "All", "subflow": "action sequences", "text": "\n".join(lines)}


PII_FIELDS = [("order", "full_address", "<address>"), ("order", "street_address", "<street address>"),
              ("personal", "email", "<email>"), ("personal", "customer_name", "<name>"),
              ("personal", "username", "<username>"), ("personal", "account_id", "<account id>"),
              ("personal", "phone", "<phone>"), ("personal", "pin_number", "<pin>"),
              ("personal", "security_answer", "<security answer>"), ("personal", "password", "<password>"),
              ("order", "order_id", "<order id>"), ("order", "zip_code", "<zip code>")]
GENERIC_PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
               (re.compile(r"\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{4}"), "<phone>"),
               (re.compile(r"\b\d{9,}\b"), "<number>"),
               (re.compile(r"\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10}\b", re.I), "<id>")]


def scrub(text, scenario):
    """Replace this conversation's personal values (known from its scenario) and any generic e-mail/phone/ID."""
    pairs = [(str(scenario.get(sec, {}).get(k) or "").strip(), ph) for sec, k, ph in PII_FIELDS]
    pairs += [(part, "<name>") for part in str(scenario.get("personal", {}).get("customer_name") or "").split()]
    for val, ph in sorted([p for p in pairs if len(p[0]) >= 3], key=lambda p: -len(p[0])):
        text = re.sub(r"(?<!\w)" + re.escape(val) + r"(?!\w)", ph, text, flags=re.I)
    for rx, ph in GENERIC_PII:
        text = rx.sub(ph, text)
    return re.sub(r"(<name>)(\s+<name>)+", r"\1", text)


ROLE_LABEL = {"agent": "Agent", "customer": "Customer", "action": "Action"}


def dialogue_document(convo):
    """One ABCD conversation as Customer: / Agent: / Action: lines."""
    sc = convo["scenario"]
    lines = [f"Customer support conversation. Flow: {sc['flow'].replace('_', ' ')}. Subflow: {sc['subflow'].replace('_', ' ')}."]
    for speaker, utt in convo["original"]:
        utt = clean_text(scrub(utt, sc))
        if utt:
            lines.append(f"{ROLE_LABEL.get(speaker, speaker.title())}: {utt}")
    return {"source": "dialogue", "flow": sc["flow"], "subflow": sc["subflow"], "text": "\n".join(lines)}


# ---------------------------------------------------------------- corpus
def load_conversations(cfg):
    raw = cfg["work"] / "data/abcd_v1.1.json.gz"
    if not raw.exists():
        local = path(cfg, "guidelines").parent / "raw/abcd_v1.1.json.gz"
        if local.exists():
            shutil.copy(local, raw)
        else:
            print("downloading ABCD v1.1 from GitHub ...")
            urllib.request.urlretrieve(cfg["paths"]["abcd_url"], raw)
    abcd = json.load(gzip.open(raw, "rt"))
    convos = sorted((c for split in ("train", "dev", "test") for c in abcd[split]), key=lambda c: c["convo_id"])
    if cfg["data"]["max_dialogues"]:
        convos = random.Random(cfg["seed"]).sample(convos, cfg["data"]["max_dialogues"])
    return convos


def split_documents(docs, cfg):
    """Held-out 10% by document, stratified (policy documents and dialogues split separately).
    Product Defect guidelines are pinned to train; a further 5% of dialogues becomes the validation set."""
    rng = random.Random(cfg["seed"])
    train, val, held = [], [], []
    policy = [d for d in docs if d["source"] != "dialogue"]
    dialogues = [d for d in docs if d["source"] == "dialogue"]
    free = [d for d in policy if d["flow"] != "Product Defect"]
    rng.shuffle(free)
    n_held = round(cfg["data"]["heldout_frac"] * len(policy))
    held += free[:n_held]; train += [d for d in policy if d not in held]
    rng.shuffle(dialogues)
    n_held, n_val = round(cfg["data"]["heldout_frac"] * len(dialogues)), round(cfg["data"]["val_frac"] * len(dialogues))
    held += dialogues[:n_held]; val += dialogues[n_held:n_held + n_val]; train += dialogues[n_held + n_val:]
    return train, val, held


def build_corpus(cfg, tokenizer=None):
    """Returns train_texts (policy documents repeated in train only), val_texts, held_texts and saves the
    cleaned splits to data/cpt/ plus corpus statistics to results/corpus_stats.csv."""
    guidelines = json.load(open(path(cfg, "guidelines")))
    kb = json.load(open(path(cfg, "kb")))
    handbook = [json.loads(l) for l in open(path(cfg, "handbook"))]
    convos = load_conversations(cfg)

    docs = guideline_documents(guidelines) + [kb_document(kb)] + [dialogue_document(c) for c in convos]
    seen, unique = set(), []
    for d in docs:                                   # exact-duplicate removal
        if d["text"].lower() not in seen:
            seen.add(d["text"].lower()); unique.append(d)
    print(f"{len(docs)} documents -> {len(unique)} after removing {len(docs) - len(unique)} exact duplicates")

    train_docs, val_docs, held_docs = split_documents(unique, cfg)
    policy_train = [d for d in train_docs if d["source"] != "dialogue"] + handbook
    train_texts = ([d["text"] for d in train_docs if d["source"] == "dialogue"]
                   + [d["text"] for d in policy_train] * cfg["data"]["policy_upsample"])
    random.Random(cfg["seed"]).shuffle(train_texts)
    val_texts, held_texts = [d["text"] for d in val_docs], [d["text"] for d in held_docs]

    out = cfg["work"] / "data/cpt"
    for name, rows in [("train", train_docs), ("validation", val_docs), ("heldout", held_docs)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    if tokenizer is not None:
        count = lambda texts: sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) + 1 for t in texts)
        stats = pd.DataFrame([
            {"split": "train (unique documents)", "documents": len(train_docs) + len(handbook),
             "tokens": count([d["text"] for d in train_docs] + [d["text"] for d in handbook])},
            {"split": f"train (policy docs x{cfg['data']['policy_upsample']})", "documents": len(train_texts), "tokens": count(train_texts)},
            {"split": "validation (early stopping)", "documents": len(val_docs), "tokens": count(val_texts)},
            {"split": "held-out 10% (perplexity)", "documents": len(held_docs), "tokens": count(held_texts)},
        ])
        stats.to_csv(cfg["work"] / "results/corpus_stats.csv", index=False)
        print(stats.to_string(index=False))
    return train_texts, val_texts, held_texts


def heldout_texts(cfg):
    """Held-out documents for perplexity: from data/cpt/heldout.jsonl if it exists, else rebuilt."""
    p = cfg["work"] / "data/cpt/heldout.jsonl"
    if p.exists():
        return [json.loads(l)["text"] for l in open(p)]
    return build_corpus(cfg)[2]


# ---------------------------------------------------------------- SFT pairs
def load_pairs(cfg):
    return [json.loads(l) for l in open(path(cfg, "sft_pairs"))]


def load_tests(cfg):
    return pd.read_csv(path(cfg, "tests"), keep_default_na=False)


def check_overlap(pairs, tests, out_csv, threshold=0.60):
    """No SFT question may copy or paraphrase a test question: exact match + character n-gram TF-IDF cosine."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    norm = lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9$ ]", " ", s.lower())).strip()
    test_q = [norm(q) for q in tests.customer_input]
    sft_q = [norm(p["question"]) for p in pairs]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(test_q + sft_q)
    sim = cosine_similarity(vec.transform(test_q), vec.transform(sft_q))
    audit = pd.DataFrame({"test_id": tests.test_id, "test_question": tests.customer_input,
                          "nearest_sft_question": [pairs[j]["question"] for j in sim.argmax(1)],
                          "cosine": sim.max(1).round(3)}).sort_values("cosine", ascending=False)
    audit.to_csv(out_csv, index=False)
    exact = set(test_q) & set(sft_q)
    print(f"overlap check: exact matches {len(exact)} | highest similarity {audit.cosine.max():.3f} (threshold {threshold})")
    assert not exact and audit.cosine.max() < threshold, "an SFT question is too close to a test question"
    return audit


def split_pairs(pairs, cfg):
    """10% of the pairs for validation, split by paraphrase group (the same fact is never on both sides)."""
    groups = sorted({p["group"] for p in pairs}); random.Random(cfg["seed"]).shuffle(groups)
    val_groups = set(groups[:round(cfg["sft"]["val_frac"] * len(groups))])
    return [p for p in pairs if p["group"] not in val_groups], [p for p in pairs if p["group"] in val_groups]
