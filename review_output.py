# review_output.py
import os
import re
from typing import List, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---- config ----
TOX_MODEL = "unitary/toxic-bert"

THRESH = {
    "max_contradiction_pairs": 0.10,  # now used with heuristic contradiction
    "max_toxicity": 0.40,
    "max_overclaim_rate": 0.40,
}

RED_FLAGS = [
    "kill myself", "suicide", "self harm", "harm myself",
    "overdose", "end my life", "hurt myself",
]

OVERCLAIM_PATTERNS = [
    r"\balways\b", r"\bnever\b", r"\bguarantee(s|d)?\b",
    r"\b100%\b", r"\bwill\b", r"\bmust\b", r"\bonly\b", r"\bproven\b",
]
OVERCLAIM_REGEX = [re.compile(p, flags=re.I) for p in OVERCLAIM_PATTERNS]

NEG_TOKENS = {"not", "no", "never", "n't"}
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","for","of","on","in","to",
    "is","are","was","were","be","being","been","it","this","that","these","those",
    "i","you","he","she","we","they","my","your","his","her","our","their"
}

# threads and device
torch.set_num_threads(int(os.getenv("REVIEW_NUM_THREADS", "1")))
device = torch.device(os.getenv("REVIEW_DEVICE", "cpu"))

# ---- lazy loader ----
_tox_tok = _tox = None

def _load_tox():
    global _tox_tok, _tox
    if _tox is None:
        _tox_tok = AutoTokenizer.from_pretrained(TOX_MODEL)
        _tox = AutoModelForSequenceClassification.from_pretrained(TOX_MODEL).to(device).eval()

# ---- helpers ----
def split_sents(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s for s in re.split(r"(?<=[\.\!\?])\s+", text) if s]

def has_red_flags(text: str) -> bool:
    t = (text or "").lower()
    return any(p in t for p in RED_FLAGS)

def _norm_tokens(s: str) -> List[str]:
    s = re.sub(r"[^a-z0-9\s']", " ", s.lower())
    toks = [t for t in s.split() if t and t not in STOPWORDS]
    return toks

def _has_negation(tokens: List[str]) -> bool:
    return any(t in NEG_TOKENS for t in tokens)

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def heuristic_contradiction_fraction(sents: List[str]) -> float:
    """
    Adjacent-pair check:
    Count as contradiction if exactly one sentence has negation and token Jaccard >= 0.5.
    """
    if len(sents) < 2:
        return 0.0
    pairs = 0
    contras = 0
    prev_tokens = None
    prev_has_neg = None

    for s in sents:
        toks = _norm_tokens(s)
        has_neg = _has_negation(toks)
        if prev_tokens is not None:
            pairs += 1
            j = _jaccard(set(prev_tokens), set(toks))
            if j >= 0.5 and (has_neg ^ prev_has_neg):
                contras += 1
        prev_tokens, prev_has_neg = toks, has_neg

    return contras / pairs if pairs else 0.0

@torch.no_grad()
def toxicity_score(text: str) -> float:
    _load_tox()
    enc = _tox_tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    probs = F.softmax(_tox(**enc).logits, dim=-1)[0].detach().cpu().numpy()
    return float(probs[1]) if probs.shape[-1] > 1 else float(probs[0])

def overclaim_rate(sents: List[str]) -> float:
    if not sents:
        return 0.0
    c = sum(1 for s in sents if any(p.search(s) for p in OVERCLAIM_REGEX))
    return c / len(sents)

# ---- main ----
def review_output(draft_answer: str) -> Dict:
    sents = split_sents(draft_answer)
    if not sents:
        return {
            "toxicity": 0.0,
            "contradiction_fraction": 0.0,
            "overclaim_rate": 0.0,
            "perplexity": None,
            "red_flag_terms": False,
            "decision": "decline",
        }

    tox = toxicity_score(draft_answer)
    contra = heuristic_contradiction_fraction(sents)
    overc = overclaim_rate(sents)
    flags = has_red_flags(draft_answer)

    fail = (
        tox > THRESH["max_toxicity"]
        or contra > THRESH["max_contradiction_pairs"]
        or overc > THRESH["max_overclaim_rate"]
        or flags
    )
    decision = "decline" if fail else "publish"

    return {
        "toxicity": round(tox, 4),
        "contradiction_fraction": round(contra, 4),
        "overclaim_rate": round(overc, 4),
        "perplexity": None,
        "red_flag_terms": bool(flags),
        "decision": decision,
    }

if __name__ == "__main__":
    sample = "I will always help. I will not help in some cases. You must follow every rule."
    print(review_output(sample))
