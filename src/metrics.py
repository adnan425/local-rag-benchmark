"""Answer-quality and retrieval metrics. EM/F1 use the standard SQuAD
normalization (lowercase, strip punctuation and articles)."""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import List

from rouge_score import rouge_scorer


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(_normalize(prediction) == _normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, gold: str) -> float:
    return _rouge.score(gold, prediction)["rougeL"].fmeasure


def recall_at_k(retrieved_ids: List[str], gold_id: str) -> float:
    return float(gold_id in retrieved_ids)


def mrr(retrieved_ids: List[str], gold_id: str) -> float:
    if gold_id in retrieved_ids:
        rank = retrieved_ids.index(gold_id) + 1
        return 1.0 / rank
    return 0.0
