# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List
import math


@dataclass
class Candidate:
    text: str
    lm_score: float
    prior_score: float
    edit_cost: float
    final_score: float = 0.0


def score_candidate(c: Candidate, alpha=1.0, beta=0.1, lamb=1.0) -> float:
    return alpha * c.lm_score + beta * c.prior_score - lamb * c.edit_cost


def softmax(xs: List[float], tau: float = 2.0) -> List[float]:
    if not xs:
        return []
    z = [x / max(tau, 1e-6) for x in xs]
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def gate_with_delta(cands: List[Candidate], original_text: str, delta: float = 0.3) -> List[Candidate]:
    cands = sorted(cands, key=lambda x: x.final_score, reverse=True)
    if not cands:
        return cands

    orig = None
    for c in cands:
        if c.text == original_text:
            orig = c
            break
    if orig is None:
        return cands

    best = cands[0]
    if best.final_score - orig.final_score < delta:
        cands = [orig] + [x for x in cands if x.text != original_text]
    return cands