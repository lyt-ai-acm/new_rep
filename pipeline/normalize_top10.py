# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, List, Tuple

import kenlm

from pipeline.homo_dict import load_word_homo, load_char_homo
from pipeline.candidate_gen import get_word_candidates, get_char_fallback, edit_cost
from pipeline.rerank_kenlm import Candidate, score_candidate, softmax, gate_with_delta
from pipeline.segmenters import Segmenter


@dataclass
class NormalizeConfig:
    m: int = 2
    beam_size: int = 50
    topn: int = 10
    topk_char: int = 15
    alpha: float = 1.0
    beta: float = 0.1
    lamb: float = 1.0
    delta: float = 0.3
    tau: float = 2.0


class HomophoneNormalizer:
    def __init__(
        self,
        kenlm_path: str,
        word_homo_path: str,
        char_homo_path: str,
        segmenter: Segmenter,
        cfg: NormalizeConfig
    ):
        self.lm = kenlm.Model(kenlm_path)
        self.word_table = load_word_homo(word_homo_path)
        self.char_table = load_char_homo(char_homo_path, cfg.topk_char)
        self.segmenter = segmenter
        self.cfg = cfg

    def lm_score(self, text: str) -> float:
        # 字符级空格序列，弱化分词差异
        seq = " ".join(list(text))
        return self.lm.score(seq, bos=True, eos=True)

    def suspicious_positions(self, toks: List[str]) -> List[int]:
        pos = []
        for i, t in enumerate(toks):
            if get_word_candidates(t, self.word_table) or get_char_fallback(t, self.char_table):
                pos.append(i)
        return pos[: self.cfg.m]

    def generate_topn(self, text: str):
        toks = self.segmenter.cut(text)
        orig_text = "".join(toks)

        beams = [(toks, 0.0, 0.0)]  # tokens, prior_sum, edit_sum
        s_pos = self.suspicious_positions(toks)

        for p in s_pos:
            nb = []
            for btoks, bprior, bedit in beams:
                orig_tok = btoks[p]

                # 原词保留
                nb.append((btoks, bprior, bedit))

                cands = get_word_candidates(orig_tok, self.word_table)
                if not cands:
                    cands = get_char_fallback(orig_tok, self.char_table)

                for c, prior in cands:
                    if c == orig_tok:
                        continue
                    nt = btoks[:]
                    nt[p] = c
                    nb.append((nt, bprior + prior, bedit + edit_cost(orig_tok, c)))

            nb = sorted(nb, key=lambda x: (x[1] - x[2]), reverse=True)[: self.cfg.beam_size]
            beams = nb

        uniq: Dict[str, Candidate] = {}

        # 原句保底
        base = Candidate(
            text=orig_text,
            lm_score=self.lm_score(orig_text),
            prior_score=0.0,
            edit_cost=0.0
        )
        base.final_score = score_candidate(base, self.cfg.alpha, self.cfg.beta, self.cfg.lamb)
        uniq[orig_text] = base

        for btoks, bprior, bedit in beams:
            s = "".join(btoks)
            if s in uniq:
                continue
            c = Candidate(
                text=s,
                lm_score=self.lm_score(s),
                prior_score=bprior,
                edit_cost=bedit
            )
            c.final_score = score_candidate(c, self.cfg.alpha, self.cfg.beta, self.cfg.lamb)
            uniq[s] = c

        cands = list(uniq.values())
        cands = gate_with_delta(cands, orig_text, self.cfg.delta)
        cands = sorted(cands, key=lambda x: x.final_score, reverse=True)[: self.cfg.topn]
        ws = softmax([c.final_score for c in cands], self.cfg.tau)

        out = []
        for c, w in zip(cands, ws):
            out.append({
                "text": c.text,
                "weight": float(w),
                "final_score": float(c.final_score),
                "lm_score": float(c.lm_score),
                "prior_score": float(c.prior_score),
                "edit_cost": float(c.edit_cost),
            })
        return out