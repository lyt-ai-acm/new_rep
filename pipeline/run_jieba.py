# -*- coding: utf-8 -*-
from typing import List, Tuple
from pypinyin import lazy_pinyin


def word_py_key(word: str) -> str:
    return "_".join(lazy_pinyin(word))


def char_py(ch: str) -> str:
    x = lazy_pinyin(ch)
    return x[0] if x else ""


def edit_cost(src: str, tgt: str) -> int:
    n = min(len(src), len(tgt))
    c = sum(1 for i in range(n) if src[i] != tgt[i])
    c += abs(len(src) - len(tgt))
    return c


def get_word_candidates(tok: str, word_table) -> List[Tuple[str, float]]:
    key = word_py_key(tok)
    cands = word_table.get(key, [])
    out = []
    for i, c in enumerate(cands):
        out.append((c, 1.0 / (i + 1)))
    return out


def get_char_fallback(tok: str, char_table) -> List[Tuple[str, float]]:
    out = []
    chars = list(tok)
    for idx, ch in enumerate(chars):
        py = char_py(ch)
        if not py:
            continue
        cands = char_table.get(py, [])
        for i, cc in enumerate(cands):
            if cc == ch:
                continue
            x = chars[:]
            x[idx] = cc
            out.append(("".join(x), 0.5 / (i + 1)))
    return out