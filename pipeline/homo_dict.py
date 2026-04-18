# -*- coding: utf-8 -*-
from typing import Dict, List


def load_word_homo(path: str) -> Dict[str, List[str]]:
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            vals = [x.strip() for x in parts[1:] if x.strip()]
            table[key] = vals[:10]
    return table


def load_char_homo(path: str, topk_char: int = 15) -> Dict[str, List[str]]:
    table = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0].strip()
            vals = [x.strip() for x in parts[1:] if x.strip()]
            table[key] = vals[:topk_char]
    return table