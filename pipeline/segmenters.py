# -*- coding: utf-8 -*-
from typing import List, Protocol
import jieba


class Segmenter(Protocol):
    def cut(self, text: str) -> List[str]:
        ...


class JiebaSegmenter:
    def cut(self, text: str) -> List[str]:
        return list(jieba.cut(text))


class PkusegSegmenter:
    def __init__(self):
        import pkuseg
        self.seg = pkuseg.pkuseg()

    def cut(self, text: str) -> List[str]:
        return self.seg.cut(text)