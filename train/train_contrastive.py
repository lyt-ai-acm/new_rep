# -*- coding: utf-8 -*-
"""
Model-agnostic contrastive training entrypoint.

This script reuses the SupCon + CE implementation in train_roberta_binary.py
while exposing a dedicated, paper-friendly filename.
"""
from train_roberta_binary import main


if __name__ == "__main__":
    main()
