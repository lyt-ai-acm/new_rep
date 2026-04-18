# -*- coding: utf-8 -*-
from _train_transformer_common import build_parser, run_transformer_finetune


def main():
    parser = build_parser(default_model_name="hfl/chinese-roberta-wwm-ext")
    args = parser.parse_args()
    run_transformer_finetune(args)


if __name__ == "__main__":
    main()
