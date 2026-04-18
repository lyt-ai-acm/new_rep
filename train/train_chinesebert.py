# -*- coding: utf-8 -*-
from _train_transformer_common import build_parser, run_transformer_finetune


def main():
    parser = build_parser(default_model_name="shannonai/ChineseBERT-base")
    args = parser.parse_args()
    if not args.trust_remote_code:
        args.trust_remote_code = True
    run_transformer_finetune(args)


if __name__ == "__main__":
    main()
