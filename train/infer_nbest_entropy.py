try:
    from train.infer_with_nbest import main
except ModuleNotFoundError:
    from infer_with_nbest import main


if __name__ == "__main__":
    main()
