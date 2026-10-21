"""Train reference evidence with pair-preserving or matched global updating."""
from experiments.geoprr_v2_reference_pair_reader import ReferencePairReader, reference_pair_losses
from experiments.train_geoprr_v2_reference_relation import main as train_reference_reader


def main(argv=None):
    return train_reference_reader(argv, reader_class=ReferencePairReader,
                                  loss_function=reference_pair_losses,
                                  full_initialization="full_best64")


if __name__ == "__main__":
    raise SystemExit(main())
