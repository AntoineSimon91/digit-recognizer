
#standard imports
import argparse

# local imports
from data_formatter import DataSet


def main():
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument('-n', '--nrows', default=None, type=int, help="number of rows to download")
    cli_arguments = cli_parser.parse_args()
    nrows = cli_arguments.nrows

    # download and clean train dataset
    train = DataSet("train.csv")
    train.download(nrows=nrows)
    train.split_X_Y()
    train.normalize()
    train.reshape()
    train.convert_digits_to_one_hot_vectors()
    print(train)

    # split trian/validation datasets
    validation = train.extract_validation(size=0.1)
    print(validation)

    # download clean test dataset
    test = DataSet("test.csv")
    test.download(nrows=nrows)
    test.set_X()
    test.normalize()
    test.reshape()
    print(test)


if __name__ == "__main__":
    main()
