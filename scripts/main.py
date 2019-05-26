
# local imports
from data_formatter import DataSet


def main():

    test = DataSet("test.csv")
    test.download()
    test.normalize(max_value=255.)
    test.reshape(matrix_shape=(-1, 28, 28, 1))

    train = DataSet("train.csv")
    train.download(nrows=100)
    train.split_X_Y()
    train.normalize(max_value=255.)
    train.reshape(matrix_shape=(-1, 28, 28, 1))
    train.convert_digits_to_one_hot_vectors()
    
    validation = train.extract_validation(
        random_seed=2,
        validation_size=0.1
    )

    
    print(train)
    print(validation)
    print(test)


if __name__ == "__main__":
    main()
