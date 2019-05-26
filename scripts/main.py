
# standard imports
from os.path import abspath, join, dirname, pardir
import argparse

# third party imports
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# local imports
from formatter import DataSet, convert_one_hot_vectors_to_digits
import evaluation
from submission import Submission


PARENT_DIRPATH = abspath(join(dirname(__file__), pardir))    
DATASETS_DIRPATH = join(PARENT_DIRPATH, "datasets")


def main():
    # Command line Interface
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dirpath', default=DATASETS_DIRPATH, help="dataset directory path")
    parser.add_argument('-n', '--n_train', default=None, type=int, help="number of rows to download on the train dataset")
    parser.add_argument('-t', '--n_test', default=None, type=int, help="number of rows to download on the test dataset")
    parser.add_argument('-e', "--epochs", default=3, type=int, help="set the number of epochs")
    parser.add_argument('-b', "--batch_size", default=86, type=int, help="set batch size")
    cli = parser.parse_args()

    # Download and clean train dataset
    train = DataSet(dirpath=cli.dirpath, filename="train.csv")
    train.download(nrows=cli.n_train)
    train.split_X_Y()
    train.normalize()
    train.reshape()
    train.convert_digits_to_one_hot_vectors()
    print(train)

    # Split trian/validation datasets
    validation = train.extract_validation(size=0.1)
    print(validation)

    # Download clean test dataset
    test = DataSet(dirpath=cli.dirpath, filename="test.csv")
    test.download(nrows=cli.n_test)
    test.set_X()
    test.normalize()
    test.reshape()
    print(test)

    # Setup convolutional neural network model
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(rate=0.5))
    model.add(Dense(10, activation="softmax"))


    # Define the optimizer
    optimizer = RMSprop(
        lr=0.001,
        rho=0.9,
        epsilon=1e-08,
        decay=0.0
    )

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Set learning rate decay
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_acc',
        patience=3,
        verbose=1,
        factor=0.5,
        min_lr=0.00001
    )

    # Perform synthetic data augmentation
    data_generator = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0.1,  # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False  # randomly flip images
    )

    data_generator.fit(train.X)

    history = model.fit_generator(
        data_generator.flow(train.X, train.Y, batch_size=cli.batch_size),
        epochs=cli.epochs,
        validation_data=(validation.X, validation.Y),
        verbose=2,
        steps_per_epoch=train.X.shape[0] // cli.batch_size,
        callbacks=[learning_rate_reduction]
    )

    # plot loss and accuracy
    evaluation.plot_loss_and_accuracy(history)

    # Predict digits for the validation dataset
    prediction = DataSet()
    prediction.Y = model.predict(validation.X)
    prediction.X = validation.X

    prediction.convert_one_hot_vectors_to_digits()
    validation.convert_one_hot_vectors_to_digits()

    confusion_mtx = confusion_matrix(validation.Y, prediction.Y)
    evaluation.plot_confusion_matrix(confusion_mtx)

    # Predict results
    results = model.predict(test.X)
    results = convert_one_hot_vectors_to_digits(results)
    print(results)

    # Generate Submission file
    submission_file = Submission(results)
    submission_file.save()


if __name__ == "__main__":
    main()
