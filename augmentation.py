from pathlib import Path
from model import ModelTester
from model.CNN import Net
from utils import RawData, DataType, Data, preprocess_de_noise, image_rotation
import pickle
from simple_chalk import chalk
import json
import os
from torch.optim import Adam, SGD
import numpy as np

OUTPUT_PATH = Path("./out")
OUTPUT_MODELS_PATH = OUTPUT_PATH / "models"
OUTPUT_RECORDS_PATH = OUTPUT_PATH / "records.json"
DATASET_PATH = Path("./dataset")


def read_data(path: Path) -> RawData:
    """
    Read data from the specified data set folder.
    :param path:
    :return:
    """
    training_x_path = path / "images_l.pkl"
    training_label_path = path / "labels_l.pkl"
    training_unlabeled_x_path = path / "images_ul.pkl"
    test_path = path / "images_test.pkl"

    with open(training_x_path, 'rb') as f:
        training_x = pickle.load(f)

    with open(training_label_path, 'rb') as f:
        training_label = pickle.load(f)

    with open(training_unlabeled_x_path, 'rb') as f:
        training_unlabeled = pickle.load(f)

    with open(test_path, 'rb') as f:
        test = pickle.load(f)

    degrees = [15, 30, 0, -15, -30]
    val_x = training_x[:1000]
    val_y = training_label[:1000]
    training_x = image_rotation(training_x[1000:], degrees)
    training_label = np.tile(training_label[1000:], (len(degrees) + 1, 1))

    validation_x, validation_y = val_x, val_y

    training_labeled = Data(training_x, training_label, DataType.LABELED_TRAINING)
    training_unlabeled = Data(training_unlabeled, None, DataType.UN_LABELED_TRAINING)
    test = Data(test, None, DataType.TEST)
    validation = Data(validation_x, validation_y, DataType.VALIDATION)

    result = RawData()
    result.labeled_training = training_labeled
    result.unlabeled_training = training_unlabeled
    result.test = test
    result.validation = validation

    return result


if __name__ == '__main__':
    # result = read_data(Path(DATASET_PATH))

    tester = ModelTester(
        read_data(Path(DATASET_PATH)),
        Net,
        preprocesses=[]
    )
    print(f'{chalk.bold("-" * 25 + "START TRAINING" + "-" * 25)}')

    # Create the folder if the folder does not exist.
    if not OUTPUT_MODELS_PATH.exists():
        os.makedirs(OUTPUT_MODELS_PATH.absolute())

    training_records = tester.train(
        epochs=50,
        optimizer=SGD,
        optimizer_options={
            "lr": 0.0005,
            "momentum": 0.99
        },
        verbose=False,
        model_save_folder=OUTPUT_MODELS_PATH
    )
    print(f'{chalk.bold("-" * 25 + "TRAINING COMPLETED" + "-" * 25)}')

    # Save training records.
    with open(OUTPUT_RECORDS_PATH, 'w') as output_file:
        json.dump(training_records, output_file)
