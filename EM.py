from pathlib import Path
import numpy as np
from model import ModelTester
from model.VGG16_Higher_MaxPooling_Leaky import Net
from utils import RawData, DataType, Data, preprocess_de_noise
import pickle
from simple_chalk import chalk
import json
import os
from torch.optim import Adam, SGD
from typing import Dict, Any
import pandas as pd

np.random.seed(42)

# OUTPUT_PATH = Path("./out")
# OUTPUT_MODELS_PATH = OUTPUT_PATH / "models"
# OUTPUT_RECORDS_PATH = OUTPUT_PATH / "records.json"
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

    validation_x, validation_y = training_x[:1000], training_label[:1000]

    training_labeled = Data(training_x[1000:], training_label[1000:], DataType.LABELED_TRAINING)
    training_unlabeled = Data(training_unlabeled, None, DataType.UN_LABELED_TRAINING)
    test = Data(test, None, DataType.TEST)
    validation = Data(validation_x, validation_y, DataType.VALIDATION)

    result = RawData()
    result.labeled_training = training_labeled
    result.unlabeled_training = training_unlabeled
    result.test = test
    result.validation = validation

    return result


def get_random_label(labels, num):
    """
    assign random labels for unlabeled images
    :param labels: training labels
    :param num:
    :return:
    """
    index_range = np.arange(0, labels.shape[0])
    rand_indices = np.random.choice(index_range, num, replace=False)
    return labels[rand_indices]


def prepare_data(training_x, training_label, training_unlabeled, test, num) -> RawData:
    """
    :param path:
    :param num: number of unlabled data to use for training
    :return:
    """
    validation_x, validation_y = training_x[:1000], training_label[:1000]
    training_labeled = Data(training_x[1000:], training_label[1000:], DataType.LABELED_TRAINING)
    training_unlabeled = Data(training_unlabeled[:num], None, DataType.UN_LABELED_TRAINING)
    test = Data(test, None, DataType.TEST)
    validation = Data(validation_x, validation_y, DataType.VALIDATION)
    result = RawData()
    result.labeled_training = training_labeled
    result.unlabeled_training = training_unlabeled
    result.test = test
    result.validation = validation
    return result


def train_model(
        container: ModelTester,
        output_models_path: Path,
        output_records_path: Path,
        training_options: Dict[str, Any]
) -> None:
    print(f'{chalk.bold("-" * 25 + "START TRAINING" + "-" * 25)}')

    # Create the folder if the folder does not exist.
    if not output_models_path.exists():
        os.makedirs(output_models_path.absolute())

    training_records = container.train(**training_options)
    print(f'{chalk.bold("-" * 25 + "TRAINING COMPLETED" + "-" * 25)}')

    # Save training records.
    with open(output_records_path, 'w') as output_file:
        json.dump(training_records, output_file)


if __name__ == '__main__':

    path = DATASET_PATH
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

    num = 10000
    random_label = get_random_label(training_label, num)
    training_x = np.concatenate((training_x, training_unlabeled[:num]), axis=0)
    training_initial = np.concatenate((training_label, random_label[:num]), axis=0)

    result = prepare_data(training_x, training_initial, training_unlabeled, test, num)

    for i in range(20):
        print(f"iteration {i + 1}")
        tester = ModelTester(
            result,
            Net,
            preprocesses=[]
        )

        training_options = {
            "epochs": 20,
            "optimizer": SGD,
            "optimizer_options": {
                "lr": 0.0005,
                # "betas": (0.9, 0.99),
                # "eps": 1e-8,
                "momentum": 0.99
            },
            "verbose": False,
            "model_save_folder": Path(f"./out{i + 1}")
        }

        output_model_path = Path(f"out{i + 1}/models")
        output_record_path = Path(f"out{i + 1}") / "records.json"

        train_model(tester, output_model_path, output_record_path, training_options)

        with open(output_record_path, 'r') as j:
            contents = json.loads(j.read())

        best_index = 0
        min_val_error = 100
        for j in range(len(contents)):
            cur_err = contents[j]['validation_error']
            if cur_err < min_val_error:
                min_val_error = cur_err
                best_index = j
        best_index += 1

        # load the best  model
        print('Load the best model', best_index)
        tester.load(Path(f'./out{i + 1}') / f'epoch_{best_index}.pt')
        # predict the labels for unlabeled data
        predicted_labels = tester.predict_unlabeled().detach().numpy()
        # update result
        training_updated = np.concatenate((training_label, predicted_labels), axis=0)
        result = prepare_data(training_x, training_updated, training_unlabeled, test, num)
