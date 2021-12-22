from pathlib import Path
import numpy as np
from model import ModelTester
from model.VGG16_Higher_MaxPooling_Leaky import Net
from utils import RawData, DataType, Data, preprocess_de_noise, plot_accuracy_trends, json_to_numpy
import pickle
from simple_chalk import chalk
import json
import os
from torch.optim import Adam, SGD
from typing import Dict, Any
import pandas as pd

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


def make_test_predict(
        model_path: Path,
        csv_result_path: Path,
        container: ModelTester
) -> None:
    print(f'{chalk.bold("-" * 25 + "START TESTING" + "-" * 25)}')
    # Load the trained model.
    container.load(model_path)
    result = container.test().detach().cpu().numpy().astype(int)

    # Convert the numpy into CSV format.
    result = result.astype(bytearray)
    copy = []
    for i in range(len(result)):
        copy.append("".join(map(lambda k: str(k), result[i])))
    result = np.array(copy)
    index = np.linspace(0, len(result) - 1, num=len(result), dtype=int).transpose()
    result = np.append(index[:, None], result[:, None], axis=1)
    frame = pd.DataFrame(result)
    frame.columns = ["# ID", "Category"]

    # Write the CSV file.
    frame.to_csv(csv_result_path.absolute(), index=False)
    print(f'{chalk.bold("-" * 25 + "TESTING COMPLETED" + "-" * 25)}')


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
    tester = ModelTester(
        read_data(Path(DATASET_PATH)),
        Net,
        preprocesses=[]
    )

    training_options = {
        "epochs": 50,
        "optimizer": SGD,
        "optimizer_options": {
            "lr": 0.0005,
            # "betas": (0.9, 0.99),
            # "eps": 1e-8,
            "momentum": 0.99
        },
        "verbose": False,
        "model_save_folder": OUTPUT_MODELS_PATH
    }

    # Make tests.
    # make_test_predict(Path("out/models") / "vgg-hm-l-967", Path("out") / "967.csv", tester)

    # Train model.
    # train_model(tester, Path("out") / "models", Path("out") / "records.json", training_options)

    # Draw the accuracy trends graphs
    legend_list = ["VGG Vanilla", "VGG Deeper", "VGG Delete Tail", "VGG Half Width", "VGG Higher MaxPooling", "VGG Higher MaxPooling with Leaky"]
    acc_list = []

    epoch_list, train_acc_list, val_acc_list = json_to_numpy("out/jsons/vgg16_vanilla.json")
    acc_list.append(val_acc_list)

    epoch_list, train_acc_list, val_acc_list = json_to_numpy("out/jsons/vgg-deeper-records.json")
    acc_list.append(val_acc_list)

    epoch_list, train_acc_list, val_acc_list = json_to_numpy("out/jsons/vgg-delete-tail-records.json")
    acc_list.append(val_acc_list)

    epoch_list, train_acc_list, val_acc_list = json_to_numpy("out/jsons/vgg-half-width-records.json")
    acc_list.append(val_acc_list)

    epoch_list, train_acc_list, val_acc_list = json_to_numpy("out/jsons/vgg-higher-max-pooling-records.json")
    acc_list.append(val_acc_list)

    epoch_list, train_acc_list, val_acc_list = json_to_numpy("out/jsons/vgg-higher-maxpooling-leaky-records.json")
    acc_list.append(val_acc_list)

    plot_accuracy_trends(epoch_list, acc_list, legend_list, "3_val_2")