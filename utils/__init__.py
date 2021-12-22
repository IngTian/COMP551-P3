import json
from enum import Enum
from pathlib import Path

import numpy as np
from typing import Union, List, Callable, Tuple
from matplotlib import pyplot as plt
import torch
import cv2 as cv
from torch.utils.data import Dataset
from simple_chalk import chalk
import psutil
import humanize
import os
import GPUtil as GPU
from scipy import ndimage
import matplotlib.pyplot as plt

PreprocessingFunction = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]


def print_m():
    """
    Print the current memory usage of GPU.
    :return:
    """
    colab_gpus = GPU.getGPUs()
    gpu = colab_gpus[0]
    process = psutil.Process(os.getpid())
    print("Gen RAM Free: " + humanize.naturalsize(psutil.virtual_memory().available),
          " |     Proc size: " + humanize.naturalsize(process.memory_info().rss))
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total     {3:.0f}MB".format(gpu.memoryFree,
                                                                                                    gpu.memoryUsed,
                                                                                                    gpu.memoryUtil * 100,
                                                                                                    gpu.memoryTotal))


class DataType(Enum):
    LABELED_TRAINING = 1
    UN_LABELED_TRAINING = 2
    TEST = 3
    VALIDATION = 4


class Data(Dataset):

    def __init__(self, x: np.ndarray, y: Union[np.ndarray, None], data_type: DataType):
        self.tensor_x = torch.from_numpy(x)
        if y is not None:
            self.tensor_y = torch.from_numpy(y)
        else:
            self.tensor_y = None
        self.data_type = data_type

    def display(self, n: int = 5) -> None:
        """
        Give a preview of the data.
        :param n: Number of data to be displayed.
        :return: Nothing.
        """
        print(f'{chalk.bold("PRINTING FOR DATA TYPE: ")} {chalk.yellow(str(self.data_type))}\n')
        for idx in range(min(n, len(self.tensor_x))):
            print(f'{chalk.bold("-" * 25 + "INDEX: " + str(idx + 1) + "-" * 25)}\n')
            plt.imshow(self.tensor_x[idx].detach().numpy())
            plt.show()
            print(f'{chalk.greenBright("y:")}\n')
            print(self.tensor_y[idx].detach().numpy())
            print(f'{chalk.bold("-" * 50)}\n')

    def preprocess_and_build(self, pipelines: List[PreprocessingFunction]) -> None:
        """
        Run the data through the preprocess pipeline,
        and build the corresponding tensors.
        :param pipelines: The pipeline of preprocessing functions.
        :return: Nothing.
        """
        for pipeline in pipelines:
            self.tensor_x, self.tensor_y = pipeline(self.tensor_x, self.tensor_y)

    def __len__(self):
        return len(self.tensor_x)

    def __getitem__(self, index):
        return self.tensor_x[index], self.tensor_y[index] if self.tensor_y is not None else torch.tensor([])


class RawData:
    labeled_training: Data
    unlabeled_training: Data
    validation: Data
    test: Data


def preprocess_de_noise(xs: torch.Tensor, ys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    for i, X in enumerate(xs):
        image_mask = X > 100
        X[image_mask] = 255
        image_mask = X <= 100
        X[image_mask] = 0
        xs[i] = X

    for i, X in enumerate(xs):
        X = np.uint8(X)
        _, thresh = cv.threshold(X, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        connectivity = 4
        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(thresh, connectivity, cv.CV_32S)
        # print(num_labels)
        sizes = stats[1:, -1]  # get CC_STAT_AREA component
        img2 = np.zeros((labels.shape), np.uint8)

        for j in range(0, num_labels - 1):
            if sizes[j] >= 3:  # filter small dotted regions
                img2[labels == j + 1] = 255

        res = cv.bitwise_not(img2)
        xs[i] = torch.from_numpy(res)

    return xs, ys


def image_rotation(xs: np.ndarray, degrees):
    result = []
    for d in degrees:
        rotated_images = [ndimage.rotate(x, d, reshape=False) for x in xs]
        result += rotated_images
    result = np.asarray(result)
    return result


def plot_accuracy_trends(epochs: np.ndarray, trends: List[np.ndarray], legends: List[str], fig_name: str):
    """
    Plot libs.
    :param fig_name: Name of the figure
    :param epochs: 1D numpy array. [1,2,3,4,...,50]
    :param trends: List of trends, each is a 1D numpy array. [[10.0,20.0,30.0,...,90.0], [20.0, 30.0,...,]]
    :param legends: List of legends ["training_accuracy", "validation accuracy"]
    :return: Nothing
    """
    for acc in trends:
        plt.plot(epochs, acc)

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(legends)

    plt.savefig(fig_name)
    plt.show()


def json_to_numpy(json_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fObj = open(json_path)
    res = json.load(fObj)
    i = 0

    epoch_list = []
    training_acc_list = []
    validation_acc_list = []

    for dic in res:
        if i < 50:
            epoch_list.append(dic.get("epoch"))
            training_acc_list.append((1 - dic.get("training_error")))
            validation_acc_list.append((1 - dic.get("validation_error")))
        i += 1

    np.array(epoch_list)
    np.array(training_acc_list)
    np.array(validation_acc_list)

    return epoch_list, training_acc_list, validation_acc_list
