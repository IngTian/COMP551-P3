# noinspection DuplicatedCode
import torch
import torch.nn as nn
from torch.nn import Sequential
from torch import Tensor
from model import Model
from typing import List, Tuple, Union


class Net(Model):

    def __init__(self, input_channel: int = 1, num_classes: int = 36):
        super(Net, self).__init__()

        # Define ConvNet Constants.
        self.conv_net_architecture: List[Union[int, str]] = \
            [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
            ]
        self.fc_architecture = [
            nn.Linear(4608, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        ]
        self.conv_kernel: Tuple[int, int] = (3, 3)
        self.conv_padding: Tuple[int, int] = (1, 1)
        self.conv_stride: Tuple[int, int] = (1, 1)
        self.input_channels = input_channel
        self.num_of_classes = num_classes

        # Build network.
        self.conv_layers = self.__create_conv_nets()
        self.fc_layers = self.__create_fc_layers()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_layers(x)
        return x

    def p(self, data: torch.Tensor) -> torch.Tensor:
        number_of_instances = len(data)

        output = self(data[:, None, :].float())
        number_predicts = output[:, :10]
        alphabet_predicts = output[:, 10:]

        number_predicts_mask = torch.zeros((number_of_instances, 10), dtype=torch.bool)
        alphabet_predicts_mask = torch.zeros((number_of_instances, 26), dtype=torch.bool)
        max_number_indices = torch.argmax(number_predicts, dim=1)
        max_alphabet_indices = torch.argmax(alphabet_predicts, dim=1)
        number_predicts_mask[range(number_of_instances), max_number_indices] = True
        alphabet_predicts_mask[range(number_of_instances), max_alphabet_indices] = True

        number_predicts[number_predicts_mask] = 1
        number_predicts[~number_predicts_mask] = 0
        alphabet_predicts[alphabet_predicts_mask] = 1
        alphabet_predicts[~alphabet_predicts_mask] = 0

        return output

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, None, :].float()

    def __create_conv_nets(self) -> Sequential:
        layers = []
        in_channels = self.input_channels

        for x in self.conv_net_architecture:
            if type(x) == int:
                out_channels = x

                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=self.conv_kernel,
                        stride=self.conv_stride,
                        padding=self.conv_padding,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return Sequential(*layers)

    def __create_fc_layers(self) -> Sequential:
        return nn.Sequential(*self.fc_architecture)

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
