import torch
import torch.nn as nn
import torch.nn.functional as f
from model import Model


class Net(Model):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, (3, 3))
        self.conv2 = nn.Conv2d(6, 16, (3, 3))
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc_number = nn.Linear(84, 10)
        self.fc_alphabet = nn.Linear(84, 26)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number as
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        x = x.view(-1, Net.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        number_predictions = f.relu(self.fc_number(x))
        alphabet_predictions = f.relu(self.fc_alphabet(x))
        return torch.cat((number_predictions, alphabet_predictions), 1)

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

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
