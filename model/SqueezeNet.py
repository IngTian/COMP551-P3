import torch
import torch.nn as nn
from model import Model

class Fire(nn.Module):

    def __init__(
        self,
        in_channels: int,
        squeeze_out: int,
        expand1x1_planes: int,
        expand3x3_planes: int
    ) -> None:
        super(Fire, self).__init__()
        self.inplanes = in_channels
        self.squeeze = nn.Conv2d(in_channels, squeeze_out, 1)
        self.squeeze_activation = nn.ReLU()
        self.expand1x1 = nn.Conv2d(squeeze_out, expand1x1_planes, 1)
        self.expand1x1_activation = nn.ReLU()
        self.expand3x3 = nn.Conv2d(squeeze_out, expand3x3_planes, 3, padding=1)
        self.expand3x3_activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)



class Net(Model):

    def __init__(self):

        super(Net, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(1, 96, 7, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            Fire(96, 16, 64, 64),
            Fire(128, 16, 64, 64),
            Fire(128, 32, 128, 128),
            nn.MaxPool2d(3, stride=2),
            Fire(256, 32, 128, 128),
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(3, stride=2),
            Fire(512, 64, 256, 256),
        )


        self.final = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 36, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.sequence(x)
        x = self.final(x)
        return torch.flatten(x, 1)


    def p(self, data: torch.Tensor) -> torch.Tensor:
        number_of_instances = len(data)
        self.eval()
        with torch.no_grad():
            output = self(data[:, None, :].float())
        self.train()
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








