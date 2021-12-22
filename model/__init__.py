from typing import Dict, Any, Callable, List, Tuple
from pathlib import Path
from utils import RawData, Data, PreprocessingFunction
from torch.nn import Module, CrossEntropyLoss
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader
from torch import device, Tensor
import torch
from simple_chalk import chalk


class TrainingRecord:

    def __init__(self, epoch: int, train_error: float, val_error: float):
        self.epoch = epoch
        self.training_error = train_error
        self.validation_error = val_error

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class Model(Module):
    def p(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions based on the given data.
        :param data: A NxD two-dimensional array.
        :return: A NxL two-dimensional array.
        """
        pass

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepare the input for the model.
        :param x: A NxD two-dimensional array.
        :return: A tensor that satisfies the model requirement.
        """
        pass


class ModelTester:

    def __init__(
            self,
            raw_data: RawData,
            model: Callable[[], Model],
            preprocesses: List[PreprocessingFunction] = [],
            verbose: bool = True
    ):
        self.__device = device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Prepare data.
        self.__training_data: Data = raw_data.labeled_training
        self.__training_data.preprocess_and_build(preprocesses)
        self.__test_data: Data = raw_data.test
        self.__test_data.preprocess_and_build(preprocesses)
        self.__training_unlabeled_data: Data = raw_data.unlabeled_training
        self.__training_unlabeled_data.preprocess_and_build(preprocesses)
        self.__validation_data: Data = raw_data.validation
        self.__validation_data.preprocess_and_build(preprocesses)

        # Prepare model.
        self.__trained_model: Model = model()
        self.__trained_model.to(self.__device)

        if verbose:
            print(
                f'{chalk.bold("CREATED MODEL TESTER")}\n'
                f'DEVICES USED           : {str(self.__device.type)}\n'
                f'# OF LABELED TRAINING  : {str(len(self.__training_data))}\n'
                f'# OF UNLABELED TRAINING: {str(len(self.__training_unlabeled_data))}\n'
                f'# OF VALIDATION        : {str(len(self.__validation_data))}\n'
                f'# OF TEST              : {str(len(self.__test_data))}'
            )

    def train(
            self,
            epochs: int = 10,
            epochs_record: int = 1,
            records_load_rate: Tuple[int] = (100, 100),
            batch_size: int = 50,
            loss_function: Callable[..., Module] = CrossEntropyLoss,
            loss_function_options: Dict[str, Any] = {},
            optimizer: Callable[..., Optimizer] = SGD,
            optimizer_options: Dict[str, Any] = {},
            verbose: bool = True,
            model_save_folder: Path = None
    ) -> List[Dict[str, Any]]:
        """
        Train the model.
        :param epochs: Number of epochs.
        :param epochs_record: Number of epochs to record performance.
        :param records_load_rate:
        :param batch_size: The training batch size.
        :param loss_function: The loss function to be used.
        :param loss_function_options: Extra parameters passed into the loss function.
        :param optimizer: The optimizer to be used.
        :param optimizer_options: Extra parameters passed into the optimizer.
        :param verbose: Whether to print logs.
        :param model_save_folder: The directory to the folder to which the model is saved each epoch.
        :return: Nothing.
        """
        criterion = loss_function(**loss_function_options)
        gradient_descent = optimizer(self.__trained_model.parameters(), **optimizer_options)
        train_data_loader = DataLoader(self.__training_data, batch_size=batch_size, shuffle=True, num_workers=1)
        records: List[TrainingRecord] = list()

        for epoch in range(1, epochs + 1):
            running_loss = 0.0

            for i, data in enumerate(train_data_loader):
                x, y = data
                assert (isinstance(x, Tensor))
                assert (isinstance(y, Tensor))

                # Prepare input.
                x = self.__trained_model.prepare_input(x)
                if self.__device.type == 'cuda':
                    x, y = x.cuda(), y.cuda()
                else:
                    x, y = x.cpu(), y.cpu()

                # Update weights.
                gradient_descent.zero_grad()
                output = self.__trained_model(x)
                l = criterion(output, y)
                l.backward()
                gradient_descent.step()

                # Report losses.
                if verbose:
                    running_loss += l.item()
                    print(f'{chalk.bold("EPOCHS: ") + str(epoch)},'
                          f'{chalk.bold("B: ") + str(i)},'
                          f'{chalk.bold("LOSS: ") + str(running_loss)}')
                    running_loss = 0.0

            # Report training accuracy.
            if epoch % epochs_record == 0:
                training_error = self.get_training_error(records_load_rate[0])
                validation_error = self.validate(records_load_rate[1])
                print(f'{chalk.bold("COMPLETED EPOCH: " + str(epoch))} '
                      f'{chalk.bold("TRAINING ERROR RATE: ")} '
                      f'{chalk.redBright("{:.1f}".format(training_error * 100))}% '
                      f'{chalk.bold("VALIDATION ERROR RATE: ")} '
                      f'{chalk.redBright("{:.1f}".format(validation_error * 100))}%')
                records.append(TrainingRecord(epoch, training_error, validation_error))

            # Save model, if required.
            if model_save_folder is not None:
                self.save(model_save_folder / f"epoch_{epoch}.pt")

        return list(map(lambda record: record.to_dict(), records))

    def get_training_error(self, batch_size: int = 2000) -> float:
        """
        Get a running average of train data,
        which avoids memory overflow.
        :param batch_size: The batch size of running average.
        :return: The average error rate.
        """
        total_error: float = 0.0
        terms = 0
        loader = DataLoader(self.__training_data, batch_size=batch_size, shuffle=False)

        for i, data in enumerate(loader):
            terms += 1
            x, y = data
            total_error += ModelTester.calculate_error_rate(
                y.cuda() if self.__device.type == 'cuda' else y.cpu(),
                self.__trained_model.p(x.cuda() if self.__device.type == 'cuda' else x.cpu())
            )

        return total_error / terms

    def validate(self, batch_size: int = 400) -> float:
        """
        Get a running average of validation data,
        which avoids memory overflow.
        :param batch_size: The batch size of running average.
        :return: The average error rate.
        """
        total_error: float = 0.0
        terms = 0
        loader = DataLoader(self.__validation_data, batch_size=batch_size, shuffle=False)

        for i, data in enumerate(loader):
            terms += 1
            x, y = data
            total_error += ModelTester.calculate_error_rate(
                y.cuda() if self.__device.type == 'cuda' else y.cpu(),
                self.__trained_model.p(x.cuda() if self.__device.type == 'cuda' else x.cpu())
            )

        return total_error / terms

    def test(self, step_size: int = 500) -> Tensor:
        """
        Predict test sets.
        :param step_size: Sized for each prediction to avoid overflow.
        :return: Predicted labels.
        """
        result = torch.tensor([])
        loader = DataLoader(self.__test_data, shuffle=False, batch_size=step_size)
        for idx, data in enumerate(loader):
            cur_x, _ = data
            cur_x = cur_x.cuda() if self.__device.type == "cuda" else cur_x.cpu()
            cur_predict = self.__trained_model.p(cur_x).detach().cpu()
            result = torch.cat((result, cur_predict), dim=0)
        return result

    def predict_unlabeled(self, step_size: int = 500)  -> Tensor:
      """
        Predict unlabled sets.
        :param step_size: Sized for each prediction to avoid overflow.
        :return: Predicted labels.
        """
      result = torch.tensor([])
      loader = DataLoader(self.__training_unlabeled_data, shuffle=False, batch_size=step_size)
      for idx, data in enumerate(loader):
          cur_x, _ = data
          cur_x = cur_x.cuda() if self.__device.type == "cuda" else cur_x.cpu()
          cur_predict = self.__trained_model.p(cur_x).detach().cpu()
          result = torch.cat((result, cur_predict), dim=0)
      return result



    def save(self, path: Path) -> None:
        """
        Save the model as a state dict.
        :param path: The path of the saved model.
        :return: Nothing.
        """
        torch.save(self.__trained_model.state_dict(), path)

    def load(self, path: Path) -> None:
        """
        Load the model from the specified path.
        :param path: The path of the model to be loaded.
        :return: Nothing.
        """
        self.__trained_model.load_state_dict(torch.load(path, map_location=torch.device(
            'cpu') if self.__device.type == 'cpu' else torch.device('cuda')))

    @staticmethod
    def calculate_error_rate(true_value: Tensor, predicted_value: Tensor) -> float:
        """
        Calculate the error rate.
        :param true_value: A NxD 2-dimensional array.
        :param predicted_value: A: NxD 2-dimensional array.
        :return: The error rate.
        """
        number_of_instances = len(true_value)
        error_count = torch.count_nonzero(torch.count_nonzero(predicted_value != true_value, dim=1)).item()
        return error_count / number_of_instances

    @property
    def model(self) -> Model:
        return self.__trained_model

    @property
    def labeled_training_data(self) -> Data:
        return self.__training_data

    @property
    def unlabeled_training_data(self) -> Data:
        return self.__training_unlabeled_data

    @property
    def validation_data(self) -> Data:
        return self.__validation_data

    @property
    def test_data(self) -> Data:
        return self.__test_data
