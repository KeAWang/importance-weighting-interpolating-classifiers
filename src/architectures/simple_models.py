from collections import OrderedDict
from collections.abc import Iterable
from typing import Union
import torch
from torch.nn import Linear, Sequential
from torch import nn
import torch.nn.functional as F


Size = Union[int, tuple]


def size_to_int(size: Size):
    if isinstance(size, Iterable):
        size = tuple(size)

    if type(size) is tuple:
        assert len(size) == 1
        return size[0]
    elif type(size) is int:
        return size
    else:
        raise ValueError(
            f"Size must be either an int or an iterable. Size is of type {type(size)}"
        )


class LinearNet(Sequential):
    def __init__(self, input_size, output_size):
        input_size = size_to_int(input_size)
        output_size = size_to_int(output_size)
        super().__init__(torch.nn.Linear(input_size, output_size))


class MLPNet(Sequential):
    """Multi-layered perception, i.e. fully-connected neural network
    Args:
        input_size: dimensionality of inputs
        hidden_size: dimensionality of hidden layers
        output_size: dimensionality of final output, usually the number of classes
        num_layers: number of hidden layers. 0 corresponds to a linear network
        activation: the string name of a torch.nn activation function
    """

    def __init__(
        self,
        input_size: Size,
        hidden_size: int,
        output_size: Size,
        num_layers: int,
        activation: str = "ReLU",
    ):
        input_size = size_to_int(input_size)
        output_size = size_to_int(output_size)
        self.depth = num_layers
        self.input_width = input_size
        self.hidden_width = hidden_size
        self.output_width = output_size
        self.activation = activation

        modules = []
        if num_layers == 0:
            modules.append(("linear1", Linear(input_size, output_size)))
        else:
            modules.append(("linear1", Linear(input_size, hidden_size)))
            for i in range(1, num_layers + 1):
                modules.append((f"{activation}{i}", getattr(torch.nn, activation)()))
                modules.append(
                    (
                        f"linear{i + 1}",
                        Linear(
                            hidden_size, hidden_size if i != num_layers else output_size
                        ),
                    )
                )
        modules = OrderedDict(modules)
        super().__init__(modules)


class ConvNet(Sequential):
    """Same architecture as Byrd & Lipton 2017 on CIFAR10
    Args:
        output_size: dimensionality of final output, usually the number of classes
    """

    input_size = (3, 32, 32)

    def __init__(self, input_size, output_size):
       assert tuple(input_size) == self.input_size
       layers = [
           torch.nn.Conv2d(3, 64, 3),
           torch.nn.ReLU(),
           torch.nn.Conv2d(64, 64, 3),
           torch.nn.ReLU(),
           torch.nn.MaxPool2d(2),
           torch.nn.Conv2d(64, 128, 3),
           torch.nn.ReLU(),
           torch.nn.Conv2d(128, 128, 3),
           torch.nn.ReLU(),
           torch.nn.Conv2d(128, 128, 3),
           torch.nn.ReLU(),
           torch.nn.MaxPool2d(2),
           torch.nn.Flatten(),
           torch.nn.Linear(2048, 512),
           torch.nn.ReLU(),
           torch.nn.Linear(512, 128),
           torch.nn.ReLU(),
           torch.nn.Linear(128, output_size),
       ]
       super().__init__(*layers)
