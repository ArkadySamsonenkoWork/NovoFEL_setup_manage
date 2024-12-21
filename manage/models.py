import itertools
import csv
import re
import shutil
import time
import typing as tp
import warnings
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch
import yaml
from matplotlib import pyplot as plt
import matplotlib

# from epics import PV
from Servers.Server import PV
from plot_app import PlotApp


class DensityGetter:
    possible_formats = ("numpy", "torch")
    def __init__(self, config_path: tp.Union[str, Path], data_format: str = "numpy", seed: int = 42):
        self.density_parameters = self._get_density_parameters(config_path)
        self.generator = self._get_generator(data_format, seed)

    def _get_density_parameters(self, config_path: tp.Union[str, Path]):
        path = Path(config_path) / "agent_configs/densities/uniform_density.yaml"
        with open(path, "r") as read_file:
            density_parameters = yaml.safe_load(read_file)
        return density_parameters

    def _get_generator(self, data_format: str, seed: int):
        minimum = []
        maximum = []
        for value in self.density_parameters.values():
            minimum.append(value["min"])
            maximum.append(value["max"])
        if data_format == "numpy":
            return self._get_numpy_generator(minimum, maximum, seed)
        elif data_format == "torch":
            return self._get_torch_generator(minimum, maximum, seed)
        else:
            raise ValueError(f"The format can be only {self.possible_formats}. Got instead {format}")

    def _get_numpy_generator(self, minimum: list[float], maximum: list[float], seed: int):
        def generator(size: int):
            return np.random.default_rng(seed).uniform(minimum, maximum, (size, len(maximum)))
        return  generator

    def _get_torch_generator(self, maximum: list[float], minimum: list[float], seed: int):
        torch.manual_seed(seed)
        minimum = torch.tensor(minimum)
        maximum = torch.tensor(maximum)
        def generator(size: int):
            return (maximum - minimum) * torch.rand((size, len(maximum))) + minimum
        return  generator

    def get_values(self, size: int):
        data = self.generator(size)
        return [dict(zip(self.density_parameters.keys(), value)) for value in data]

class Model(ABC):
    @abstractmethod
    def __call__(self, values: dict[str: float]) -> dict[str: float]:
        pass

    @abstractmethod
    def add(self, state: dict[str: float]) -> None:
        pass

    @abstractmethod
    def metadata(self) -> dict[tp.Any]:
        """
        :return: The dict of all model config parameters
        """
        pass

class BayessianModel(Model):
    def __init__(self, names: list[str]):
        self.names = names

    def __call__(self, values):
        pass

    def add(self, state):
        pass

    def metadata(self):
        return {"1": "some metadata"}
