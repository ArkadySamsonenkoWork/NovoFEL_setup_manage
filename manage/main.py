from abc import ABC, abstractmethod
import time
import shutil
import typing as tp
import sys
import os
import re
from functools import partial
from tkinter.font import names

import torch
import yaml
from pathlib import Path

from numpy.ma.core import minimum
from twisted.internet.process import detector

#from epics import PV
from Servers.Server import PV
import csv
import numpy as np


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


class Saver:
    def __init__(self, folder_path, history_limit, names: list[str]):
        self.history = []
        self.history_limit = history_limit
        path_time = time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())
        path_time = path_time.replace(":", "_")
        path_time = f"{path_time}_full_data"
        path = Path(folder_path) / Path(path_time)
        self.file = open(path, 'a')
        self.writer =  csv.DictWriter(self.file, delimiter=' ', fieldnames=names + ["time", "milliseconds"])
        self.writer.writeheader()
        self.start_time_m = time.time() * 1000

    def save(self):
        for data in self.history:
            self.writer.writerow(data)
        self.history = []

    def _get_milliseconds(self):
        return self.start_time_m - time.time() * 1000

    def append(self, data: dict[str, tp.Union[float, int]]):
        history = data.copy()
        current_time = time.strftime("%Y:%m:%d:%T", time.localtime())
        history["time"] = current_time
        history["milliseconds"] = self._get_milliseconds()
        self.history.append(history)
        if len(self.history) >= self.history_limit:
            self.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        self.file.close()

class Device:
    correctors_prefix_start = ("MXY", "MQ")
    solenoids_prefix_start = ("MS", )
    detectors_prefix_start = ("BPM",)
    delimiter_eps = 1e-5
    def __init__(self, path_to_save: tp.Union[str, Path], names: list[str], device_config_path: tp.Union[str, Path],
                 save_history_limit: int = 1000, save_eps: float=1e-3):
        self._names = names
        self._handlers = {name: PV(self._full_name(name)) for name in names}
        self.save_history_limit = save_history_limit
        self.path_to_save = path_to_save
        self.save_eps = save_eps
        self.detector_names, self.solenoid_names, self.corrector_names = self._get_typed_names()

    def _get_typed_names(self):
        detector_names = [name for name in self._names if name.startswith(self.detectors_prefix_start)]
        solenoid_names = [name for name in self._names if name.startswith(self.solenoids_prefix_start)]
        corrector_names = [name for name in self._names if name.startswith(self.correctors_prefix_start)]
        return detector_names, solenoid_names, corrector_names

    def _set_limits(self, device_config_path):
        path = Path(device_config_path) / Path("device_limits.yaml")
        with open(path, "r") as read_file:
            data = yaml.safe_load(read_file)
        self.limits =  data["solenoids"] + data["correctors"]

    def __enter__(self):
        return self

    def _init_saver(self):
        mean_steps = 100
        self.saver = Saver(self.path_to_save, self.save_history_limit, self._names)
        self.parameters = self._get_mean_values(mean_steps)

    def _full_name(self, name: str):
        if re.sub(r"[0-9]", "", name).startswith(self.correctors_prefix_start):
            return f"MSC_{name}_IIN"
        elif re.sub(r"[0-9]", "", name).startswith(self.solenoids_prefix_start):
            return f"MSC_{name}_IIN"
        elif re.sub(r"[0-9]", "", name).startswith(self.detectors_prefix_start):
            return f"BPMS1_{name}"
        else:
            assert False, f"Unidentified prefix in name: {name}"

    def read(self, name: str) -> float:
        value = self._handlers[name].get()
        if value is None:
            value = 0
            raise Warning(f"The handler {name} is None. Set the zero value")
        return value

    def set(self, name: str, value: float):
        if self.limits[name][0] <= value <= self.limits[name][1]:
            self._handlers[name].put(value)
        else:
            raise Warning(f"You are out for the device limits for {name}. You want to set value {value} but limits are {self.limits[name][0]} and {self.limits[name][1]}")

    def _check_updates(self, old_values, new_values):
        flag = False
        for old_value, new_value in zip(old_values.values(), new_values.values()):
            delta = abs(new_value - old_value) / (old_value + self.delimiter_eps)
            if delta >= self.save_eps:
                flag = True
                return flag
        return flag

    def _get_mean_values(self, mean_steps: int):
        values = [0 for name in self._names]
        for step in range(mean_steps):
            values = [values[i] + self.read(name) for i, name in enumerate(self._names)]
        values = {name: values[i] / mean_steps for i, name in enumerate(self._names)}
        return values

    def get_parameters(self, mean_steps: int = 100):
        new_parameters = self._get_mean_values(mean_steps)
        if self._check_updates(self.parameters , new_parameters):
            self.parameters = new_parameters
            self.saver.append(self.parameters)
        return new_parameters

    def get_detectors_mean_std(self, num_points: int = 10):
        detectors_data = np.zeros((num_points, len(self.detector_names)))
        for step in range(num_points):
            parameters = self.get_parameters()
            for i, name in enumerate(self.detector_names):
                detectors_data[step, i] = parameters[name]
        detectors_mean = np.mean(detectors_data, axis=0)
        detectors_std = np.std(detectors_data, axis=0)
        mean_std =\
            {self.detector_names[i]: {"mean": detectors_mean[i], "std": detectors_std[i]}
             for i in range(len(self.detector_names))}
        return mean_std

    def run_continuous_reading(self):
        while True:
            _ = self.get_parameters()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saver.__exit__(exc_type, exc_val, exc_tb)

class DensityGetter:
    possible_formats = ("numpy", "torch")
    def __init__(self, config_path: tp.Union[str, Path], data_format: str = "numpy", seed: int = 42):
        self.density_parameters = self._get_density_parameters(config_path)
        self.generator = self._get_generator(data_format, seed)

    def _get_density_parameters(self, config_path: tp.Union[str, Path]):
        with open(config_path, "r") as read_file:
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

    def _get_numpy_generator(self, maximum: list[float], minimum: list[float], seed: int):
        def generator(size: int):
            return np.random.default_rng(seed).uniform(minimum, maximum, size)
        return  generator

    def _get_torch_generator(self, maximum: list[float], minimum: list[float], seed: int):
        torch.manual_seed(seed)
        minimum = torch.tensor(minimum)
        maximum = torch.tensor(maximum)
        def generator(size: int):
            return (maximum - minimum) * torch.rand((len(maximum), size)) + minimum
        return  generator

    def get_values(self, size: int):
        data = self.generator(size)
        return [dict(zip(self.density_parameters.keys(), value)) for value in data]


class BayessianModel(Model):
    def __init__(self, names: list[str],
                 magnets_meta: dict)
        self.magnets_meta = magnets_meta
        self._names = names


class Agent:
    correctors_prefix_start = ("MXY", "MQ")
    solenoids_prefix_start = ("MS", )
    detectors_prefix_start = ("BPM",)
    def __init__(self, folder_data_path: tp.Union[str, Path], config_folder: tp.Union[str, Path], names: str, model: Model,
                 device: Device):
        self.model = model
        self.density_generator = DensityGetter(config_folder, "numpy", 42)
        self._names = names
        mean_std_path, derivative_steps_path, detector_noize_level = self._get_pathes(config_folder)
        self.mean_std = self._read_mean_std(mean_std_path)
        self._save_meta(folder_data_path, mean_std_path)
        self.detector_names, self.solenoid_names, self.corrector_names = self._get_typed_names()
        self.device = device
        self.derivative_steps = self._get_derivative_step(derivative_steps_path)
        self.detector_noize_level = detector_noize_level
        self.folder_data_path = folder_data_path

    def _get_pathes(self, config_path: tp.Union[str, Path]):
        mean_std_path = Path(config_path) / Path("agent_config/mean_std_turn_1")
        derivative_steps_path = Path(config_path) / Path("agent_config/derivative_steps")
        detector_noize_level = Path(config_path) / Path("device_config/derivative_steps")
        return mean_std_path, derivative_steps_path, detector_noize_level

    def _save_meta(self, folder_path: tp.Union[str, Path], mean_std_path: tp.Union[str, Path]):
        path_time = time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())
        path_time = path_time.replace(":", "_")
        path_mean_std = f"{path_time}_mean_std.yaml"
        path_mean_std = Path(folder_path) / Path(path_mean_std)
        shutil.copy(mean_std_path, path_mean_std)

        model_meta = self.model.metadata()
        path_meta = f"{path_time}_model_meta.yaml"
        with open(path_meta, 'w') as file:
            yaml.dump(model_meta, file, default_flow_style=False)

    def _read_noize_level(self, path: tp.Union[str, Path]):
        with open(path, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def _read_mean_std(self, path: tp.Union[str, Path]):
        with open(path, "r") as read_file:
            data = yaml.safe_load(read_file)
            mean_std = {k: v for d in data.values() for k, v in d.items()}
        return mean_std

    def _normalize(self, values: dict[str: float]):
        norm_values = {}
        for key in values:
            mean_std = self.mean_std[key]
            norm_values[key] = (values[key] - mean_std["mean"]) / mean_std["std"]
        return norm_values

    def _normalize_std(self, values: dict[str: float]):
        norm_values = {}
        for key in values:
            mean_std = self.mean_std[key]
            norm_values[key] = (values[key]) / mean_std["std"]
        return norm_values

    def _denormalize_std(self, values: dict[str: float]):
        denorm_values = {}
        for key in values:
            mean_std = self.mean_std[key]
            denorm_values[key] = mean_std["std"] * values[key]
        return denorm_values

    def _denormalize(self, values: dict[str: float]):
        denorm_values = {}
        for key in values:
            mean_std = self.mean_std[key]
            denorm_values[key] = mean_std["std"] * values[key] + mean_std["mean"]
        return denorm_values

    def _get_typed_names(self):
        detector_names = [name for name in self._names if name.startswith(self.detectors_prefix_start)]
        solenoid_names = [name for name in self._names if name.startswith(self.solenoids_prefix_start)]
        corrector_names = [name for name in self._names if name.startswith(self.correctors_prefix_start)]
        return detector_names, solenoid_names, corrector_names

    def _get_derivative_step(self, path: tp.Union[str, Path]):
        with open(path, "r") as read_file:
            data = yaml.safe_load(read_file)
        return data

    def measure_derivatives(self, detector_names: list[str], solenoid_names: list[str]):
        derivatives = {}
        for solenoid in solenoid_names:
            parameters = self.device.get_parameters()
            detectors_old = [parameters[detector] for detector in detector_names]
            parameters[solenoid] += self.derivative_steps[solenoid]
            self.device.set(solenoid, parameters[solenoid])
            parameters = self.device.get_parameters()
            detectors_new = [parameters[detector] for detector in detector_names]
            derivatives[solenoid] =\
                {detector_names[i]: detectors_new[i] - detectors_old[i] for i in range(len(detector_names))}
        return derivatives

    def start_optimization(self):
        path_time = time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())
        path_time = path_time.replace(":", "_")
        path_start_values = f"{path_time}_start_work.yaml"
        parameters = self._get_full_data()
        with open(path_start_values, 'w') as file:
            yaml.dump(parameters, file, default_flow_style=False)

    def end_optimization(self):
        path_time = time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())
        path_time = path_time.replace(":", "_")
        path_end_values = f"{path_time}_end_work.yaml"
        parameters = self._get_full_data()
        with open(path_end_values, 'w') as file:
            yaml.dump(parameters, file, default_flow_style=False)

    def _get_full_data(self):
        derivatives = self.measure_derivatives(self.detector_names, self.solenoid_names)
        parameters = self.device.get_parameters()
        parameters.update({f"derivative_{key}": value for key, value in derivatives.items()})
        return parameters

    def _read_detectors_noize(self):
        detectors_noize = yaml.safe_load(self.detector_noize_level)
        return detectors_noize

    def _measure_detectors_noize(self, folder_data_path):
        path_time = time.strftime("%Y:%m:%d:%H:%M:%S", time.localtime())
        path_time = path_time.replace(":", "_")
        path_detectors = f"{path_time}_detectors_mean_std.yaml"
        detectors_mean_std = self.device.get_detectors_mean_std(10)
        with open(path_detectors, 'w') as file:
            yaml.dump(detectors_mean_std, file, default_flow_style=False)
        return {name: data["std"] for name, data in detectors_mean_std}


def get_names(path):
    with open(path, "r") as read_file:
        names = yaml.safe_load(read_file)
    return names

def experement_1():
    names = get_names("./configs/device_config/names.yaml")["ordered_data"]
    print(names)
    folder_path = "./data"
    device = Device(names)
    device.run_reading(folder_path)

def experement_2():
    names = get_names("../analysis/names.json")["ordered_data"]
    device = Device(names)
    device.set("1MS1", 5.6)

if __name__ == "__main__":
    experement_1()












