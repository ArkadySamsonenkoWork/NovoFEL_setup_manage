import json
from abc import ABC, abstractmethod
import time
import typing as tp
import sys
import re
import yaml
from pathlib import Path

#from epics import PV
from Servers.Server import PV
import csv


class Model(ABC):
    @abstractmethod
    def __call__(self, values: dict[str: float]) -> dict[str: float]:
        pass

    @abstractmethod
    def add(self, state: dict[str: float]) -> None:
        pass


class Agent:
    def __init__(self, names: str, mean_std_path: str, model: Model):
        self.model = model
        self.names = names
        self.mean_std = self._read_mean_std(mean_std_path)

    def _read_mean_std(self, path: str):
        with open(path, "r") as read_file:
            mean_std = json.load(read_file)
        for name in self.names:
            mean_std.setdefault(name, {"mean": 0.0, "std": 1.0})
        return mean_std

    def _normalize(self, values: dict[str: float]):
        norm_values = {}
        for key in values:
            mean_std = self.mean_std[key]
            norm_values[key] = (values[key] - mean_std["mean"]) / mean_std["std"]
        return norm_values

    def _denormalize(self, values: dict[str: float]):
        denorm_values = {}
        for key in values:
            mean_std = self.mean_std[key]
            denorm_values[key] = mean_std["std"] * values[key] + mean_std["mean"]
        return denorm_values

    def act(self, state: dict[str: float]):
        values = self._normalize(state)
        self.model.add(state)
        action = self.model(values)
        action = self._denormalize(action)
        return action

class Saver:
    def __init__(self, folder_path, history_limit, names: list[str]):
        self.history = []
        self.history_limit = history_limit
        path_time = time.strftime("%Y:%m:%d", time.localtime())
        path_time = path_time.replace(":", "_")
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
    magnets_prefix_start = ("MXY", "MS", "MQ")
    detectors_prefix_start = ("BPM", )
    delimiter_eps = 1e-5
    def __init__(self, path_to_save: tp.Union[str, Path], names: list[str], device_config_path: tp.Union[str, Path],
                 save_history_limit: int = 1000, save_eps: float=1e-3):
        self._names = names
        self._handlers = {name: PV(self._full_name(name)) for name in names}
        self.save_history_limit = save_history_limit
        self.path_to_save = path_to_save
        self.save_eps = save_eps

    def _set_limits(self, device_config_path):
        path = Path(device_config_path) / Path("devise_limits.yaml")
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
        if re.sub(r"[0-9]", "", name).startswith(self.magnets_prefix_start):
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

    def run_continuous_reading(self):
        while True:
            _ = self.get_parameters()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saver.__exit__(exc_type, exc_val, exc_tb)

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












