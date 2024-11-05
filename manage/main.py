import json
from abc import ABC, abstractmethod
import time
import sys
import re
from pathlib import Path

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
        self.writer =  csv.DictWriter(self.file, delimiter=' ', fieldnames=names + ["time"])
        self.writer.writeheader()

    def save(self):
        for data in self.history:
            print(1234)
            self.writer.writerow(data)
        self.history = []

    def append(self, data: dict[str, float | str]):
        history = data.copy()
        current_time = time.strftime("%H:%M:%S", time.localtime())
        history["time"] = current_time
        self.history.append(history)
        if len(data) >= self.history_limit:
            self.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save()
        self.file.close()


class Device:
    magnets_prefix_start = ("MXY", "MS")
    detectors_prefix_start = ("BPM", )
    def __init__(self, names: list[str], setup="manual"):
        self._names = names
        self._handlers = {name: PV(self._full_name(name)) for name in names}
        self.setup = setup

    def _full_name(self, name: str):
        if re.sub(r"[0-9]", "", name).startswith(self.magnets_prefix_start):
            return f"MSC_{name}_IIN"
        elif re.sub(r"[0-9]", "", name).startswith(self.detectors_prefix_start):
            return f"BPMS_{name}"
        else:
            assert False, f"Unidentified prefix in name: {name}"

    def read(self, name: str) -> float:
        return self._handlers[name].get()

    def set(self, name: str, value: float):
        self._handlers[name].put(value)

    def _check_updates(self, old_values, new_values, eps):
        flag = False
        for old_value, new_value in zip(old_values.values(), new_values.values()):
            delta = abs(new_value - old_value) / old_value
            if delta >= eps:
                flag = True
                return flag
        return flag

    def run_reading(self, path, timing=2, eps=1e-2, history_limit=10_000):
        with Saver(path, history_limit, self._names) as saver:
            old_values = {name: self.read(name) for name in self._names}
            saver.append(old_values)
            while True:
                time.sleep(timing)
                new_values = {name: self.read(name) for name in self._names}
                if self._check_updates(old_values, new_values, eps):
                    old_values = new_values
                    saver.append(old_values)

def get_names(path):
    with open(path, "r") as read_file:
        names = json.load(read_file)
    return names

def experement_1():
    names = get_names("../analysis/names.json")["ordered_data"]
    folder_path = "./data"
    device = Device(names)
    device.run_reading(folder_path)

def experement_2():
    names = get_names("../analysis/names.json")["ordered_data"]
    device = Device(names)
    device.set("1MS1", 5.6)

if __name__ == "__main__":
    experement_1()












