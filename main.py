import json
from abc import ABC, abstractmethod

from epics import PV


class Model(ABC):
    @abstractmethod
    def __call__(self, values: dict[str: float]) -> dict[str: float]:

    @abstractmethod
    def add(self, state: dict[str: float]) -> None:



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

class Device:
    def __init__(self, names: str):
        self._names = names
        self._handlers = {name: PV(self._full_name(name)) for name in names}

    def _full_name(self, name: str):
        return f"MSC_{name}_IIN"

    def read(self, name: str) -> float:
        return self._handlers[name].get()

    def set(self, name: str, value: float):
        self._handlers[name].put(value)


