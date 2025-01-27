import csv
import re
import os
import time
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import yaml

#from epics import PV
from Servers.Server import PV
from plot_app import PlotApp
import utils

#os.environ['EPICS_CA_ADDR_LIST'] = '192.168.3.92'

def save_start_time(folder_data_path: tp.Union[str, Path]):
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    data = {"start_time": timestamp}
    utils.yaml_save(folder_data_path, "start_time", data)

def save_finish_time(folder_data_path: tp.Union[str, Path]):
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    data = {"finish_time": timestamp}
    utils.yaml_save(folder_data_path, "finish_time", data)

class DataSaver:
    DELIMITER_EPS = 1e-5
    def __init__(self, folder_data_path, history_limit, names: list[str], save_eps: float) -> None:
        self.history = []
        self.history_limit = history_limit
        path_time = f"full_data"
        path = Path(folder_data_path) / Path(path_time)
        self.folder_data_path = folder_data_path
        self.file = open(path, 'a')
        self.writer =  csv.DictWriter(self.file, delimiter=' ', fieldnames=names + ["time", "milliseconds"])
        self.writer.writeheader()
        self.start_time_m = time.time() * 1000
        save_start_time(folder_data_path)
        self.parameters = {name: 0.0 for name in names}
        self.save_eps = save_eps

    def save(self):
        for data in self.history:
            self.writer.writerow(data)
        self.history = []

    def _get_elapsed_milliseconds(self):
        return self.start_time_m - time.time() * 1000

    def _check_updates(self, old_values, new_values):
        flag = False
        for old_value, new_value in zip(old_values.values(), new_values.values()):
            delta = abs(new_value - old_value) / (old_value + self.DELIMITER_EPS)
            if delta >= self.save_eps:
                flag = True
                return flag
        return flag

    def append(self, parameters: dict[str, tp.Union[float, int]]):
        if self._check_updates(self.parameters, parameters):
            self.parameters = parameters
            history = parameters.copy()
            current_time = time.strftime("%Y_%m_%d_%T", time.localtime())
            history["time"] = current_time
            history["milliseconds"] = self._get_elapsed_milliseconds()
            self.history.append(history)
            if len(self.history) >= self.history_limit:
                self.save()
        else:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        save_finish_time(self.folder_data_path)
        self.save()
        self.file.close()

class LaserSetup:
    DETECTOR_PREFIXES = ("BPM",)
    SOLENOID_PREFIXES = ("MS",)
    CORRECTOR_PREFIXES = ("MXY", "MQ")

    def __init__(self, folder_data_path: tp.Union[str, Path], config_path: tp.Union[str, Path], element_names: list[str],
                 mean_steps: int = 1, save_history_limit: int = 1000, save_eps: float=1e-3, plot_figs=False,
                 read_every: float = 0.2):
        self.read_every = read_every
        self.mean_steps = mean_steps
        self.current_time = time.time()
        self.element_names = element_names
        self.parameters = {name: 0.0 for name in element_names}

        self.limits = self._load_limits(config_path)
        self._handlers = {name: PV(self._full_name(name)) for name in element_names}
        self.save_history_limit = save_history_limit
        self.folder_data_path = folder_data_path
        self.detector_names, self.solenoid_names, self.corrector_names = utils.get_typed_names(self.element_names)
        utils.yaml_save(
            self.folder_data_path,  "reading_hyperparameters",
            {"mean_steps": self.mean_steps, "read_every": read_every}
        )
        self.saver, self.parameters = self._initialize_saver(save_eps)
        self.plot_app = PlotApp(element_names) if plot_figs else None

    def _full_name(self, name: str):
        def _filter_name(prefixes: tuple[str]):
            return re.sub(r"[0-9]", "", name).startswith(prefixes)
        if _filter_name(self.CORRECTOR_PREFIXES):
            return f"MSC_{name}_IIN"
        elif _filter_name(self.SOLENOID_PREFIXES):
            return f"MSC_{name}_IIN"
        elif _filter_name(self.DETECTOR_PREFIXES):
            return f"BPMS1_{name}"
        else:
            assert False, f"Unidentified prefix in name: {name}"


    def _load_limits(self, config_path):
        path = Path(config_path) / Path("device_configs/device_limits.yaml")
        limits = {}
        with open(path, "r") as read_file:
            data = yaml.safe_load(read_file)
        limits.update(data["solenoids"])
        limits.update(data["correctors"])
        return limits

    def _initialize_saver(self, save_eps: float):
        saver = DataSaver(self.folder_data_path, self.save_history_limit, self.element_names, save_eps)
        parameters = self._get_mean_parameters(self.mean_steps)
        saver.append(parameters)
        return saver, parameters

    def _wait_time(self):
        delta_time = (self.current_time - time.time())
        if delta_time < self.read_every:
            time.sleep(self.read_every - delta_time)
            self.current_time = time.time()

    def _read(self, name: str) -> float:
        value = self._handlers[name].get()
        if value is None:
            value = 0
            warnings.warn(f"The handler {name} is None. Set the zero value")
        return value

    def set(self, name: str, value: float):
        if self.limits[name][0] <= value <= self.limits[name][1]:
            self._handlers[name].put(value)
        else:
            warnings.warn(f"You are out for the device limits for {name}. You want to set value {value} but limits are {self.limits[name][0]} and {self.limits[name][1]}")

    def _read_parameters_set(self):
        self._wait_time()
        parameters = {name: self._read(name) for name in self.element_names}
        while parameters == self.parameters:
            time.sleep(self.read_every / 2)
            parameters = {name: self._read(name) for name in self.element_names}
        self.parameters = parameters
        return self.parameters

    def _get_mean_parameters(self, steps: int):
        total_values = {name: 0.0 for name in self.element_names}
        for _ in range(steps):
            new_parameters = self._read_parameters_set()
            for name in self.element_names:
                total_values[name] += new_parameters[name]
        parameters = {name: total / steps for name, total in total_values.items()}
        return parameters

    def get_parameters(self):
        new_parameters = self._get_mean_parameters(self.mean_steps)
        self.saver.append(self.parameters)
        if self.plot_app is not None:
            self.plot_app.update(new_parameters)
        return new_parameters

    def get_detectors_mean_var(self, num_points: int = 10):
        detectors_data = np.zeros((num_points, len(self.detector_names)))
        for step in range(num_points):
            parameters = self.get_parameters()
            for i, name in enumerate(self.detector_names):
                detectors_data[step, i] = parameters[name]
        detectors_mean = np.mean(detectors_data, axis=0)
        detectors_var = np.var(detectors_data, axis=0, ddof=1)
        mean_var =\
            {self.detector_names[i]: {"mean": detectors_mean[i].item(), "var": detectors_var[i].item()}
             for i in range(len(self.detector_names))}
        return mean_var

    def run_continuous_reading(self):
        while True:
            _ = self.get_parameters()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saver.__exit__(exc_type, exc_val, exc_tb)