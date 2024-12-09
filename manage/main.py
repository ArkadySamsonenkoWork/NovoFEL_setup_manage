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


def yaml_save(folder_data_path: tp.Union[str, Path], name: str, data: dict[str, tp.Any]):
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    path_file = f"{timestamp}_{name}.yaml"
    path_file = Path(folder_data_path) / Path(path_file)
    with open(path_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)


class Saver:
    def __init__(self, folder_data_path, history_limit, names: list[str]):
        self.history = []
        self.history_limit = history_limit
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path_time = f"{timestamp}_full_data"
        path = Path(folder_data_path) / Path(path_time)
        self.file = open(path, 'a')
        self.writer =  csv.DictWriter(self.file, delimiter=' ', fieldnames=names + ["time", "milliseconds"])
        self.writer.writeheader()
        self.start_time_m = time.time() * 1000

    def save(self):
        for data in self.history:
            self.writer.writerow(data)
        self.history = []

    def _get_elapsed_milliseconds(self):
        return self.start_time_m - time.time() * 1000

    def append(self, data: dict[str, tp.Union[float, int]]):
        history = data.copy()
        current_time = time.strftime("%Y_%m_%d_%T", time.localtime())
        history["time"] = current_time
        history["milliseconds"] = self._get_elapsed_milliseconds()
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

    def __init__(self, path_to_save: tp.Union[str, Path], config_path: tp.Union[str, Path], names: list[str],
                 save_history_limit: int = 1000, save_eps: float=1e-3):
        self.names = names
        self.limits = self._load_limits(config_path)
        self._handlers = {name: PV(self._full_name(name)) for name in names}
        self.save_history_limit = save_history_limit
        self.path_to_save = path_to_save
        self.save_eps = save_eps
        self.detector_names, self.solenoid_names, self.corrector_names = self._categorize_names()
        self.saver, self.parameters = self._initialize_saver()

    def __enter__(self):
        return self

    def _categorize_names(self):
        detector_names = [name for name in self.names if re.sub(r"[0-9]", "", name).startswith(self.detectors_prefix_start)]
        solenoid_names = [name for name in self.names if re.sub(r"[0-9]", "", name).startswith(self.solenoids_prefix_start)]
        corrector_names = [name for name in self.names if re.sub(r"[0-9]", "", name).startswith(self.correctors_prefix_start)]
        return detector_names, solenoid_names, corrector_names

    def _load_limits(self, config_path):
        path = Path(config_path) / Path("device_config/device_limits.yaml")
        limits = {}
        with open(path, "r") as read_file:
            data = yaml.safe_load(read_file)
        limits.update(data["solenoids"])
        limits.update(data["correctors"])
        return limits

    def _initialize_saver(self):
        mean_steps = 100
        saver = Saver(self.path_to_save, self.save_history_limit, self.names)
        parameters = self._get_mean_parameters(mean_steps)
        saver.append(parameters)
        return saver, parameters

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
            warnings.warn(f"The handler {name} is None. Set the zero value")
        return value

    def set(self, name: str, value: float):
        if self.limits[name][0] <= value <= self.limits[name][1]:
            self._handlers[name].put(value)
        else:
            warnings.warn(f"You are out for the device limits for {name}. You want to set value {value} but limits are {self.limits[name][0]} and {self.limits[name][1]}")

    def _check_updates(self, old_values, new_values):
        flag = False
        for old_value, new_value in zip(old_values.values(), new_values.values()):
            delta = abs(new_value - old_value) / (old_value + self.delimiter_eps)
            if delta >= self.save_eps:
                flag = True
                return flag
        return flag

    def _get_mean_parameters(self, steps: int):
        total_values = {name: 0.0 for name in self.names}
        for _ in range(steps):
            for name in self.names:
                total_values[name] += self.read(name)
        values = {name: total / steps for name, total in total_values.items()}
        return values

    def get_parameters(self, mean_steps: int = 100):
        new_parameters = self._get_mean_parameters(mean_steps)
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
            {self.detector_names[i]: {"mean": detectors_mean[i].item(), "std": detectors_std[i].item()}
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
        path = Path(config_path) / Path("agent_configs/densities/uniform_density.yaml")
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


class Agent:
    correctors_prefix_start = ("MXY", "MQ")
    solenoids_prefix_start = ("MS", )
    detectors_prefix_start = ("BPM",)
    def __init__(self, folder_data_path: tp.Union[str, Path], config_folder: tp.Union[str, Path], names: str, model: Model,
                 device: Device):
        self.model = model
        self.density_generator = DensityGetter(config_folder, "torch", 42)
        self.names = names
        self.folder_data_path = folder_data_path

        mean_std_path, derivative_steps_path, detector_noize_path = self._get_pathes(config_folder)
        self.mean_std = self._read_mean_std(mean_std_path)
        self._save_meta(folder_data_path, config_folder)
        self.detector_names, self.solenoid_names, self.corrector_names = self._get_typed_names()
        self.device = device
        self.derivative_steps = self._get_derivative_steps(derivative_steps_path)
        self.derivative_steps_norm = self._denormalize_std(self.derivative_steps)
        self.detector_noize_level = detector_noize_path

    def _get_pathes(self, config_path: tp.Union[str, Path]):
        mean_std_path = Path(config_path) / Path("agent_configs/mean_std_turn_1.yaml")
        derivative_steps_path = Path(config_path) / Path("agent_configs/derivative_steps.yaml")
        detector_noize_level = Path(config_path) / Path("device_configs/detector_noize_level.yaml")
        return mean_std_path, derivative_steps_path, detector_noize_level

    def _save_meta(self, folder_path: tp.Union[str, Path], config_path: tp.Union[str, Path]):
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        updated_config_path = f"{timestamp}_config_folder"
        updated_config_path = Path(folder_path) / Path(updated_config_path)
        shutil.copytree(config_path, updated_config_path)

        model_meta = self.model.metadata()
        path_meta = f"{timestamp}_model_meta.yaml"
        path_meta = Path(folder_path) / Path(path_meta)
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
        detector_names = [name for name in self.names if re.sub(r"[0-9]", "", name).startswith(self.detectors_prefix_start)]
        solenoid_names = [name for name in self.names if re.sub(r"[0-9]", "", name).startswith(self.solenoids_prefix_start)]
        corrector_names = [name for name in self.names if re.sub(r"[0-9]", "", name).startswith(self.correctors_prefix_start)]
        return detector_names, solenoid_names, corrector_names

    def _get_derivative_steps(self, path: tp.Union[str, Path]):
        with open(path, "r") as read_file:
            steps = yaml.safe_load(read_file)
        return steps

    def _get_parameters_difference(self, detector_names: list[str], solenoid: str):
        parameters = self.device.get_parameters()
        detectors_old = [parameters[detector] for detector in detector_names]
        parameters[solenoid] += self.derivative_steps[solenoid]
        self.device.set(solenoid, parameters[solenoid])

        parameters = self.device.get_parameters()
        detectors_new = [parameters[detector] for detector in detector_names]
        parameters[solenoid] -= self.derivative_steps[solenoid]
        self.device.set(solenoid, parameters[solenoid])

        return detectors_old, detectors_new

    def get_full_data(self):
        derivatives = self.measure_derivatives(self.detector_names, self.solenoid_names)
        parameters = self.device.get_parameters()
        parameters.update({f"derivative_{key}": value for key, value in derivatives.items()})
        return parameters

    def read_detectors_noize(self):
        detectors_noize = yaml.safe_load(self.detector_noize_level)
        return detectors_noize

    def measure_detectors_noize(self, times: int=10):
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path_detectors = f"{timestamp}_detectors_mean_std.yaml"
        path_detectors = Path(self.folder_data_path) / Path(path_detectors)
        detectors_mean_std = self.device.get_detectors_mean_std(times)
        with open(path_detectors, 'w') as file:
            yaml.dump(detectors_mean_std, file, default_flow_style=False)
        return {name: data["std"] for name, data in detectors_mean_std.items()}

    def measure_derivatives(self, detector_names: list[str], solenoid_names: list[str]):
        derivatives = {}
        for solenoid in solenoid_names:
            detectors_old, detectors_new = self._get_parameters_difference(detector_names, solenoid)
            derivatives[solenoid] =\
                {detector_names[i]: (detectors_new[i] - detectors_old[i]) / self.derivative_steps[solenoid]
                 for i in range(len(detector_names))
                 }
        return derivatives

    def measure_norm_derivatives(self, detector_names: list[str], solenoid_names: list[str]):
        derivatives = {}
        for solenoid in solenoid_names:
            detectors_old, detectors_new = self._get_parameters_difference(detector_names, solenoid)
            derivatives[solenoid] = \
                {detector_names[i]: (detectors_new[i] - detectors_old[i]) / self.derivative_steps_norm[solenoid]
                 for i in range(len(detector_names))
                 }
            derivatives[solenoid] = self._normalize_std(derivatives[solenoid])
        return derivatives

    def measure_differences(self, detector_names: list[str], solenoid_names: list[str]):
        differences = {}
        for solenoid in solenoid_names:
            detectors_old, detectors_new = self._get_parameters_difference(detector_names, solenoid)
            differences[solenoid] = \
                {detector_names[i]: (detectors_new[i] - detectors_old[i])
                 for i in range(len(detector_names))}
        return differences

    def save_full_data(self, name: str):
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        path_values = f"{timestamp}_{name}.yaml"
        path_values = Path(self.folder_data_path) / Path(path_values)
        parameters = self.get_full_data()
        with open(path_values, 'w') as file:
            yaml.dump(parameters, file, default_flow_style=False)

    def _measure_execution_time(self, funct: tp.Callable, *args):
        start_time = time.time()
        out = funct(*args)
        end_time = time.time()
        return out, end_time - start_time

    def _save_differences_solenoids(self,
                                    subfolder: tp.Union[str,Path],
                                    increments: list[float],
                                    differences: dict[int, dict[str, tp.Any]],
                                    derivatives: dict[int, dict[str, tp.Any]],
                                    solenoid_steps: dict[int, dict[str, tp.Any]]
                                    ):
        font = {'size': 6}
        matplotlib.rc('font', **font)
        solenoid_names = list(list(solenoid_steps.values())[0].keys())
        tot_subplots = len(solenoid_names)

        i = 0
        for plot_num in range(2):
            fig_solenoids, axes = plt.subplots(tot_subplots // 4, 2)
            for axis in itertools.chain(*axes):
                steps = [solenoid_steps[increment][solenoid_names[i]] for increment in increments]
                axis.scatter(increments, steps)
                axis.set_title(solenoid_names[i])
                i += 1
            path = Path(subfolder) / Path(f"solenoids_vs_increments_{plot_num}.png")
            fig_solenoids.savefig(path, pad_inches=0.0, dpi=200)

        i = 0
        for plot_num in range(2):
            fig_solenoids, axes = plt.subplots(tot_subplots // 4, 2)
            for axis in itertools.chain(*axes):
                differences_detectors = [differences[increment][solenoid_names[i]] for increment in increments]
                steps = [solenoid_steps[increment][solenoid_names[i]] for increment in increments]
                for detector in self.detector_names:
                    differences_detector = [data[detector] for data in differences_detectors]
                    axis.scatter(steps, differences_detector, label=detector)
                axis.set_title(solenoid_names[i])
                axis.legend()
                i += 1
            path = Path(subfolder) / Path(f"differences_vs_solenoids_{plot_num}.png")
            fig_solenoids.savefig(path, pad_inches=0.0, dpi=200)

        i = 0
        for plot_num in range(2):
            fig_solenoids, axes = plt.subplots(tot_subplots // 4, 2)
            for axis in itertools.chain(*axes):
                derivatives_detectors = [derivatives[increment][solenoid_names[i]] for increment in increments]
                steps = [solenoid_steps[increment][solenoid_names[i]] for increment in increments]
                for detector in self.detector_names:
                    differences_detector = [data[detector] for data in derivatives_detectors]
                    axis.scatter(steps, differences_detector, label=detector)
                axis.set_title(solenoid_names[i])
                axis.legend()
                i += 1
            path = Path(subfolder) / Path(f"derevatieves_vs_solenoids_{plot_num}.png")
            fig_solenoids.savefig(path, pad_inches=0.0, dpi=200)


    def measure_differences_solenoids(self, subfolder_path: tp.Union[str, Path],
                                      times: dict[str, tp.Any], meta: dict[str, tp.Any]):
        increments = [1.4, 1.2, 1.0, 0.8, 0.6, 0.4, 0.2]
        start_steps = self.derivative_steps.copy()
        differences = {}
        derivatives = {}
        times["differences"] = {}
        times["derivatives"] = {}
        solenoid_steps = {}
        for increment in increments:
            self.derivative_steps = {key: value * increment for key, value in start_steps.items()}
            differences[increment], timedel = self._measure_execution_time(
                self.measure_differences, self.detector_names, self.solenoid_names
            )
            times["differences"][increment] = timedel
            derivatives[increment], timedel = self._measure_execution_time(
                self.measure_derivatives, self.detector_names, self.solenoid_names
            )
            times["derivatives"][increment] = timedel
            solenoid_steps[increment] = self.derivative_steps.copy()

        meta["differences"] = differences
        meta["derivatives"] = derivatives
        meta["solenoid_steps"] = solenoid_steps
        self._save_differences_solenoids(subfolder_path, increments, differences, derivatives, solenoid_steps)
        return meta, times


    def measure_hyper_parameters(self):
        meta = {}
        times = {}

        subfolder_path = Path(self.folder_data_path) / Path(f"measured_meta_hyperparameters")
        subfolder_path.mkdir(mode=0o777, parents=False, exist_ok=True)

        detector_noize, timedel = self._measure_execution_time(self.measure_detectors_noize, 10)
        meta["detector_noize"] = detector_noize
        times["detector_noize"] = timedel
        full_data, timedel = self._measure_execution_time(self.get_full_data)
        times["full_data"] = timedel
        meta["full_data"] = full_data
        meta, times = self.measure_differences_solenoids(subfolder_path, times, meta)
        meta["times"] = times

        names = ["times", "full_data", "detector_noize", "differences", "derivatives", "solenoid_steps"]
        for name in names:
            yaml_save(subfolder_path, name, meta[name])

def get_subfolder_path(folder_path: tp.Union[str, Path]):
    data_time = time.strftime("%Y_%m_%d", time.localtime())
    subfolder_path = Path(folder_path) / Path(data_time)
    subfolder_path.mkdir(mode=0o777, parents=False, exist_ok=True)
    run = 1
    while True:
        subfolder_path_run = subfolder_path / Path(f"run_{run}")
        if Path.exists(subfolder_path_run):
            run += 1
        else:
            subfolder_path_run.mkdir(mode=0o777, parents=False, exist_ok=False)
            break
    return subfolder_path_run

def get_names(path):
    with open(path, "r") as read_file:
        names = yaml.safe_load(read_file)
    return names

def experement_1():
    config_path = "./configs/"
    names = get_names("./configs/device_config/names.yaml")["ordered_data"]
    folder_path = get_subfolder_path("./data")
    with Device(folder_path, config_path, names) as device:
        parameters = device.get_parameters()
        model = BayessianModel(names)
        agent = Agent(folder_path, config_path, names, model, device)
        agent.measure_detectors_noize()
        agent.save_full_data("start")
        agent.save_full_data("end")
        density = agent.density_generator.get_values(100)
        agent.measure_hyper_parameters()
        print(density[0])


def experement_2():
    names = get_names("../analysis/names.json")["ordered_data"]
    device = Device(names)
    device.set("1MS1", 5.6)

if __name__ == "__main__":
    experement_1()












