import itertools
import shutil
import time
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import yaml
# from altair.vega import legend
from matplotlib import pyplot as plt
import torch

import models
import devices
import utils

from manage.models import MultiOutputModel


class ConfigPaths:
    AGENT_CONFIG_FOLDER = "agent_configs"
    DEVICE_CONFIG_FOLDER = "device_configs"

    @staticmethod
    def get_agent_paths(config_path: tp.Union[str, Path]):
        config_path = Path(config_path)
        return (
            config_path / f"{ConfigPaths.AGENT_CONFIG_FOLDER}/mean_var_turn_1.yaml",
            config_path / f"{ConfigPaths.AGENT_CONFIG_FOLDER}/derivative_steps.yaml",
            config_path / f"{ConfigPaths.DEVICE_CONFIG_FOLDER}/detector_noize_var.yaml",
            config_path / f"{ConfigPaths.AGENT_CONFIG_FOLDER}/correctors_bounds.yaml",
        )


class DataHandler:
    @staticmethod
    def read_yaml(path: tp.Union[str, Path]):
        with open(path, "r") as file:
            return yaml.safe_load(file)

    @staticmethod
    def save_yaml(folder: tp.Union[str, Path], filename: str, data: dict):
        utils.yaml_save(folder, filename, data)

    @staticmethod
    def copy_config(source: tp.Union[str, Path], destination: tp.Union[str, Path]):
        shutil.copytree(source, destination)


class Plotter:
    @staticmethod
    def save_scatter_plots(steps, data, labels, title, save_path):
        fig, axes = plt.subplots(len(labels), 1)
        for axis, label in zip(axes, labels):
            axis.scatter(steps, data[label], label=label)
            axis.legend()
        fig.suptitle(title)
        fig.savefig(save_path, pad_inches=0.0, dpi=200)

    @staticmethod
    def save_solenoid_increments(solenoid_names: list[str], steps: dict[float, dict[str, float]],
                                 increments: list[float], save_path: Path):
        solenoid_num = 0
        for plot_num in range(2):
            fig, axes = plt.subplots(2, 2)
            for axis in itertools.chain(*axes):
                if solenoid_num < len(solenoid_names):
                    solenoid_name = solenoid_names[solenoid_num]
                    axis.scatter(increments, [steps[inc][solenoid_name] for inc in increments], label=solenoid_name)
                    axis.legend()
                    solenoid_num += 1
            fig.savefig(f"{save_path}_{plot_num}", pad_inches=0.0, dpi=200)

    @staticmethod
    def save_solenoid_plot(folder: Path, solenoid_name: str, plot_name: str, increments: list[float], steps, solenoid_data):
        solenoid_folder = folder / f"solenoid_{solenoid_name}"
        solenoid_folder.mkdir(parents=True, exist_ok=True)
        Plotter.save_scatter_plots(
            [steps[inc][solenoid_name] for inc in increments],
            {det: [dif[det] for dif in solenoid_data.values()] for det in solenoid_data[increments[0]]},
            solenoid_data[increments[0]].keys(),
            f"{plot_name}: {solenoid_name}",
            solenoid_folder / f"{plot_name}_{solenoid_name}.png",
        )

    @staticmethod
    def save_solenoid_plots(folder: Path, solenoid_name: str, increments: list[float], steps, differences: dict, derivatives:dict):
        Plotter.save_solenoid_plot(folder, solenoid_name, "differences", increments, steps, differences)
        Plotter.save_solenoid_plot(folder, solenoid_name, "derivatives", increments, steps, derivatives)

class Normalizer:
    def __init__(self, mean_std: dict):
        self.mean_std = mean_std

    def normalize(self, values: dict, mode: str = "full") -> dict:
        result = {}
        for key, value in values.items():
            stats = self.mean_std.get(key, {"mean": 0, "std": 1})
            if mode == "full":
                result[key] = (value - stats["mean"]) / stats["std"]
            elif mode == "std":
                result[key] = value / stats["std"]
            elif mode == "denormalize_std":
                result[key] = value * stats["std"]
            elif mode == "denormalize_full":
                result[key] = value * stats["std"] + stats["mean"]
        return result

class DeviceMeasurer:
    def __init__(self, element_names: list[str], device: devices.LaserSetup,
                 folder: tp.Union[str, Path], mean_var_path: tp.Union[Path, str], derivative_steps: dict[float]):
        self.device = device
        self.folder = folder
        self.detector_names, self.solenoid_names, self.corrector_names = utils.get_typed_names(element_names)
        self.mean_var = DataHandler.read_yaml(mean_var_path)
        self.normalizer = Normalizer(self.mean_var)
        self.derivative_steps = derivative_steps

    def measure_noize(self, times: int):
        detectors_mean_var = self.device.get_detectors_mean_var(times)
        DataHandler.save_yaml(self.folder, "detectors_mean_var", detectors_mean_var)
        return {name: {"mean": data["mean"], "var": data["var"]} for name, data in detectors_mean_var.items()}

    def _get_param_diffs(self, solenoid):
        params = self.device.get_parameters()
        old_vals = [params[det] for det in self.detector_names]

        params[solenoid] += self.derivative_steps[solenoid]
        self.device.set(solenoid, params[solenoid])

        new_vals = [self.device.get_parameters()[det] for det in self.detector_names]
        params[solenoid] -= self.derivative_steps[solenoid]
        self.device.set(solenoid, params[solenoid])

        return old_vals, new_vals

    def _measure_normalized(self, solenoid_names: list[str], measure_function: tp.Callable):
        raw = measure_function(solenoid_names)
        return {
            solenoid: self.normalizer.normalize(raw[solenoid], mode="std")
            for solenoid in raw
        }

    def _measure_derivatives(self, solenoid_names: list[str]):
        derivatives = {}
        for solenoid in solenoid_names:
            old_vals, new_vals = self._get_param_diffs(solenoid)
            derivatives[solenoid] = {
                det: (new - old) / self.derivative_steps[solenoid]
                for det, old, new in zip(self.detector_names, old_vals, new_vals)
            }
        return derivatives

    def _measure_normalized_derivatives(self, solenoid_names: list[str]):
        return self._measure_normalized(solenoid_names, self._measure_derivatives)

    def _measure_differences(self, solenoid_names: list[str]):
        differences = {}
        for solenoid in solenoid_names:
            old_vals, new_vals = self._get_param_diffs(solenoid)
            differences[solenoid] = {
                det: new - old
                for det, old, new in zip(self.detector_names, old_vals, new_vals)
            }
        return differences

    def _measure_normalized_differences(self, solenoid_names: list[str]):
        return self._measure_normalized(solenoid_names, self.measure_differences)

    def measure_shifts(self, solenoid_names: list[str], derivative: bool = True, normalized: bool=False):
        if derivative:
            return self._measure_normalized_derivatives(solenoid_names) if normalized else self._measure_derivatives(solenoid_names)
        else:
            return self._measure_normalized_differences(solenoid_names) if normalized else self._measure_differences(solenoid_names)

    def measure_parameters(self):
        return self.device.get_parameters()

    def set_parameters(self, parameters: dict[str, float]):
        for name, parameter in parameters.items():
            self.device.set(name, parameter)

    def measure_steps_dependance(self):
        increments = [0.2 * i for i in range(1, 8)]
        differences = {}
        derivatives = {}
        steps = {}
        init_derivative_steps = self.derivative_steps.copy()
        for increment in increments:
            self.derivative_steps = {solenoid: val * increment for solenoid, val in
                                              init_derivative_steps.items()}
            differences[increment] = self.measure_shifts(self.solenoid_names, derivative=False)
            derivatives[increment] = self.measure_shifts(self.solenoid_names, derivative=True)
            steps[increment] = {sol: init_derivative_steps[sol] * increment for sol in self.solenoid_names}
        self.derivative_steps = init_derivative_steps
        return increments, steps, differences, derivatives

    def _measure_execution_time(self, funct: tp.Callable, *args, **kwargs):
        start_time = time.time()
        out = funct(*args, **kwargs)
        end_time = time.time()
        return out, end_time - start_time

    def measure_full_data(self, include_timing: bool = True):
        if include_timing:
            parameters, parameters_time = self._measure_execution_time(self.device.get_parameters)
            derivatives, derivatives_time = self._measure_execution_time(self.measure_shifts,
                                                                         self.solenoid_names, derivative=True)
            meta_time = {"parameters_time": parameters_time, "derivatives_time": derivatives_time}
        else:
            parameters = self.device.get_parameters()
            derivatives = self.measure_shifts(self.solenoid_names, derivative=True)
            meta_time = None

        derivatives = {f"derivative_{key}": value for key, value in derivatives.items()}

        meta_data = {}
        meta_data.update(derivatives)
        meta_data.update(parameters)
        meta = {"data": meta_data, "executions_time": meta_time}
        return meta

class Agent:
    def __init__(self, folder_data_path: tp.Union[str, Path], config_folder: tp.Union[str, Path],
                 element_names: str, device: devices.LaserSetup):
        self.folder_data_path = Path(folder_data_path)
        self.element_names = element_names
        self.config_folder = Path(config_folder)
        self.detector_names, self.solenoid_names, self.corrector_names = utils.get_typed_names(element_names)

        (mean_var_path, derivative_steps_path,
         detector_noize_path, correctors_bounds_path) = ConfigPaths.get_agent_paths(config_folder)

        self.detector_noize_var = DataHandler.read_yaml(detector_noize_path)
        self.correctors_bounds = DataHandler.read_yaml(correctors_bounds_path)

        DataHandler.copy_config(config_folder, self.folder_data_path / "config_folder")
        derivative_steps = DataHandler.read_yaml(derivative_steps_path)
        self.measurer = DeviceMeasurer(element_names, device, folder_data_path, mean_var_path, derivative_steps)
        self.init_corrector_values(["1MXY1_Y"])

    def _generate_x_init_points(self, bounds:list):
        bounds = [[mn / 2, mx / 2] for mn, mx in bounds] # give
        center = [(mn + mx) / 2 for mn, mx in bounds]
        corners = itertools.product(*bounds)
        points = [list(corner) for corner in corners]
        return [center] + points

    def _set_and_get_parameters(self, update_parameters):
        self.measurer.set_parameters(update_parameters)
        new_parameters = self.measurer.measure_parameters()
        update_values = list(update_parameters.values())
        updated_values = [new_parameters[name] for name in update_parameters.keys()]
        if not np.allclose(update_values, updated_values, atol=1e-03):
            warnings.warn(f"old correctors and new_correctors values are not equal. The old values are {update_values}"
                                    f"the new values are {updated_values}")
        return new_parameters

    def _get_stack_points(self, parameters_stack):
        Y = []
        for update_parameters in parameters_stack:
            parameters = self._set_and_get_parameters(update_parameters)
            detectors = {name: parameters[name] for name in self.detector_names}
            Y.append(list(detectors.values()))
        return Y

    def init_corrector_values(self, correctors_name: list[str]):
        dtype = torch.float64
        Yvar = torch.tensor(list(self.detector_noize_var.values()), dtype=dtype)
        bounds = [
            [self.correctors_bounds[name]["min"],  self.correctors_bounds[name]["max"]]
                  for name in correctors_name
        ]
        bounds_t = torch.tensor(bounds, dtype=dtype).transpose(0, 1)

        parameters = self.measurer.measure_parameters()
        correctors = {name: parameters[name] for name in correctors_name}
        detectors = {name: parameters[name] for name in self.detector_names}

        x = list(correctors.values())
        y = list(detectors.values())
        _init_points = self._generate_x_init_points(bounds)
        parameters_stack = [{name: value for name, value in zip(correctors_name, _init_point)}
                            for _init_point in _init_points]

        X = torch.tensor([x] + _init_points, dtype=dtype)
        stack_points = self._get_stack_points(parameters_stack)
        Y = torch.tensor([y] + stack_points, dtype=dtype)
        gp_model = models.MultiListModel(train_X=X, train_Y=Y, Yvar=Yvar, bounds=bounds_t,
                                         weights=torch.tensor([1, 1, 0])
                                         )
        DataHandler.save_yaml(self.folder_data_path, "model_meta", gp_model.__repr__())
        gp_model, X_stat, Y_stat =\
            self.optimize_corrector_values(correctors_name, gp_model, bounds=bounds_t, steps=10)
        X_stat = [x] + _init_points + X_stat
        Y_stat = [y] + stack_points + Y_stat
        best_X = gp_model.best_X.tolist()
        if best_X != X_stat[-1]:
            parameters_stack = [{name: value for name, value in zip(correctors_name, X)} for X in [best_X]]
            new_Y = self._get_stack_points(parameters_stack)
        Y_stat += new_Y
        X_stat.append(best_X)

    def optimize_corrector_values(self, correctors_name: list[str],
                                  gp_model: MultiOutputModel, bounds, steps: int = 12):
        gp_model.train()
        retrain = 2
        alphas = [i / 10 for i in range(steps-2)] + [1.0] * 2
        X_stat = []
        Y_stat = []
        for step, alpha in enumerate(alphas):
            if step % retrain == 0:
                gp_model.train()
            new_X = gp_model.get_new_candidate_point(bounds=bounds, alpha=alpha).tolist()
            X_stat += new_X
            parameters_stack = [{name: value for name, value in zip(correctors_name, X)} for X in new_X]
            new_Y = self._get_stack_points(parameters_stack)
            Y_stat += new_Y
            X = torch.tensor(new_X)
            Y = torch.tensor(new_Y)
            gp_model.add_training_points(X, Y)
        return gp_model, X_stat, Y_stat

    def measure_noize(self):
        subfolder_path = self.folder_data_path / "measurements_noize"
        subfolder_path.mkdir(parents=True, exist_ok=True)
        noize = self.measurer.measure_noize(times=20)
        self.save_measurements(subfolder_path, "noize", noize)

    def measure_hyperparameters(self):
        subfolder_path = self.folder_data_path / "measurements_hyperparameters"
        subfolder_path.mkdir(parents=True, exist_ok=True)
        increments, steps, differences, derivatives = self.measurer.measure_steps_dependance()
        noize = self.measurer.measure_noize(times=20)
        full_data = self.measurer.measure_full_data(include_timing=True)
        meta = {"full_data": full_data, "noize": noize,
                "increments": {"increments": increments},
                "steps": steps, "differences": differences, "derivatives": derivatives}
        self.save_plot_measurements(subfolder_path, increments, steps, differences, derivatives)
        for key, value in meta.items():
            self.save_measurements(subfolder_path, key, value)

    def save_measurements(self, subfolder_path, name: str, data: dict[str, tp.Any]):
        DataHandler.save_yaml(subfolder_path, name, data)

    def save_plot_measurements(self, subfolder: Path, increments: list[float], steps: dict,
                          differences: dict, derivatives: dict):
        subfolder_path = subfolder / "plots"
        subfolder_path.mkdir(parents=True, exist_ok=True)
        for solenoid in self.solenoid_names:
            differences_solenoid = {increment: differences[increment][solenoid] for increment in increments}
            derivatives_solenoid = {increment: derivatives[increment][solenoid] for increment in increments}
            Plotter.save_solenoid_plots(subfolder_path, solenoid, increments, steps, differences_solenoid, derivatives_solenoid)
        path = subfolder_path / "steps_increments"
        Plotter.save_solenoid_increments(self.solenoid_names, steps, increments, path)