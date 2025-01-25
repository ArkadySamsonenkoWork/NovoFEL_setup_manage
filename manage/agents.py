import itertools
import shutil
import time
import typing as tp
import warnings
from pathlib import Path

import numpy as np
import math
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
            config_path / f"{ConfigPaths.DEVICE_CONFIG_FOLDER}/detector_noise_var.yaml",
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
    def save_solenoid_plots(folder: Path, solenoid_name: str, increments: list[float], steps: dict, differences: dict, derivatives: dict):
        Plotter.save_solenoid_plot(folder, solenoid_name, "differences", increments, steps, differences)
        Plotter.save_solenoid_plot(folder, solenoid_name, "derivatives", increments, steps, derivatives)

class Normalizer:
    def __init__(self, mean_var: dict):
        self.mean_var = mean_var

    def normalize(self, values: dict, mode: str = "full") -> dict:
        result = {}
        for key, value in values.items():
            stats = self.mean_var.get(key, {"mean": 0, "var": 1})
            if mode == "full":
                result[key] = (value - stats["mean"]) / stats["var"] ** 0.5
            elif mode == "std":
                result[key] = value / stats["var"] ** 0.5
            elif mode == "denormalize_std":
                result[key] = value * stats["var"] ** 0.5
            elif mode == "denormalize_full":
                result[key] = value * stats["var"] ** 0.5 + stats["mean"]
        return result


ParameterDict = dict[str, float]
DerivativeSteps = dict[str, float]

class DeviceMeasurer:
    """Measures and analyzes device parameters, noise, and system responses for a laser setup."""

    def __init__(self, element_names: list[str], device: devices.LaserSetup,
                 folder: tp.Union[str, Path], mean_var_path: tp.Union[Path, str], derivative_steps: dict[float]):
        self.element_names = element_names
        self.device = device
        self.folder = folder
        self.detector_names, self.solenoid_names, self.corrector_names = utils.get_typed_names(element_names)
        self.mean_var = DataHandler.read_yaml(mean_var_path)
        self.normalizer = Normalizer(self.mean_var)
        self.derivative_steps = derivative_steps

    def measure_noise(self, times: int):
        detectors_mean_var = self.device.get_detectors_mean_var(times)
        DataHandler.save_yaml(self.folder, "detectors_mean_var", detectors_mean_var)
        return {name: {"mean": data["mean"], "var": data["var"]} for name, data in detectors_mean_var.items()}

    def _perturb_solenoid_and_measure(self, solenoid_name: str) -> tuple[ParameterDict, list[float], list[float]]:
        """Applies a step change to a solenoid and measures detector responses.

        Args:
            solenoid_name: Solenoid to perturb.

        Returns:
            Tuple containing:
            - Original device parameters before perturbation
            - Original detector readings
            - Detector readings after solenoid perturbation
        """
        original_parameters = self.device.get_parameters()
        original_detector_readings = [original_parameters[detector] for detector in self.detector_names]

        step_size = self.derivative_steps[solenoid_name]
        perturbed_value = original_parameters[solenoid_name] + step_size
        self.device.set(solenoid_name, perturbed_value)
        perturbed_detector_readings = [self.device.get_parameters()[detector] for detector in self.detector_names]

        self.device.set(solenoid_name, original_parameters[solenoid_name])

        return original_parameters, original_detector_readings, perturbed_detector_readings

    def _calculate_parameter_changes(
            self, solenoid_names: list[str], calculate_derivative: bool
    ) -> tuple[ParameterDict, ParameterDict, dict[str, dict[str, float]]]:
        """Calculates parameter changes or derivatives from solenoid perturbations.

        Args:
            solenoid_names: List of solenoids to evaluate
            calculate_derivative: Return derivatives (True) or absolute differences (False)

        Returns:
            Tuple containing:
            - Controlled parameters (non-detector elements)
            - Detector readings from original state
            - Dictionary of calculated changes keyed by solenoid
        """
        reading_changes = {}
        for solenoid in solenoid_names:
            original_params, original_readings, perturbed_readings = self._perturb_solenoid_and_measure(solenoid)
            step_size = self.derivative_steps[solenoid]

            reading_changes[solenoid] = {
                detector: (perturbed - original) / step_size if calculate_derivative else (perturbed - original)
                for detector, original, perturbed in zip(self.detector_names, original_readings, perturbed_readings)
            }

        controlled_parameters = {name: original_params[name] for name in self.element_names if
                                 name not in self.detector_names}
        detector_readings = {name: original_params[name] for name in self.detector_names}

        return controlled_parameters, detector_readings, reading_changes

    def _normalize_data(
            self, data: dict[str, dict[str, float]], normalization_mode: str = "full"
    ) -> dict[str, dict[str, float]]:
        """Applies normalization to measurement data.

        Args:
            data: Raw values to normalize, keyed by element name
            normalization_mode: Normalization strategy ('std' or 'full')

        Returns:
            Dictionary of normalized values with same structure as input
        """
        return {key: self.normalizer.normalize(values, mode=normalization_mode) for key, values in data.items()}

    def measure_shifts(
        self, solenoid_names: list[str], calculate_derivative: bool = True, normalized: bool = False
    ) -> tuple[ParameterDict, ParameterDict, dict[str, dict[str, float]]]:
        """Measures system response to solenoid parameter changes.

        Args:
            solenoid_names: Solenoids to evaluate
            calculate_derivative: Return derivatives (True) or absolute differences (False)
            normalized: Apply normalization to results (True)

        Returns:
            Tuple containing:
            - Controlled parameter values
            - Detector readings (normalized if requested)
            - Calculated changes/derivatives (normalized if requested)
        """
        controlled_parameters, detector_readings, reading_changes = self._calculate_parameter_changes(solenoid_names, calculate_derivative)

        if normalized:
            detector_readings = self._normalize_data(detector_readings, "full")
            reading_changes = self._normalize_data(reading_changes, "std")

        return controlled_parameters, detector_readings, reading_changes

    def measure_step_response(self):
        increments = [0.2 * i for i in range(1, 8)]
        differences = {}
        derivatives = {}
        steps = {}
        init_derivative_steps = self.derivative_steps.copy()
        for increment in increments:
            self.derivative_steps = {solenoid: val * increment for solenoid, val in
                                              init_derivative_steps.items()}
            _, _, differences[increment] = self.measure_shifts(self.solenoid_names, calculate_derivative=False)
            _, _, derivatives[increment] = self.measure_shifts(self.solenoid_names, calculate_derivative=True)
            steps[increment] = {sol: init_derivative_steps[sol] * increment for sol in self.solenoid_names}
        self.derivative_steps = init_derivative_steps
        return {"increments": increments, "steps": steps, "differences": differences, "derivatives": derivatives}

    def measure_full_system_state(self, include_timing: bool = True):
        if include_timing:
            parameters, parameters_time = self._measure_execution_time(self.device.get_parameters)
            (_, _, derivatives), derivatives_time = self._measure_execution_time(self.measure_shifts,
                                                                         self.solenoid_names, calculate_derivative=True)
            meta_time = {"parameters_time": parameters_time, "derivatives_time": derivatives_time}
        else:
            parameters = self.device.get_parameters()
            _, _, derivatives = self.measure_shifts(self.solenoid_names, calculate_derivative=True)
            meta_time = None

        derivatives = {f"derivative_{key}": value for key, value in derivatives.items()}

        meta_data = {}
        meta_data.update(derivatives)
        meta_data.update(parameters)
        meta = {"data": meta_data, "executions_time": meta_time}
        return meta

    def _measure_execution_time(self, func: tp.Callable, *args, **kwargs) -> tuple[tp.Any, float]:
        """Times execution of a function and returns both result and duration.

        Args:
            func: Function to execute and time.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Tuple (function_result, execution_time_seconds).
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        return result, time.time() - start_time

    def get_current_parameters(self) -> ParameterDict:
        """Retrieves current device parameters."""
        return self.device.get_parameters()

    def get_parameters(self, normalize: bool = False) -> tuple[ParameterDict, ParameterDict]:
        """Retrieves and categorizes current device parameters.

        Args:
            normalize: Apply normalization to detector readings

        Returns:
            Tuple containing:
            - Controlled parameters (solenoids/correctors)
            - Detector readings (normalized if requested)
        """
        params = self.get_current_parameters()
        controlled = {name: params[name] for name in self.element_names if name not in self.detector_names}
        detectors = {name: params[name] for name in self.detector_names}
        return controlled, self._normalize_data(detectors) if normalize else detectors

    def set_parameters(self, new_parameters: ParameterDict) -> None:
        """Updates device parameters to specified values.

        Args:
            new_parameters: Dictionary of parameter_name: value pairs.
        """
        for param_name, value in new_parameters.items():
            self.device.set(param_name, value)

class Agent:
    def __init__(self, folder_data_path: tp.Union[str, Path], config_folder: tp.Union[str, Path],
                 element_names: str, device: devices.LaserSetup):
        self.folder_data_path = Path(folder_data_path)
        self.element_names = element_names
        self.config_folder = Path(config_folder)
        self.detector_names, self.solenoid_names, self.corrector_names = utils.get_typed_names(element_names)

        (mean_var_path, derivative_steps_path,
         detector_noise_path, correctors_bounds_path) = ConfigPaths.get_agent_paths(config_folder)

        self.detector_noise_var = DataHandler.read_yaml(detector_noise_path)
        self.correctors_bounds = DataHandler.read_yaml(correctors_bounds_path)

        DataHandler.copy_config(config_folder, self.folder_data_path / "config_folder")
        derivative_steps = DataHandler.read_yaml(derivative_steps_path)
        self.measurer = DeviceMeasurer(element_names, device, folder_data_path, mean_var_path, derivative_steps)
        self._current_optimization_stage = 0

    def _generate_initial_points(self, bounds: list[list[float]]) -> list[list[float]]:
        """Generate initial sampling points (center + corner points) for given bounds."""
        center = [(mn + mx) / 2 for mn, mx in bounds]
        corners = itertools.product(*bounds)
        return [center] + [list(corner) for corner in corners]

    def _verify_parameter_update(self, parameters: dict, expected_values: dict) -> None:
        """Check if parameters were set correctly on the device."""
        for name, expected in expected_values.items():
            actual = parameters[name]
            if not math.isclose(expected, actual, abs_tol=1e-3):
                warnings.warn(
                    f"Parameter {name} set to {expected} but read back as {actual}. "
                    "Device communication may be unreliable."
                )

    def _collect_detector_readings(self, parameter_sets: list[dict]) -> list[list[float]]:
        """Measure detector values for multiple parameter configurations."""
        readings = []
        for params in parameter_sets:
            self.measurer.set_parameters(params)
            current_params = self.measurer.get_current_parameters()
            self._verify_parameter_update(current_params, params)
            readings.append([current_params[name] for name in self.detector_names])
        return readings

    def _get_start_state(self, correctors: list[str]) -> tuple[dict, dict]:
        parameters = self.measurer.get_current_parameters()
        begin_point = [parameters[name] for name in correctors]
        begin_readings = [parameters[name] for name in self.detector_names]
        return begin_point, begin_readings

    def _get_final_values(self, gp_model: MultiOutputModel,
                                   correctors: list[str],
                                   points: list[list[float]],
                                   readings: list[list[float]]):
        best_X = gp_model.best_X.tolist()
        if best_X != points[-1]:
            parameter_sets = [{name: value for name, value in zip(correctors, X)} for X in [best_X]]
            new_reading = self._collect_detector_readings(parameter_sets)
            readings += new_reading
            points.append(best_X)
        return points, readings

    def _translate_to_metadata(self, gp_model: MultiOutputModel,
                                   correctors: list[str],
                                   points: list[list[float]],
                                   readings: list[list[float]]):
        named_points = [{name: value for name, value in zip(correctors, X)} for X in points]
        named_readings = [{name: value for name, value in zip(self.detector_names, X)} for X in readings]
        meta_data = {
            "optimized_correctors": named_points, "detectors_values": named_readings, "model": gp_model.__repr__()
        }
        return meta_data

    def _save_optimization_data(self, start_timestamp: str,
                                end_timestamp: str,
                                meta_data: dict[tp.Any]):
        meta_data.update({"start_optimization_time": start_timestamp})
        meta_data.update({"end_optimization_time": end_timestamp})
        meta_data.update({"optimization_stage": self._current_optimization_stage})

        subfolder_path = self.folder_data_path / "optimization"
        subfolder_path.mkdir(parents=True, exist_ok=True)

        DataHandler.save_yaml(subfolder_path, f"stage_{self._current_optimization_stage}", meta_data)

    def optimize_corrector_values(self, correctors: list[str], normalized: bool = False) -> None:
        """Initialize optimal corrector values using Gaussian Process optimization."""
        start_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        bounds = [
            [self.correctors_bounds[name]["min"], self.correctors_bounds[name]["max"]]
            for name in correctors
        ]
        begin_point, begin_reading = self._get_start_state(correctors)

        points = self._generate_initial_points([[mn / 2, mx / 2] for mn, mx in bounds])
        test_parameters = [{name: val for name, val in zip(correctors, point)}
                           for point in points]
        detector_readings = self._collect_detector_readings(test_parameters)
        initial_points = [begin_point] + points
        detector_readings = [begin_reading] + detector_readings

        gp_model = self._create_gp_model(initial_points, detector_readings, bounds)

        optimized_model, new_points, new_readings = self._iterate_corrector_values(gp_model, correctors, bounds)

        points = points + new_points
        detector_readings = detector_readings + new_readings
        points, detector_readings =\
            self._get_final_values(optimized_model, correctors, points, detector_readings)
        self._current_optimization_stage += 1

        end_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        meta_data = self._translate_to_metadata(gp_model, correctors, points, detector_readings)
        self._save_optimization_data(start_timestamp, end_timestamp, meta_data)

    def _create_gp_model(self, initial_points: list[list[float]],
                         detector_readings: list[list[float]],
                         bounds: list[list[float]]) -> MultiOutputModel:
        """Create and configure Gaussian Process model."""
        Y_var = torch.tensor(list(self.detector_noise_var.values()), dtype=torch.float64)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64).T

        return models.MultiListModel(
            train_X=torch.tensor(initial_points, dtype=torch.float64),
            train_Y=torch.tensor(detector_readings, dtype=torch.float64),
            Yvar=Y_var,
            bounds=bounds_tensor,
            weights=torch.tensor([1, 1, 0])
        )

    def _iterate_corrector_values(self, model: MultiOutputModel,
                                   correctors: list[str],
                                   bounds: list[list[float]]) ->tuple[
        MultiOutputModel, list[list[float]], list[list[float]]
    ]:
        """Run Bayesian optimization loop to find optimal corrector values."""
        optimization_steps = 12
        retrain_interval = 2
        exploration_factors = [i / 10 for i in range(optimization_steps - 2)] + [1.0] * 2
        bounds_t = torch.tensor(bounds).T
        points = []
        readings = []
        for step, alpha in enumerate(exploration_factors):
            if step % retrain_interval == 0:
                model.train()

            # Get new candidate point from acquisition function
            new_point = model.get_new_candidate_point(
                bounds=bounds_t,
                alpha=alpha
            )
            new_point_lst = new_point.tolist()
            # Measure device response
            new_reading = self._collect_detector_readings(
                [{name: value for name, value in zip(correctors, X)} for X in new_point_lst]
            )

            # Update model with new data
            model.add_training_points(
                new_point,
                torch.tensor(new_reading)
            )
            points += new_point_lst
            readings += new_reading
        return model, points, readings

    def measure_noise_characteristics(self) -> None:
        """Measure and save detector noise characteristics."""
        measurement_folder = self.folder_data_path / "noise_measurements"
        measurement_folder.mkdir(parents=True, exist_ok=True)

        noise_data = self.measurer.measure_noise(times=20)
        DataHandler.save_yaml(measurement_folder, "noise_statistics", noise_data)

    def characterize_system_parameters(self) -> None:
        """Full system characterization including step response and timing."""
        results_folder = self.folder_data_path / "system_characterization"
        results_folder.mkdir(parents=True, exist_ok=True)

        step_response = self.measurer.measure_step_response()
        noise_data = self.measurer.measure_noise(times=20)
        full_system_data = self.measurer.measure_full_system_state()

        characterization_data = {
            "step_response": step_response,
            "noise_characteristics": noise_data,
            "full_system_state": full_system_data
        }

        for name, data in characterization_data.items():
            DataHandler.save_yaml(results_folder, name, data)

        self._generate_characterization_plots(results_folder, **step_response)

    def _generate_characterization_plots(self, subfolder: Path, increments: list[float], steps: dict,
                          differences: dict, derivatives: dict):
        subfolder_path = subfolder / "plots"
        subfolder_path.mkdir(parents=True, exist_ok=True)

        for solenoid in self.solenoid_names:
            differences_solenoid = {increment: differences[increment][solenoid] for increment in increments}
            derivatives_solenoid = {increment: derivatives[increment][solenoid] for increment in increments}
            Plotter.save_solenoid_plots(subfolder_path, solenoid, increments, steps, differences_solenoid, derivatives_solenoid)
        path = subfolder_path / "step_response_overview"
        Plotter.save_solenoid_increments(self.solenoid_names, steps, increments, path)
