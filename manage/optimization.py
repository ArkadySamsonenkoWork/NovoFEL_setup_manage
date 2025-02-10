import itertools
import time
import typing as tp
import warnings
from pathlib import Path
from utils import DataHandler

import math
import torch

import models

from manage.models import MultiOutputModel
from system_interface import DeviceMeasurer, Normalizer
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, detector_names: list[str], detector_noise_var,
                 normalized: bool, folder_data_path: Path | str,
                 measurer: DeviceMeasurer,
                 normalizer: Normalizer):
        self.detector_names = detector_names
        self.measurer = measurer
        self.normalized = normalized
        self.normalizer = normalizer
        self.folder_data_path = folder_data_path
        self._current_optimization_stage = 0
        if normalized:
            detector_noise_var = self.normalizer.normalize_squares(detector_noise_var, mode="var")
        self.target_var = self.set_target_var(detector_noise_var)
        self.detector_noise_var = detector_noise_var

    @abstractmethod
    def set_target_var(self, detector_noise_var):
        pass

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

    @abstractmethod
    def _collect_detector_readings(self, parameter_sets: list[dict], **kwargs) -> list[list[float]]:
        pass

    @abstractmethod
    def _get_start_state(self, correctors: list[str]) -> tuple[dict, dict]:
        pass

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


class OptimizerShift(Optimizer):
    def set_target_var(self, detector_noise_var):
        target_var = {}
        for key, value in detector_noise_var.items():
            target_var[key] = value
            target_var[f"{key}_shift"] = 2 * value
        return target_var

    def _collect_detector_readings(self, parameter_sets: list[dict], solenoids: list[str]) -> list[list[float]]:
        """Measure detector values for multiple parameter configurations."""
        readings = []
        for params in parameter_sets:
            self.measurer.set_parameters(params)
            current_params, detector_readings, reading_changes =\
                self.measurer.measure_shifts(normalized=self.normalized)
            self._verify_parameter_update(current_params, params)
            new_reading = []
            for name in self.detector_names:
                new_reading.append(detector_readings[name])
                new_reading.extend([reading_changes[solenoid][name] for solenoid in solenoids])
            readings.append(new_reading)
        return readings

    def _get_start_state(self, correctors: list[str], solenoids: list[str]) -> tuple[dict, dict]:
        parameters, readings, shifts  = self.measurer.measure_shifts(normalized=self.normalized)
        begin_point = [parameters[name] for name in correctors]
        begin_readings = []
        for name in self.detector_names:
            begin_readings.append(readings[name])
            begin_readings.extend([shifts[solenoid][name] for solenoid in solenoids])
        return begin_point, begin_readings

    def _parse_readings(self, solenoids, readings):
        num_detectors = self.detector_names
        for reading in readings:
            pass

    def _translate_to_metadata(self, gp_model: MultiOutputModel,
                               correctors: list[str],
                               solenoids: list[str],
                               points: list[list[float]],
                               readings: list[list[float]]):

        if self.normalized:
            named_points = [
                self.normalizer.normalize({name: value for name, value in zip(correctors, X)}, mode="denormalize_full")
                for X in points]
            named_readings = [
                self.normalizer.normalize({name: value for name, value in zip(self.detector_names, X)}, mode="denormalize_full")
                for X in readings]
            Yvar = self.normalizer.normalize_squares(self.detector_noise_var, mode="denormalize_var")
        else:
            named_points = [
                {name: value for name, value in zip(correctors, X)}
                for X in points]
            named_readings = [
                {name: value for name, value in zip(self.detector_names, X)}
                for X in points]
            Yvar = self.target_var
        meta_data = {
            "optimized_correctors": named_points, "detectors_values": named_readings,
            "detector_noise_var": Yvar,
            "is used normalization": self.normalized,
            "model": gp_model.__repr__()
        }
        return meta_data


class OptimizerValues(Optimizer):
    def set_target_var(self, detector_noise_var):
        return detector_noise_var


    def _collect_detector_readings(self, parameter_sets: list[dict]) -> list[list[float]]:
        """Measure detector values for multiple parameter configurations."""
        readings = []
        for params in parameter_sets:
            self.measurer.set_parameters(params)
            current_params = self.measurer.get_current_parameters(normalized=self.normalized)
            self._verify_parameter_update(current_params, params)
            readings.append([current_params[name] for name in self.detector_names])
        return readings

    def _get_start_state(self, correctors: list[str]) -> tuple[dict, dict]:
        parameters = self.measurer.get_current_parameters(normalize=self.normalized)
        begin_point = [parameters[name] for name in correctors]
        begin_readings = [parameters[name] for name in self.detector_names]
        return begin_point, begin_readings

    def _translate_to_metadata(self, gp_model: MultiOutputModel,
                               correctors: list[str],
                               points: list[list[float]],
                               readings: list[list[float]]):
        if self.normalized:
            named_points = [
                self.normalizer.normalize({name: value for name, value in zip(correctors, X)}, mode="denormalize_full")
                for X in points]
            named_readings = [
                self.normalizer.normalize({name: value for name, value in zip(self.detector_names, X)}, mode="denormalize_full")
                for X in readings]
            Yvar = self.normalizer.normalize_squares(self.target_var, mode="denormalize_var")
        else:
            named_points = [
                {name: value for name, value in zip(correctors, X)}
                for X in points]
            named_readings = [
                {name: value for name, value in zip(self.detector_names, X)}
                for X in points]
            Yvar = self.target_var
        meta_data = {
            "optimized_correctors": named_points, "detectors_values": named_readings,
            "detector_noise_var": Yvar,
            "is used normalization": self.normalized,
            "model": gp_model.__repr__()
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

    def optimize_corrector_values(self, correctors: list[str], correctors_bounds) -> None:
        """Initialize optimal corrector values using Gaussian Process optimization."""
        self.correctors_bounds = correctors_bounds
        start_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

        mins = {name: self.correctors_bounds[name]["min"] for name in correctors}
        maxes = {name: self.correctors_bounds[name]["max"] for name in correctors}
        if self.normalized:
            mins = self.normalizer.normalize(mins, mode="full")
            maxes = self.normalizer.normalize(maxes, mode="full")
        bounds = [[mins[name], maxes[name]] for name in correctors]
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
        points, detector_readings = \
            self._get_final_values(optimized_model, correctors, points, detector_readings)
        self._current_optimization_stage += 1

        end_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        meta_data = self._translate_to_metadata(gp_model, correctors, points, detector_readings)
        self._save_optimization_data(start_timestamp, end_timestamp, meta_data)

    def _create_gp_model(self, initial_points: list[list[float]],
                         detector_readings: list[list[float]],
                         bounds: list[list[float]]) -> MultiOutputModel:
        """Create and configure Gaussian Process model."""
        Yvar = torch.tensor(list(self.detector_noise_var.values()), dtype=torch.float64)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64).T

        return models.MultiListModel(
            train_X=torch.tensor(initial_points, dtype=torch.float64),
            train_Y=torch.tensor(detector_readings, dtype=torch.float64),
            Yvar=Yvar,
            bounds=bounds_tensor,
            weights=torch.tensor([1, 1, 0])
        )

    def _iterate_corrector_values(self, model: MultiOutputModel,
                                  correctors: list[str],
                                  bounds: list[list[float]]) -> tuple[
        MultiOutputModel, list[list[float]], list[list[float]]
    ]:
        """Run Bayesian optimization loop to find optimal corrector values."""
        optimization_steps = 30
        retrain_interval = 2
        exploration_factors = [i / 10 for i in range(optimization_steps - 2)] + [1.0] * 4
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