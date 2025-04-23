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
import numpy as np


class MetadataParser(ABC):
    def __init__(self, normalizer, detector_names, detector_noise_var, normalized=False):
        self.normalizer = normalizer
        self.detector_names = detector_names
        self.normalized = normalized
        self.detector_noise_var = detector_noise_var

    def create_metadata(self, gp_model: MultiOutputModel,
                               corrector_names: list[str],
                               corrector_values: list[list[float]],
                               readings: list[list[float]], **kwargs):

        return {
            "optimized_correctors": self._process_correctors(corrector_names, corrector_values),
            **self._process_readings(readings, **kwargs),
            "detector_noise_var": self._process_variance(),
            "is used normalization": self.normalized,
            "model": repr(gp_model)
        }

    def _process_correctors(self, corrector_names: list[str], corrector_values: list[list[float]]):
        if self.normalized:
            named_correctors = [
                self.normalizer.normalize_parameters({name: value for name, value in zip(corrector_names, X)},
                                                     mode="denormalize_full") for X in corrector_values
            ]
        else:
            named_correctors = [
                {name: value for name, value in zip(corrector_names, X)} for X in corrector_values
            ]
        return named_correctors

    def _process_variance(self):
        if self.normalized:
            Yvar = self.normalizer.normalize_squares(self.detector_noise_var, mode="denormalize_var")
        else:
            Yvar = self.detector_noise_var,
        return Yvar

    @abstractmethod
    def _process_readings(self, readings, **kwargs) -> dict[str, tp.Any]:
        pass

class ReadingsMetadataParser(MetadataParser):
    """Handles Class 1's simple value-only metadata"""
    def _process_readings(self, readings, **_) -> dict[str, tp.Any]:
        return {
            "detectors_values": [
                self._normalize_reading(reading)
                for reading in readings
            ]
        }

    def _normalize_reading(self, reading):
        if self.normalized:
            return self.normalizer.normalize_parameters(
                dict(zip(self.detector_names, reading)), "denormalize_full"
            )
        return dict(zip(self.detector_names, reading))


class ShiftMetadataParser(MetadataParser):
    """Handles Class 2's shift+value metadata"""
    def _process_readings(self, readings, solenoids) -> dict[str, tp.Any]:
        return {
            "detectors_values": self._parse_values(readings, solenoids),
            "detectors_shift": self._parse_shifts(readings, solenoids)
        }

    def _parse_values(self, readings, solenoids):
        return [self._parse_reading(r, solenoids, is_shift=False) for r in readings]

    def _parse_shifts(self, readings, solenoids):
        return [self._parse_reading(r, solenoids, is_shift=True) for r in readings]

    def _parse_reading(self, reading, solenoids, is_shift):
        read_shift = len(solenoids)
        base_dict = {
            detector: value
            for detector, value in zip(
                self.detector_names,
                reading[1::read_shift] if is_shift else reading[::read_shift]
            )
        }
        if is_shift:
            return {
                solenoid: self._normalize_shift_part(base_dict, i)
                for i, solenoid in enumerate(solenoids)
            }
        return self._normalize_value_part(base_dict)

    def _normalize_value_part(self, values):
        if self.normalized:
            return self.normalizer.normalize_parameters(values, "denormalize_full")
        return values

    def _normalize_shift_part(self, values, solenoid_idx):
        if self.normalized:
            return self.normalizer.normalize_shift(values, "denormalize_full")
        return values

class BaseOptimizer(ABC):
    stage = 0
    def __init__(self, detector_names: list[str], detector_noise_var,
                 normalized: bool, folder_data_path: Path | str,
                 measurer: DeviceMeasurer,
                 normalizer: Normalizer):
        self.detector_names = detector_names
        self.measurer = measurer
        self.normalized = normalized
        self.normalizer = normalizer
        self.folder_data_path = folder_data_path
        if normalized:
            detector_noise_var = self.normalizer.normalize_squares(detector_noise_var, mode="var")
        self.target_var = self.set_target_var(detector_noise_var)
        self.detector_noise_var = detector_noise_var
        self.metadata_parser = self.get_metadata_parser()

    def _collect_initial_readings(self, corrector_names: list[str], corrector_values: list[list[float]],
                                  **kwargs) -> list[list[float]]:
        """Common initial readings collection"""
        param_sets = [{n: v for n, v in zip(corrector_names, p)} for p in corrector_values]
        return self._collect_detector_readings(param_sets, **kwargs)

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
    def set_target_var(self, detector_noise_var):
        pass

    @abstractmethod
    def _collect_detector_readings(self, parameter_sets: list[dict], **kwargs) -> list[list[float]]:
        pass

    def get_metadata_parser(self) -> MetadataParser:
        return self._get_metadata_parser_type()(
            self.normalizer, self.detector_names, self.detector_noise_var, self.normalized)

    @abstractmethod
    def _get_metadata_parser_type(self) -> tp.Type[MetadataParser]:
        pass

    @abstractmethod
    def _get_initial_state(self, correctors: list[str], **kwargs) -> tuple[list[float], list[float]]:
        pass

    def _save_optimization_data(self, start_timestamp: str,
                                end_timestamp: str,
                                meta_data: dict[tp.Any]):
        meta_data.update({"start_optimization_time": start_timestamp})
        meta_data.update({"end_optimization_time": end_timestamp})
        meta_data.update({"optimization_stage": BaseOptimizer.stage})
        subfolder_path = self.folder_data_path / "optimization"
        subfolder_path.mkdir(parents=True, exist_ok=True)
        DataHandler.save_yaml(subfolder_path, f"stage_{BaseOptimizer.stage}", meta_data)

    def _create_gp_model(self, initial_points: list[list[float]],
                         detector_readings: list[list[float]], weights: torch.Tensor,
                         bounds: list[list[float]]) -> MultiOutputModel:
        """Create and configure Gaussian Process model."""
        tot_len = len(detector_readings[0])
        variance_list = list(self.target_var.values())
        target_var = [1] * tot_len
        for idx in range(tot_len):
            if (idx) % (tot_len // 3) == 0:
                target_var[idx] = variance_list[0]
            else:
                target_var[idx] = variance_list[1]
        Yvar = torch.tensor(target_var, dtype=torch.float64)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64).T
        return models.MultiListModel(
            train_X=torch.tensor(initial_points, dtype=torch.float64),
            train_Y=torch.tensor(detector_readings, dtype=torch.float64),
            Yvar=Yvar,
            bounds=bounds_tensor,
            weights=weights
        )

    def _create_initial_model(self, points: list[list[float]], readings: list[list[float]],
                              weights: tp.Union[list[float], torch.Tensor], bounds: list[list[float]]) -> MultiOutputModel:
        """Common model initialization"""
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights)

        return self._create_gp_model(
            points,
            readings,
            weights,
            bounds
        )

    def create_metadata(
            self, gp_model: MultiOutputModel,
                    corrector_names: list[str],
                    corrector_values: list[list[float]],
                    detector_readings: list[list[float]], **kwargs):
        return self.metadata_parser.create_metadata(gp_model, corrector_names, corrector_values,
                                                     detector_readings, **kwargs)

    def optimize_corrector_values(self, corrector_names: list[str], corrector_limits: dict,
                                  weights: tp.Union[list[float], torch.Tensor], centralized: bool = True,
                                  scale_factor: float = 1/2, exploration_steps: int = 10, exploitation_steps: int = 3,
                                  **kwargs) -> None:
        """

        :param corrector_names: names of correctors which value must be optimized
        :param corrector_limits: the limits in which the correctors values can vary
        :param weights:
        :param centralized:
        :param scale_factor:
        :param exploration_steps:
        :param exploitation_steps
        :param kwargs:
        :return:
        """
        start_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

        begin_point, begin_reading = self._get_initial_state(corrector_names, **kwargs)
        init_points, bounds = self._process_init_points(corrector_names, corrector_limits, scale_factor,
                                                        begin_point, centralized)
        initial_readings = self._collect_initial_readings(corrector_names, init_points, **kwargs)
        initial_readings = [begin_reading] + initial_readings
        points = [begin_point] + init_points  # sometimes it can be measured twice

        gp_model = self._create_initial_model(points, initial_readings, weights, bounds)

        optimized_model, new_points, new_readings = self._run_optimization_loop(
            gp_model, corrector_names, bounds, exploration_steps, exploitation_steps, **kwargs)

        points = points + new_points
        detector_readings = initial_readings + new_readings
        points, detector_readings = \
            self._get_final_values(optimized_model, corrector_names, points, detector_readings, **kwargs)

        BaseOptimizer.stage += 1
        end_timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        meta_data = self.create_metadata(gp_model, corrector_names, points, detector_readings, **kwargs)
        self._save_optimization_data(start_timestamp, end_timestamp, meta_data)

    def _process_init_points(self, corrector_names: list[str], correctors_limits: dict,
                        scale_factor: float, begin_point: list[float], centralized: bool)\
            -> tuple[list[list[float]], list[list[float]]]:
        """Common bounds processing"""
        bounds = self._get_init_bounds(corrector_names, correctors_limits)
        if centralized:
            center, bounds = self._get_transformed_bounds(bounds, scale_factor, begin_point)
        else:
            center, bounds = self._get_transformed_bounds(bounds, scale_factor, None)
        init_points = self._generate_init_points(bounds, center)
        return init_points, bounds

    def _get_init_bounds(self, correctors: list[str], correctors_bounds: dict) -> list[list[float]]:
        mins = {n: correctors_bounds[n]["min"] for n in correctors}
        maxes = {n: correctors_bounds[n]["max"] for n in correctors}
        if self.normalized:
            mins = self.normalizer.normalize_parameters(mins, "full")
            maxes = self.normalizer.normalize_parameters(maxes, "full")
        return [[mins[n], maxes[n]] for n in correctors]

    def _get_transformed_bounds(self, bounds: list[list[float]], scale_factor: float,
                                 center: list[float] | None = None):
        center = center or [(low + high) / 2 for low, high in bounds]
        shifts = [((high - low) / 2) * scale_factor for low, high in bounds]
        offset_bounds = [[c - s, c + s] for c, s in zip(center, shifts)]
        clipped_bounds = [
            [max(orig[0], off[0]), min(orig[1], off[1])]
            for orig, off in zip(bounds, offset_bounds)
        ]
        return center, clipped_bounds

    def _generate_init_points(self, bounds: list[list[float]], center: list[float] | None = None)\
            -> list[list[float]]:
        """
            Generate initial sampling points (center + corner points) for given bounds,
            using no explicit loops—only comprehensions.

            Args:
                bounds: List of [min, max] per dimension.
                center: Optional center coordinates; defaults to midpoints.
            Returns:
                List of points: center plus corners.
            """
        corners = [list(pt) for pt in itertools.product(*bounds)]
        return [center] + corners

    def _get_final_values(self, gp_model: MultiOutputModel,
                          corrector_names: list[str],
                          corrector_values: list[list[float]],
                          detector_readings: list[list[float]],
                          **kwargs,
                          ):
        best_X = gp_model.best_X.tolist()
        if best_X != corrector_values[-1]:
            parameter_sets = [{name: value for name, value in zip(corrector_names, X)} for X in [best_X]]
            new_reading = self._collect_detector_readings(parameter_sets, **kwargs)
            detector_readings += new_reading
            corrector_values.append(best_X)
        return corrector_values, detector_readings

    def _run_optimization_loop(self, model: MultiOutputModel, correctors: list[str],
                               bounds: list[list[float]], exploration_steps: int, exploitation_steps: int,
                               **kwargs) -> tuple[
        MultiOutputModel, list[list[float]], list[list[float]]]:
        """Common optimization loop structure"""
        bounds_t = torch.tensor(bounds).T
        points, readings = [], []
        exploration_factors = self._get_exploration_factors(exploration_steps, exploitation_steps)
        retrain_interval = self._get_retrain_interval()
        for step, alpha in enumerate(exploration_factors):
            if step % retrain_interval == 0:
                model.train()

            new_point = model.get_new_candidate_point(bounds_t, alpha)
            new_reading = self._collect_new_reading(correctors, new_point, **kwargs)
            model.add_training_points(new_point, torch.tensor(new_reading))
            points += new_point.tolist()
            readings += new_reading
        return model, points, readings

    def _collect_new_reading(self, corrector_names: list[str], corrector_values: torch.Tensor, **kwargs):
        solenoids = kwargs["solenoids"]
        return self._collect_detector_readings(
            [{n: v for n, v in zip(corrector_names, X)} for X in corrector_values.tolist()], solenoids
        )

    def _get_retrain_interval(self) -> int:
        return 2

    def _get_exploration_factors(self, exploration_steps: int, exploitation_steps: int) -> list[float]:
        return [i/exploration_steps for i in range(exploration_steps)] + [1.0] * exploitation_steps


class OptimizerShift(BaseOptimizer):
    def set_target_var(self, detector_noise_var):
        target_var = {}
        for key, value in detector_noise_var.items():
            target_var[key] = value
            target_var[f"{key}_shift"] = 2 * value
        return target_var

    def _get_metadata_parser_type(self) -> tp.Type[MetadataParser]:
        return ShiftMetadataParser

    def _collect_detector_readings(self, parameter_sets: list[dict], solenoids: list[str]) -> list[list[float]]:
        """Measure detector values for multiple parameter configurations."""
        readings = []
        for params in parameter_sets:
            self.measurer.set_parameters(params)
            current_params, value_reading, shift_reading =\
                self.measurer.measure_shifts(normalized=self.normalized, solenoid_names=solenoids)
            self._verify_parameter_update(current_params, params)
            new_reading = []
            for name in self.detector_names:
                new_reading.append(value_reading[name])
                new_reading.extend([shift_reading[solenoid][name] for solenoid in solenoids])
            readings.append(new_reading)
        return readings

    def _get_initial_state(self, correctors: list[str], solenoids: list[str]) -> tuple[dict, dict]:
        parameters, value_reading, shift_reading = self.measurer.measure_shifts(normalized=self.normalized,
                                                                                 solenoid_names=solenoids)
        begin_points = [parameters[name] for name in correctors]
        begin_readings = []
        for name in self.detector_names:
            begin_readings.append(value_reading[name])
            begin_readings.extend([shift_reading[solenoid][name] for solenoid in solenoids])
        return begin_points, begin_readings


class OptimizerValues(BaseOptimizer):
    def set_target_var(self, detector_noise_var):
        return detector_noise_var

    def _get_metadata_parser_type(self) -> tp.Type[MetadataParser]:
        return ReadingsMetadataParser

    def _collect_detector_readings(self, parameter_sets: list[dict]) -> list[list[float]]:
        """Measure detector values for multiple parameter configurations."""
        readings = []
        for params in parameter_sets:
            self.measurer.set_parameters(params)
            current_params, readings = self.measurer.measure_parameters(normalized=self.normalized)
            self._verify_parameter_update(current_params, params)
            readings.append([readings[name] for name in self.detector_names])
        return readings

    def _get_initial_state(self, correctors: list[str], **kwargs) -> tuple[dict, dict]:
        current_params, readings = self.measurer.measure_parameters(normalized=self.normalized)
        begin_points = [current_params[name] for name in correctors]
        begin_readings = [readings[name] for name in self.detector_names]
        return begin_points, begin_readings


class RandomOptimizerValues(OptimizerValues):
    def get_random_points(self, corrector_names: list[str], correctors_limits: dict, points: int = 10) \
            -> tuple[list[list[float]], list[list[float]]]:
        bounds = self._get_init_bounds(corrector_names, correctors_limits)
        mins = [bound[0] for bound in bounds]
        maxes = [bound[1] for bound in bounds]
        init_points = []
        for _ in range(points):
            uniform_points = np.random.uniform(mins, maxes)
            init_point = {key: value.item() for key, value in zip(bounds, uniform_points)}
            init_points.append(init_point)
        return bounds, init_points

    def optimize_corrector_values(self, corrector_names: list[str], corrector_limits: dict,
                                  weights: tp.Union[list[float], torch.Tensor], centralized: bool = True,
                                  scale_factor: float = 1/2, exploration_steps: int = 10, exploitation_steps: int = 4,
                                  **kwargs) -> None:
        """

        :param corrector_names: names of correctors which value must be optimized
        :param corrector_limits: the limits in which the correctors values can vary
        :param weights:
        :param centralized:
        :param scale_factor:
        :param kwargs:
        :return:
        """
        init_readings = [self.measurer.measure_parameters()[1][detector] for detector in self.detector_names]
        init_loss = sum(weight * reading**2 for weight, reading in zip(weights, init_readings))
        Yvar = [noise for noise in self.detector_noise_var.values()]
        ideal_loss = sum(weight * reading for weight, reading in zip(weights, Yvar))
        if init_loss <= ideal_loss * 10:
            print("init_loss is OK")
            pass
        else:
            print("init_loss is not OK")
            init_points, bounds = self.get_random_points(corrector_names, corrector_limits, points=exploration_steps)
            losses = []
            readings = []
            for point in init_points:
                new_reading = self._collect_new_reading(corrector_names, point, **kwargs)
                loss = sum(weight * reading ** 2 for weight, reading in zip(weights, new_reading[0]))
                losses.append(loss)
                readings += new_reading
                if loss <= ideal_loss * 10:
                    break

            # Переделать потом!!!!!!
            min_value = min(losses)
            index = losses.index(min_value)
            min_point = init_points[index]
            readings += self._collect_new_reading(corrector_names, min_point, **kwargs)
        BaseOptimizer.stage += 1