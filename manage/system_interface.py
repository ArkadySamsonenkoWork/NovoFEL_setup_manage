import time
import typing as tp
from pathlib import Path
import devices
import utils

class Normalizer:
    def __init__(self, mean_var: dict):
        self.mean_var = mean_var
        self.mean_var = {k: v for data in list(self.mean_var.values()) for k, v in data.items()}

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

    def normalize_shift(self, values: dict, mode: str = "full") -> dict:
        result = {}
        for key, value in values.items():
            stats = self.mean_var.get(key, {"mean": 0, "var": 1})
            if (mode == "full") or (mode == "std"):
                result[key] = value / (stats["var"] * 2) ** 0.5
            elif mode == "denormalize_std":
                result[key] = value * (stats["var"] * 2) ** 0.5
            elif mode == "denormalize_full":
                result[key] = value * (stats["var"] * 2) ** 0.5
        return result

    def normalize_squares(self, values: dict, mode: str = "full") -> dict:
        result = {}
        for key, value in values.items():
            stats = self.mean_var.get(key, {"mean": 0, "var": 1})
            if mode == "var":
                result[key] = value / stats["var"]
            elif mode == "denormalize_var":
                result[key] = value * stats["var"]
        return result


ParameterDict = dict[str, float]
DerivativeSteps = dict[str, float]

class DeviceMeasurer:
    """Measures and analyzes device parameters, noise, and system responses for a laser setup."""

    def __init__(self, element_names: list[str], device: devices.LaserSetup,
                 folder: tp.Union[str, Path], normalizer: Normalizer, derivative_steps: dict[float]):
        self.element_names = element_names
        self.device = device
        self.folder = folder
        self.detector_names, self.solenoid_names, self.corrector_names = utils.get_typed_names(element_names)
        self.normalizer = normalizer
        self.derivative_steps = derivative_steps

    def measure_mean_var(self, times: int):
        detectors_mean_var = self.device.get_detectors_mean_var(times)
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
        return self.normalizer.normalize(data, mode=normalization_mode)

    def _normalize_shift(
            self, data: dict[str, dict[str, float]], normalization_mode: str = "full"
    ) -> dict[str, dict[str, float]]:
        """Applies normalization to measurement data.
        Args:
            data: Raw values to normalize, keyed by element name
            normalization_mode: Normalization strategy ('std' or 'full')
        Returns:
            Dictionary of normalized values with same structure as input
        """
        for key in data:
            data[key] = self.normalizer.normalize_shift(data[key], mode=normalization_mode)
        return data

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
        controlled_parameters, detector_readings, reading_changes =\
            self._calculate_parameter_changes(solenoid_names, calculate_derivative)

        if normalized:
            detector_readings = self._normalize_data(detector_readings, "full")
            controlled_parameters = self._normalize_data(controlled_parameters, "full")
            reading_changes = self._normalize_shift(reading_changes, "std")
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

    def get_current_parameters(self, normalized: bool = False) -> ParameterDict:
        """Retrieves and categorizes current device parameters.

        Args:
            normalized: Apply normalization to detector readings

        Returns:
            Tuple containing:
            - Controlled parameters (solenoids/correctors)
            - Detector readings (normalized if requested)
        """
        params = self.device.get_parameters()
        #controlled = {name: params[name] for name in self.element_names if name not in self.detector_names}
        #detectors = {name: params[name] for name in self.detector_names}
        return self._normalize_data(params) if normalized else params


    def set_parameters(self, new_parameters: ParameterDict, denormalized: bool =False) -> None:
        """Updates device parameters to specified values.

        Args:
            new_parameters: Dictionary of parameter_name: value pairs.
            denormalized: Should denormalize data to set.
        """
        if not denormalized:
            new_parameters = self.normalizer.normalize(new_parameters, mode="denormalize_full")
        for param_name, value in new_parameters.items():
            self.device.set(param_name, value)