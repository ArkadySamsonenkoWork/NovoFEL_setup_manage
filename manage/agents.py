import itertools
import typing as tp
from pathlib import Path
import time

from matplotlib import pyplot as plt
import numpy as np

from abc import ABC, abstractmethod
from dataclasses import dataclass

import devices
import utils
from utils import DataHandler, ConfigPaths
from system_interface import Normalizer, DeviceMeasurer, DeviceSolenoidAnalyser

from optimization import BaseOptimizer, OptimizerValues, OptimizerShift, RandomOptimizerValues


class Strategy(ABC):
    def __init__(self, detector_names: list[str], corrector_names: list[str],
                 corrector_limits: dict[str, dict], normalizer, measurer,
                 folder_data_path, detector_noise_var, normalized: bool = True):
        self.corrector_names = corrector_names
        self.corrector_limits = corrector_limits
        self.detector_names = detector_names
        self.optimizer = self.create_optimizer()(
            detector_names=detector_names, measurer=measurer,
            normalizer=normalizer, folder_data_path=folder_data_path,
            detector_noise_var=detector_noise_var, normalized=normalized)

    @abstractmethod
    def get_weights(self, **kwargs) -> list[float]:
        pass

    @abstractmethod
    def create_optimizer(self) -> tp.Type[BaseOptimizer]:
        pass

    def run(self, corrector_names: list[str], **kwargs):
        filtered_limits = {
            c: self.corrector_limits[c]
            for c in corrector_names
            if c in self.corrector_limits
        }
        weights = self.get_weights(**kwargs)
        self.optimizer.optimize_corrector_values(
            weights=weights,
            corrector_names=corrector_names,
            corrector_limits=filtered_limits,
            **kwargs
        )

    def stage(self):
        return self.optimizer.stage

class InjectorStrategy(Strategy):
    def get_weights(self, **kwargs):
        solenoids = kwargs.get("solenoids", [])
        weights = [1.0] * len(self.detector_names) * len(solenoids) + [1.0] * len(self.detector_names)
        for idx in range(len(weights)):
            if (idx+1) % len(self.detector_names) == 0:
                weights[idx] = 0.0
        return weights

class RandomStrategy(InjectorStrategy):
    def create_optimizer(self):
        return RandomOptimizerValues

class ValueStrategy(InjectorStrategy):
    def create_optimizer(self):
        return OptimizerValues

class ShiftStrategy(InjectorStrategy):
    def create_optimizer(self):
        return OptimizerShift



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

class BaseAgent(ABC):
    def __init__(self, folder_data_path: tp.Union[str, Path], config_folder: tp.Union[str, Path],
                 element_names: list[str], device: devices.LaserSetup):
        self.folder_data_path = Path(folder_data_path)
        self.element_names = element_names
        self.config_folder = Path(config_folder)
        self.detector_names, self.solenoid_names, self.corrector_names = utils.get_typed_names(element_names)

        (mean_var_path, derivative_steps_path, correctors_bounds_path) = ConfigPaths.get_agent_paths(config_folder)
        mean_var = DataHandler.read_yaml(mean_var_path)
        detector_noise_var = {key: value["var"] for key, value in mean_var["detectors"].items()}

        self.detector_noise_var = detector_noise_var
        self.corrector_limits = DataHandler.read_yaml(correctors_bounds_path)

        DataHandler.copy_config(config_folder, self.folder_data_path / "config_folder")
        derivative_steps = DataHandler.read_yaml(derivative_steps_path)

        self.normalizer = Normalizer(mean_var)
        self.measurer = DeviceMeasurer(element_names, device, folder_data_path, self.normalizer, derivative_steps)


@dataclass
class PipelineStage:
    strategy_name: str
    corrector_names: list[str]
    kwargs: dict



class AgentInjector(BaseAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._strategies = self._init_strategies()
        self.pipeline = []

    def _init_strategies(self) -> dict[str, InjectorStrategy]:
        random_strategy = RandomStrategy(self.detector_names, self.corrector_names,
                                         self.corrector_limits, self.normalizer,
                                         self.measurer, self.folder_data_path, self.detector_noise_var,
                                         normalized=True,
                                         )
        value_strategy = ValueStrategy(self.detector_names, self.corrector_names,
                                         self.corrector_limits, self.normalizer,
                                         self.measurer, self.folder_data_path, self.detector_noise_var,
                                         normalized=True,
                                         )
        shift_strategy = ShiftStrategy(self.detector_names, self.corrector_names,
                                         self.corrector_limits, self.normalizer,
                                         self.measurer, self.folder_data_path, self.detector_noise_var,
                                         normalized=True,
                                         )
        return {"random": random_strategy, "value": value_strategy, "shift": shift_strategy}


    def add_stage(self, strategy_name: str, corrector_names: list[str], **kwargs):
        self.pipeline.append(PipelineStage(
            strategy_name=strategy_name,
            corrector_names=corrector_names,
            kwargs=kwargs
        ))

    def run_pipeline(self):
        """Execute the entire optimization pipeline"""
        for i, stage in enumerate(self.pipeline):
            strategy = self._strategies[stage.strategy_name]

            print(f"Running {stage.strategy_name} strategy with:")
            print(f"Correctors: {stage.corrector_names}")

            strategy.run(
                corrector_names=stage.corrector_names,
                **stage.kwargs
            )
            print(f"finish stage {i}")


    def create_pipeline_phase_0(self):
        strategy_name = "random"
        corrector_names = self.corrector_names
        centralized = False,
        exploration_steps = 10
        exploitation_steps = 3
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps, exploitation_steps=exploitation_steps)

    def create_pipeline_phase_1(self):
        strategy_name = "shift"
        centralized = True,
        scale_factor = 0.8
        exploration_steps = 14
        exploitation_steps = 3

        corrector_names = ["1MXY1_X", "1MXY1_Y"]
        solenoids = ["1MS1"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

        corrector_names = ["1MXY2_X", "1MXY2_Y"]
        solenoids = ["1MS2"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

        corrector_names = ["1MXY3_X", "1MXY4_X", "1MXY3_Y", "1MXY4_Y"]
        solenoids = ["1MS3"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

        corrector_names = ["1MXY5_Y", "1MXY5_X", "1MXY6_Y"]
        solenoids = ["1MS4", "1MS6"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

        corrector_names = ["1MQ71_X"]
        solenoids = ["1MS8"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

    def create_pipeline_phase_2(self):
        strategy_name = "shift"
        centralized = True,
        scale_factor = 0.4
        exploration_steps = 10
        exploitation_steps = 3

        corrector_names = ["1MXY1_X", "1MXY1_Y", "1MXY2_X", "1MXY2_Y"]
        solenoids = ["1MS1", "1MS2"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

        corrector_names = ["1MXY3_X", "1MXY4_X", "1MXY3_Y", "1MXY4_Y"]
        solenoids = ["1MS3"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)


        corrector_names = ["1MXY5_Y", "1MXY5_X", "1MXY6_Y"]
        solenoids = ["1MS4", "1MS6"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

        corrector_names = ["1MQ71_X"]
        solenoids = ["1MS8"]
        self.add_stage(strategy_name, corrector_names, centralized=centralized,
                       exploration_steps=exploration_steps,
                       exploitation_steps=exploitation_steps, solenoids=solenoids, scale_factor=scale_factor)

    def create_pipeline(self):
        self.create_pipeline_phase_0()
        self.create_pipeline_phase_1()
        self.create_pipeline_phase_2()

    def run(self):
        start_time = time.time()
        self.create_pipeline()
        self.run_pipeline()
        end_time = time.time()
        print(f"spent time {end_time-start_time}")


class AnalyzeAgent(BaseAgent):
    def __init__(self, folder_data_path: tp.Union[str, Path], config_folder: tp.Union[str, Path],
                 element_names: str, device: devices.LaserSetup):
        super().__init__(folder_data_path, config_folder, element_names, device)
        self.analyzer = DeviceSolenoidAnalyser(self.measurer)


    def characterize_system_parameters(self) -> None:
        """Full system characterization including step response and timing."""
        results_folder = self.folder_data_path / "system_characterization"
        results_folder.mkdir(parents=True, exist_ok=True)

        step_response = self.analyzer.measure_step_response()
        detectors_mean_var = self.analyzer.measure_mean_var(times=20)
        full_system_data = self.analyzer.measure_full_system_state()

        characterization_data = {
            "step_response": step_response,
            "detectors_mean_var": detectors_mean_var,
            "full_system_state": full_system_data
        }

        for name, data in characterization_data.items():
            DataHandler.save_yaml(results_folder, name, data)

        self._generate_characterization_plots(results_folder, **step_response)

    def characterize_noize(self) -> None:
        """Full system characterization including step response and timing."""
        results_folder = self.folder_data_path / "system_noize"
        results_folder.mkdir(parents=True, exist_ok=True)
        detectors_mean_var = self.analyzer.measure_mean_var(times=20)
        characterization_data = {
            "detectors_mean_var": detectors_mean_var,
        }

        for name, data in characterization_data.items():
            DataHandler.save_yaml(results_folder, name, data)

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
