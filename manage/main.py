import time

import agents
import devices
import models
import utils


def experement_1():
    config_path = "./configs/"
    names = utils.get_names("configs/device_configs/names.yaml")["ordered_data"]
    folder_path = utils.get_subfolder_path("./data")
    with devices.device.Device(folder_path, config_path, names) as device:
        model = models.MultiListModel(names)
        agent = agents.Agent(folder_path, config_path, names, model, device)
        agent.measure_detectors_noise()
        agent.save_full_data("start")
        agent.save_full_data("end")
        density = agent.density_generator.get_values(100)
        agent.measure_hyper_parameters()


def measure_parameters():
    config_path = "./configs/"
    names = utils.get_names("configs/device_configs/names.yaml")["ordered_data"]
    folder_path = utils.get_subfolder_path("./data")
    with devices.Device(folder_path, config_path, names, plot_figs=True) as device:
        model = models.MultiListModel(names)
        agent = agents.Agent(folder_path, config_path, names, model, device)
        agent.measure_detectors_noise()
        agent.save_full_data("start")
        agent.save_full_data("end")
        _ = agent.density_generator.get_values(100)
        agent.measure_hyper_parameters()
        agent.save_full_data("end")

def _get_parameters():
    config_path = "./configs/"
    names = utils.get_names("configs/device_configs/names.yaml")["ordered_data"]
    folder_path = utils.get_subfolder_path("./data")
    with devices.Device(folder_path, config_path, names, plot_figs=True) as device:
        time.sleep(4)
        for i in range(100):
            _ = device.get_parameters()

def _mean_var():
    config_path = "./configs/"
    names = utils.get_names("configs/device_configs/names.yaml")["ordered_data"]
    folder_path = utils.get_subfolder_path("./data")
    with devices.Device(folder_path, config_path, names, plot_figs=True) as device:
        model = models.MultiListModel(names)
        agent = agents.Agent(folder_path, config_path, names, model, device)
        time.sleep(10)
        for i in range(20):
            _ = device.get_parameters()
            agent.measure_detectors_noise()

def _hyper_parameters():
    config_path = "./configs/"
    names = utils.get_names("configs/device_configs/names.yaml")["ordered_data"]
    folder_path = utils.get_subfolder_path("./data")
    with devices.LaserSetup(folder_path, config_path, names, plot_figs=False, read_every=0.2) as las_setup:
        agent = agents.Agent(folder_path, config_path, names, las_setup)
        correctors = ["1MXY1_Y", "1MXY1_X"]
        #agent.characterize_system_parameters()
        agent.optimize_corrector_values(correctors)

if __name__ == "__main__":
    _hyper_parameters()












