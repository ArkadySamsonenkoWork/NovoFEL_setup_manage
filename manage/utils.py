import re
import typing as tp
from pathlib import Path
import time

import yaml


def yaml_save(folder_data_path: tp.Union[str, Path], name: str, data: dict[str, tp.Any]):
    #timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    path_file = f"{name}.yaml"
    path_file = Path(folder_data_path) / Path(path_file)
    with open(path_file, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def get_typed_names(element_names):
    DETECTOR_PREFIXES = ("BPM",)
    SOLENOID_PREFIXES = ("MS",)
    CORRECTOR_PREFIXES = ("MXY", "MQ")
    def filter_names(prefixes):
        return [name for name in element_names if re.sub(r"[0-9]", "", name).startswith(prefixes)]

    return (
        filter_names(DETECTOR_PREFIXES),
        filter_names(SOLENOID_PREFIXES),
        filter_names(CORRECTOR_PREFIXES),)

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
