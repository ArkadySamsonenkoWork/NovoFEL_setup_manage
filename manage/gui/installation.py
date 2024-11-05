import sys
from collections import  OrderedDict
import typing

from PyQt6.QtWidgets import (
    QApplication, QWidget, QMenu, QSlider, QLabel, QVBoxLayout, QMainWindow, QGridLayout, QLineEdit,
)
from PyQt6.QtCore import Qt, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QAction

from widgets import SettingsWidget, MeasureWidget, Slider

WIDGET_WIDTH = 60
WIDGET_HEIGHT = 120
WIDGET_DISTANCE = 20
WINDOW_HEIGHT = 100
WINDOW_SIZE = (1600, 800)

from typing import NamedTuple


class SetupConfig(NamedTuple):
    type: str
    options: list[str]
    limits: dict[str: tuple[float, float]]
    default: dict[str: float]

class InstallationWindow(QMainWindow):
    def __init__(self, setups: OrderedDict[str: SetupConfig]):
        super().__init__()
        self.tot_names = len(setups)
        self.setWindowTitle(f"Installation")
        self.resize(*WINDOW_SIZE)

        centralwidget = QWidget()
        #slider = SettingsWidget(self, (40, 40, 100), (90, 90, 100, 100), ["X"])
        #slider = QLabel(self)

        #slider.setGeometry(150, 150, 100, 100)
        self.widgets = self._get_widgets(setups)
        #self.setCentralWidget(centralwidget)
        #layout = self._get_layout()
        #self.setCentralWidget(centralwidget)

    def _get_positions(self):
        shift = WIDGET_DISTANCE
        for i in range(self.tot_names):
            position =\
                (shift, WINDOW_SIZE[1] // 2, WIDGET_WIDTH, WIDGET_HEIGHT)
            shift += WIDGET_DISTANCE + WIDGET_WIDTH
            yield position

    def _get_color(self, widget_type: str) -> tuple[int, int, int]:
        if widget_type == "MS":
            color = (255, 255, 0)
        elif widget_type == "MXY":
            color = (0, 0, 255)
        elif widget_type == "BPM":
            color = (255, 0, 0)
        else:
            color = (0, 0, 0)
        return color

    def _get_widgets(self, setups: OrderedDict[str: dict[str, SetupConfig]]):
        setting_widgets = OrderedDict()
        positions = self._get_positions()
        for name, parameters in setups.items():
            color = self._get_color(parameters["type"])
            position = next(positions)
            options = parameters["options"]
            limits = parameters["limits"]
            default = parameters["default"]
            if parameters["type"] == "BPM":
                widget = MeasureWidget(self, color, position, options, default, limits)
            else:
                widget = SettingsWidget(self, color, position, options, default, limits)
            widget.setGeometry(*position)
            setting_widgets[name] = widget
        return setting_widgets


class InstallationManager(QWidget):
    def __init__(self, setups: OrderedDict[str: SetupConfig]):
        super().__init__()
        self.install_window = InstallationWindow(setups)
        #self.InstallationWindow.widgets["1MS1"].context_windows["X"].slider.slider.valueChanged.connect(self.check)
        self.install_window.widgets["1MS1"].context_windows["X"].slider.slider.valueChanged.connect(self.install_window.widgets["1MS1"].context_windows["X"].slider.update_label)
    def check(self, value):
        print(123)

    def show(self):
        self.install_window.show()






if __name__ == "__main__":
    setups = OrderedDict()
    setups["1MS1"] = {"type": "MS", "options": ("X", "Y"), "limits": [(-4.2, 4.5), (-3, 2.3)], "default": [4.2, 2.1]}
    setups["1MS2"] = {"type": "MS", "options": ("X", "Y"), "limits": [(-4.2, 4.5), (-3, 2.3)], "default": [4.2, 2.1]}
    setups["1MS3"] = {"type": "BPM", "options": ("X", "Y"), "limits": [(-4.2, 4.5), (-3, 2.3)], "default": [4.2, 2.1]}
    app = QApplication(sys.argv)
    main_window = InstallationManager(setups)
    main_window.show()
    sys.exit(app.exec())