import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMenu, QSlider, QLabel, QVBoxLayout, QMainWindow
)
from PyQt6.QtCore import Qt, QPoint
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QMenu, QSlider, QLabel, QVBoxLayout, QMainWindow
)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QPainter, QColor, QAction

from typing import Type
from abc import ABC, abstractmethod
from functools import partial

class Slider(QWidget):
    MAXIMUM = 100
    STEP = 1
    def __init__(self, parent: QMainWindow | QWidget, default_value: float, limits: tuple[float, float]):
        super().__init__(parent)
        # Horizontal slider
        layout = QVBoxLayout()
        self.limits = limits
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.MAXIMUM)
        self.slider.setTickInterval(20)
        self.slider.setSingleStep(self.STEP)
        self.slider.setSliderPosition(self._value_to_pos(default_value))

        # Label to show the slider value
        self.label = QLabel(f"Current: {default_value:.3f}")

        layout.addWidget(self.slider)
        layout.addWidget(self.label)

        self.setLayout(layout)
        #self.slider.valueChanged.connect(self.update_label)

    def _value_to_pos(self, value: float):
        norm_value = (value - self.limits[0]) / (self.limits[1] - self.limits[0])
        pos = int(norm_value * self.MAXIMUM)
        return pos

    def _pos_to_value(self, pos: int):
        norm_value = pos / 100
        value = norm_value * (self.limits[1] - self.limits[0]) + self.limits[0]
        return value

    def set_value(self, value: float):
        pos = self._value_to_pos(value)
        self.slider.setTickPosition(pos)
        self.update_label(pos)

    def update_label(self, pos: int):
        print(pos)
        self.label.setText(f'Current: {self._pos_to_value(pos):.3f}')


class SliderWindow(QMainWindow):
    def __init__(self, parent: QWidget, option_name: str, default_value: float, limits: tuple[float, float] | None):
        super().__init__(parent)
        self.slider = Slider(self, default_value, limits)
        self.setWindowTitle(f"Slider for {option_name}")
        self.setGeometry(100, 100, 300, 100)
        self.setCentralWidget(self.slider)


class MeasureWindow(QMainWindow):
    def __init__(self, parent: QWidget, option_name: str, default_value: float, limits: tuple[float, float] | None=None):
        super().__init__(parent)
        self.label = QLabel(option_name)
        self.setWindowTitle(f"Measurments {option_name}")
        self.setGeometry(100, 100, 300, 100)
        self.setCentralWidget(self.label)
        self.label.setText(f"Current: {default_value:.3f}")
        self.value = default_value

    def set_value(self, value: float):
        self.value = value
        self.label.setText(f"Current: {value:.3f}")

    def get_value(self):
        return self.value


class DesignWidget(QWidget):
    def __init__(self, parent: QMainWindow, color: tuple[int, int, int],
                 position: tuple[int, int, int, int], options: list[str],
                 default_values: list[float],  limits: list[tuple[float, float]] | None=None):
        super().__init__(parent)
        context_window = self.context_window()
        self.color = color
        self.position = position
        self.options = options
        self.default_values = default_values
        if limits is None:
            self.limits = [None] * len(self.options)
        else:
            self.limits = limits
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setGeometry(position[0], position[1], position[2] + 1, position[3] + 1)
        self.context_windows = self.create_context_windows(context_window)

    def paintEvent(self, event):
        raise NotImplementedError

    def context_window(self):
        raise NotImplementedError


    def show_context_menu(self, position: QPoint):
        context_menu = QMenu(self)
        for option in self.options:
            action = QAction(option, self)
            action.triggered.connect(partial(self.open_context_window, option))
            context_menu.addAction(action)
        context_menu.exec(self.mapToGlobal(position))

    def create_context_windows(self, window: Type[MeasureWindow] | Type[SliderWindow]):
        context_windows = {}
        for option, default_value, limits in zip(self.options, self.default_values, self.limits):
            context_window = window(self, option, default_value, limits)
            context_windows[option] = context_window
        return context_windows

    def open_context_window(self, option_name: str):
        self.context_windows[option_name].show()

    def mousePressEvent(self, event):
        pass
        # Check if left button is pressed
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if the click is within the rectangle
            if 150 <= event.pos().x() <= 250 and 150 <= event.pos().y() <= 250:
                self.show_context_menu(event.pos())

class MeasureWidget(DesignWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(*self.color))
        painter.drawEllipse(0, 0, self.position[2], self.position[3])  # Drawing a red rectangle in the center

    def context_window(self):
        return MeasureWindow


class SettingsWidget(DesignWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(*self.color))
        painter.drawRect(0, 0, self.position[2], self.position[3])  # Drawing a red rectangle in the center

    def context_window(self):
        return SliderWindow

