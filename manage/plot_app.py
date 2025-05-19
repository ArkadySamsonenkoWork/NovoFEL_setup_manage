import dash
import plotly.graph_objects as go
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

import queue
import threading
import typing as tp
import logging
import math
import re

logging.getLogger('werkzeug').setLevel(logging.ERROR)

class Plotter:
    DETECTOR_PREFIXES = ("BPM",)
    SOLENOID_PREFIXES = ("MS",)
    CORRECTOR_PREFIXES = ("MXY", "MQ")

    def __init__(self, names: list[str], measurement_queue: queue.Queue, mexlen: int):
        self.element_names = names
        self.measurement_queue = measurement_queue
        self.detector_names, self.solenoid_names, self.corrector_names = self._get_typed_names()
        self.typed_names = {"detectors": self.detector_names,
                            "solenoids": self.solenoid_names,
                            "correctors": self.corrector_names}
        self.parameters = {name: [] for name in self.element_names}
        self.steps = []
        self.figures = self._create_figures()
        self.maxlen = mexlen

    def update_from_queue(self):
        """
        Update figures with measurements from the queue.

        :return: Boolean indicating if new data was processed
        """
        try:
            # Try to get the latest measurement without blocking
            measurement = self.measurement_queue.get_nowait()

            # Update the figure with new measurement
            if len(self.steps) > 0:
                step = self.steps[-1] + 1
            else:
                step = 0
            self.steps.append(step)

            for name, value in measurement.items():
                self.parameters[name].append(value)

            # Update traces for each type of element
            for typed_name, names in self.typed_names.items():
                for subplot_index, name in enumerate(names, start=0):
                    trace = self.figures[typed_name].data[subplot_index]
                    trace.x = self.steps
                    trace.y = self.parameters[name]

            if len(self.steps) > self.maxlen:
                self.steps = self.steps[self.maxlen // 2:]
                for name, value in measurement.items():
                    self.parameters[name] = self.parameters[name][self.maxlen // 2:]
                #self.parameters = self.parameters[self.max_len // 2:]

            return True
        except queue.Empty:
            return False

    def _get_max_lines_cols(self, names):
        if len(names) <= 4:
            return len(names), 1
        elif len(names) <= 8:
            return math.ceil(len(names) // 2), 2
        else:
            return math.ceil(len(names) // 3), 3

    def _get_col_line_indxes(self, names):
        max_indexes = self._get_max_lines_cols(names)
        for idx, name in enumerate(names):
            yield name, (1 + idx // max_indexes[1], 1 + idx % max_indexes[1])

    def _create_figures(self):
        figures = {}
        for typed_name, names in self.typed_names.items():
            figure = make_subplots(*self._get_max_lines_cols(names), shared_xaxes=True)
            for name, idxs in self._get_col_line_indxes(names):
                figure.add_trace(go.Scatter(x=[], y=[], name=name), row=idxs[0], col=idxs[1])
            figures[typed_name] = figure
        return figures

    def _get_typed_names(self):
        def filter_names(prefixes):
            return [name for name in self.element_names if re.sub(r"[0-9]", "", name).startswith(prefixes)]

        return (
            filter_names(self.DETECTOR_PREFIXES),
            filter_names(self.SOLENOID_PREFIXES),
            filter_names(self.CORRECTOR_PREFIXES),)


class PlotApp:
    def __init__(self, names: list[str], mexlen: int = 100):
        self.measurement_queue = queue.Queue(maxsize=mexlen)
        self.plotter = Plotter(names, self.measurement_queue, mexlen)
        self._run_thread()

    def update(self, parameters: dict[str, tp.Any]):
        try:
            self.measurement_queue.put(parameters, block=False)
        except queue.Full:
            # If queue is full, remove the oldest item and add new one
            try:
                self.measurement_queue.get_nowait()
            except queue.Empty:
                pass
            self.measurement_queue.put(parameters)

    def _create_dash_app(self):
        """
        Create and configure the Dash application.

        :param plotter: Plotter instance with measurement queue
        :return: Configured Dash app
        """
        app = dash.Dash(__name__)

        # Define the layout of the Dash app
        app.layout = html.Div(
            [
                dcc.Graph(id="detectors-plot", figure=self.plotter.figures["detectors"]),
                dcc.Graph(id="solenoids-plot", figure=self.plotter.figures["solenoids"]),
                dcc.Graph(id="correctors-plot", figure=self.plotter.figures["correctors"]),
                dcc.Interval(id="animateInterval", interval=1000, n_intervals=0),
            ]
        )

        @app.callback(
            [Output("detectors-plot", "figure"),
             Output("solenoids-plot", "figure"),
             Output("correctors-plot", "figure")],
            Input("animateInterval", "n_intervals"),
        )
        def update_plots(n):
            self.plotter.update_from_queue()
            return self.plotter.figures["detectors"], self.plotter.figures["solenoids"], self.plotter.figures["correctors"]
        return app

    def _run_dash_server(self, app, port=8050):
        """
        Run Dash server in a separate thread.

        :param app: Dash application
        :param port: Port to run the server on
        """
        app.run_server(debug=False, port=port, use_reloader=False)

    def _run_thread(self):
        dash_app = self._create_dash_app()
        # Option 1: Run server in a separate thread
        server_thread = threading.Thread(target=self._run_dash_server, args=(dash_app,), daemon=True)
        server_thread.start()
