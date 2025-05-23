from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from ui.panels.tiff_canvas import TiffCanvas
from ui.panels.file_controls import FileControls
from ui.panels.intensity_controls import IntensityControls
from ui.panels.frame_controls import FrameControls
from ui.panels.peak_controls import PeakControls
from ui.panels.export_controls import ExportControls
from ui.panels.trace_plot import TracePlotWidget
from utils.settings_manager import SettingsManager
from ui.data_analysis_gui import DataAnalysisGUI

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data analysis and extraction GUI")
        self.setGeometry(100, 100, 1000, 600)

        # Instantiate core widgets
        self.settings = SettingsManager()
        self.canvas = TiffCanvas()
        self.trace_plot = TracePlotWidget()

        # Control panels
        self.file_controls = FileControls()
        self.intensity_controls = IntensityControls()
        self.frame_controls = FrameControls()
        self.peak_controls = PeakControls()
        self.export_controls = ExportControls()

        # In a second step, pass all the widgets to the controller
        self.controller = DataAnalysisGUI(
            settings=self.settings,
            canvas=self.canvas,
            trace_plot=self.trace_plot,
            file_controls=self.file_controls,
            intensity_controls = self.intensity_controls,
            frame_controls=self.frame_controls,
            peak_controls=self.peak_controls,
            export_controls=self.export_controls
        )

        # Phase 4: Let the controls initialize themselves now that controller is known
        self.file_controls.init_with_controller(self.controller, self)
        self.intensity_controls.init_with_controller(self.controller)
        self.frame_controls.init_with_controller(self.controller)
        self.peak_controls.init_with_controller(self.controller)
        self.export_controls.init_with_controller(self.controller)

        # Layout left and right side
        left_layout = QVBoxLayout()
        left_layout.addWidget(self.file_controls)
        left_layout.addWidget(self.canvas, stretch=1)
        left_layout.addWidget(self.intensity_controls)
        left_layout.addWidget(self.frame_controls)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.peak_controls)
        right_layout.addWidget(self.export_controls)
        right_layout.addWidget(self.trace_plot)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)