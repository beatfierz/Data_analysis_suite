
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from ui.Data_analysis_extraction import TiffViewer
from ui.panels.file_controls import FileControls
from ui.panels.intensity_controls import IntensityControls
from ui.panels.frame_controls import FrameControls
from ui.panels.peak_controls import PeakControls
from ui.panels.export_controls import ExportControls
from utils.settings_manager import SettingsManager
from ui.panels.trace_plot import TracePlotWidget

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data analysis and extraction GUI")
        self.setGeometry(100, 100, 1000, 600)

        self.settings = SettingsManager()
        self.tiff_viewer = TiffViewer(self.settings)
        self.trace_plot = TracePlotWidget()
        self.tiff_viewer.trace_plot_widget = self.trace_plot

        # generate all the control elements
        file_controls = FileControls(self.tiff_viewer, self)
        intensity_controls = IntensityControls(self.tiff_viewer)
        frame_controls = FrameControls(self.tiff_viewer)
        export_controls = ExportControls(self.tiff_viewer)
        peak_controls = PeakControls(self.tiff_viewer)

        # organize the GUI
        left_panel = QVBoxLayout()
        left_panel.addWidget(file_controls)
        left_panel.addWidget(self.tiff_viewer, stretch=1)
        left_panel.addWidget(intensity_controls)
        left_panel.addWidget(frame_controls)

        right_panel = QVBoxLayout()
        right_panel.addWidget(peak_controls)
        right_panel.addWidget(export_controls)
        right_panel.addWidget(self.trace_plot)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(right_panel, stretch=3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
