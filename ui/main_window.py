
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout

from ui.Data_analysis_extraction import TiffViewer
from ui.panels.file_controls import FileControls
from ui.panels.intensity_controls import IntensityControls
from ui.panels.frame_controls import FrameControls
from ui.panels.peak_controls import PeakControls
from ui.panels.export_controls import ExportControls
from utils.settings_manager import SettingsManager
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data analysis and extraction GUI")
        self.setGeometry(100, 100, 1000, 600)

        self.settings = SettingsManager()
        self.tiff_viewer = TiffViewer(self.settings)

        file_controls = FileControls(self.tiff_viewer, self)
        intensity_controls = IntensityControls(self.tiff_viewer)
        frame_controls = FrameControls(self.tiff_viewer)
        peak_controls = PeakControls(self.tiff_viewer)
        export_controls = ExportControls(self.tiff_viewer)

        left_panel = QVBoxLayout()
        left_panel.addLayout(file_controls)
        left_panel.addWidget(self.tiff_viewer, stretch=1)
        left_panel.addLayout(intensity_controls)
        left_panel.addLayout(frame_controls)

        right_panel = QVBoxLayout()
        right_panel.addLayout(peak_controls)
        right_panel.addLayout(export_controls)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel, stretch=1)
        main_layout.addLayout(right_panel)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
