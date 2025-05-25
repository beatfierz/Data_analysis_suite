from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSlider

class PeakControls(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        #--- panel title ---
        self.panel_label = QLabel("--- Peak detection and selection ---")

        #--- frame for detection ---
        self.peak_input = QLineEdit("1")
        self.peak_input.setFixedWidth(50)
        self.peak_input.setToolTip("Frame for peak detection")
        self.peak_input_label = QLabel("Frame for peak detection:")

        #--- detection level / threshold slider ---
        self.threshold_slider_label = QLabel("Detection Threshold:")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(65535)
        self.threshold_slider.setValue(1000)
        self.threshold_slider.setMinimumWidth(200)
        self.threshold_slider.setMaximumWidth(200)

        #--- buttons
        self.detect_peak_button = QPushButton("Detect Peaks")
        self.detect_peak_button.setMaximumWidth(120)
        self.save_peaklst_button = QPushButton("Save Peaklist")
        self.save_peaklst_button.setMaximumWidth(120)
        self.load_peaklst_button = QPushButton("Load Peaklist")
        self.load_peaklst_button.setMaximumWidth(120)

        self.add_peak_button = QPushButton("Add Peak")
        self.add_peak_button.setMaximumWidth(120)
        self.del_peak_button = QPushButton("Delete Peak")
        self.del_peak_button.setMaximumWidth(120)
        self.clear_allpeaks_button = QPushButton("Clear All Peaks")
        self.clear_allpeaks_button.setMaximumWidth(120)

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        #--- callbacks ---
        self.detect_peak_button.clicked.connect(self.controller.detect_peaks)
        self.save_peaklst_button.clicked.connect(self.controller.save_peaklist)
        self.load_peaklst_button.clicked.connect(self.controller.load_peaklist)
        self.add_peak_button.clicked.connect(self.controller.activate_add_peak_mode)
        self.del_peak_button.clicked.connect(self.controller.activate_delete_peak_mode)
        self.clear_allpeaks_button.clicked.connect(self.controller.clear_all_peaks)
        self.threshold_slider.valueChanged.connect(self.controller.update_peak_threshold)

        # --- panel title ---
        title_row = QHBoxLayout()
        title_row.addWidget(self.panel_label)
        # --- peak detection input ---
        input_row = QHBoxLayout()
        input_row.addWidget(self.peak_input_label)
        input_row.addWidget(self.peak_input)
        input_row.addStretch()
        input_row.setAlignment(Qt.AlignLeft)

        # --- threshold slider ---
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_slider_label)
        threshold_layout.addWidget(self.threshold_slider, alignment=Qt.AlignLeft)
        input_row.addStretch()
        threshold_layout.setAlignment(Qt.AlignLeft)

        # --- peaklist buttons ---
        peak_button_row = QHBoxLayout()
        peak_button_row.addWidget(self.detect_peak_button)
        peak_button_row.addWidget(self.save_peaklst_button)
        peak_button_row.addWidget(self.load_peaklst_button)
        peak_button_row.addStretch()
        peak_button_row.setAlignment(Qt.AlignLeft)

        # --- edit buttons ---
        edit_button_row = QHBoxLayout()
        edit_button_row.addWidget(self.add_peak_button)
        edit_button_row.addWidget(self.del_peak_button)
        edit_button_row.addWidget(self.clear_allpeaks_button)
        edit_button_row.addStretch()
        edit_button_row.setAlignment(Qt.AlignLeft)

        # --- assemble panel ---
        layout = QVBoxLayout()
        layout.addLayout(title_row)
        layout.addLayout(input_row)
        layout.addLayout(threshold_layout)
        layout.addLayout(peak_button_row)
        layout.addLayout(edit_button_row)

        layout.addStretch()
        self.setLayout(layout)

    def set_threshold(self, val):
        self.threshold_slider.setValue(int(val))

    def set_max_threshold(self, val):
        self.threshold_slider.setMaximum(int(val))

    def set_min_threshold(self, val):
        self.threshold_slider.setMinimum(int(val))

    def get_threshold(self):
        return self.threshold_slider.value()

    def set_pdet_frame(self, index):
        self.peak_input.setText(str(index))

    def get_pdet_frame(self):
        return int(self.peak_input.text())