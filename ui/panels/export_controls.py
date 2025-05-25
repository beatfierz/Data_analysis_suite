from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout, QCheckBox
from PyQt5.QtCore import Qt

class ExportControls(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.panel_label = QLabel("--- Export parameters ---")
        self.sel_peak_button = QPushButton("Select peak")
        self.sel_peak_button.setMaximumWidth(120)
        self.desel_peak_button = QPushButton("Deselect peak")
        self.desel_peak_button.setMaximumWidth(120)
        self.selAll_peak_button = QPushButton("Select all peaks")
        self.selAll_peak_button.setMaximumWidth(120)
        self.clcAll_peak_button = QPushButton("Clear selection")
        self.clcAll_peak_button.setMaximumWidth(120)
        self.extract_button = QPushButton("Extract trace")
        self.extract_button.setMaximumWidth(120)
        self.fit_chk = QCheckBox("Enable stepfit")
        self.fit_chk.setChecked(False)

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):

        # --- Peak selection buttons callbacks---
        self.sel_peak_button.clicked.connect(self.controller.select_peaks)
        self.desel_peak_button.clicked.connect(self.controller.deselect_peaks)
        self.selAll_peak_button.clicked.connect(self.controller.select_all_peaks)
        self.clcAll_peak_button.clicked.connect(self.controller.clcAll_peaks)
        self.extract_button.clicked.connect(self.controller.extract_trace)

        titel_row = QHBoxLayout()
        peak_button_row = QHBoxLayout()
        titel_row.addWidget(self.panel_label)
        peak_button_row.addWidget(self.sel_peak_button)
        peak_button_row.addWidget(self.desel_peak_button)
        peak_button_row.addWidget(self.selAll_peak_button)
        peak_button_row.addWidget(self.clcAll_peak_button)
        peak_button_row.addStretch()
        peak_button_row.setAlignment(Qt.AlignLeft)

        sel_fit_row = QHBoxLayout()
        sel_fit_row.addWidget(self.fit_chk)
        sel_fit_row.addStretch()
        sel_fit_row.setAlignment(Qt.AlignLeft)

        edit_button_row = QHBoxLayout()
        edit_button_row.addWidget(self.extract_button)
        edit_button_row.addStretch()
        edit_button_row.setAlignment(Qt.AlignLeft)

        # --- Final Layout ---
        layout = QVBoxLayout()
        layout.addLayout(titel_row)
        layout.addLayout(peak_button_row)
        layout.addLayout(sel_fit_row)
        layout.addLayout(edit_button_row)
        layout.addStretch()

        self.setLayout(layout)

    def stepfit_enabled(self):
        return self.fit_chk.isChecked()

