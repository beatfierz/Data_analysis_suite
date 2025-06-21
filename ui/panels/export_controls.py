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
        self.det_thrs_txt = QLabel("Detection threshold:")
        self.det_thrs_ed = QLineEdit("500")
        self.det_thrs_ed.setMaximumWidth(50)
        self.det_TN_txt = QLabel("Trace no.:")
        self.det_TN_ed = QLineEdit("N/A")
        self.det_TN_ed.setMaximumWidth(50)

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):

        # --- Peak selection buttons callbacks---
        self.sel_peak_button.clicked.connect(self.controller.mouse_select_peaks)
        self.desel_peak_button.clicked.connect(self.controller.deselect_peaks)
        self.selAll_peak_button.clicked.connect(self.controller.select_all_peaks)
        self.clcAll_peak_button.clicked.connect(self.controller.clcAll_peaks)
        self.extract_button.clicked.connect(self.controller.extract_trace)
        self.det_TN_ed.returnPressed.connect(self.controller.txt_select_peaks)

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

        trace_row = QHBoxLayout()
        trace_row.addWidget(self.det_TN_txt)
        trace_row.addWidget(self.det_TN_ed)
        trace_row.addStretch()
        trace_row.setAlignment(Qt.AlignLeft)

        edit_button_row = QHBoxLayout()
        edit_button_row.addWidget(self.extract_button)
        edit_button_row.addWidget(self.det_thrs_txt)
        edit_button_row.addWidget(self.det_thrs_ed)
        edit_button_row.addStretch()
        edit_button_row.setAlignment(Qt.AlignLeft)

        # --- Final Layout ---
        layout = QVBoxLayout()
        layout.addLayout(titel_row)
        layout.addLayout(trace_row)
        layout.addLayout(peak_button_row)
        layout.addLayout(sel_fit_row)
        layout.addLayout(edit_button_row)
        layout.addStretch()

        self.setLayout(layout)

    def stepfit_enabled(self):
        return self.fit_chk.isChecked()

    def set_trace_no(self, idx):
        self.det_TN_ed.blockSignals(True)
        self.det_TN_ed.setText(str(idx))
        self.det_TN_ed.blockSignals(False)

    def get_trace_no(self):
        return int(self.det_TN_ed.text())

    def get_threshold(self):
        return int(self.det_thrs_ed.text())

