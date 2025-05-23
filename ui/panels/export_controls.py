from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

class ExportControls(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
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

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        # --- Timescale input ---
        timebase_row = QHBoxLayout()
        self.timebase_input = QLineEdit("0.1")
        self.timebase_input.setFixedWidth(50)
        self.timebase_input.setToolTip("Movie timescale")
        timebase_row.addWidget(QLabel("Movie timescale:"))
        timebase_row.addWidget(self.timebase_input)
        timebase_row.addStretch()
        timebase_row.setAlignment(Qt.AlignLeft)

        # --- Peak selection buttons callbacks---
        self.sel_peak_button.clicked.connect(self.controller.select_peaks)
        self.desel_peak_button.clicked.connect(self.controller.deselect_peaks)
        self.selAll_peak_button.clicked.connect(self.controller.select_all_peaks)
        self.clcAll_peak_button.clicked.connect(self.controller.clcAll_peaks)
        self.extract_button.clicked.connect(self.controller.extract_trace)

        peak_button_row = QHBoxLayout()
        peak_button_row.addWidget(self.sel_peak_button)
        peak_button_row.addWidget(self.desel_peak_button)
        peak_button_row.addWidget(self.selAll_peak_button)
        peak_button_row.addWidget(self.clcAll_peak_button)
        peak_button_row.addStretch()
        peak_button_row.setAlignment(Qt.AlignLeft)

        edit_button_row = QHBoxLayout()
        edit_button_row.addWidget(self.extract_button)
        edit_button_row.addStretch()
        edit_button_row.setAlignment(Qt.AlignLeft)

        # --- Final Layout ---
        layout = QVBoxLayout()
        layout.addLayout(timebase_row)
        layout.addLayout(peak_button_row)
        layout.addLayout(edit_button_row)
        layout.addStretch()

        self.setLayout(layout)

    def get_timebase(self) -> float:
        """Return the movie timescale as a float (default to 0.1 if invalid)."""
        try:
            return float(self.timebase_input.text())
        except ValueError:
            return 0.1
