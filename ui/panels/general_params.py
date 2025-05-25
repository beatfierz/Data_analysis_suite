from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

class GeneralParams(QWidget):
    def __init__(self):
        super().__init__()
        self.panel_label = QLabel("--- General parameters ---")

        self.controller = None
        self.timebase_input = QLineEdit("0.1")
        self.timebase_input.setFixedWidth(50)
        self.timebase_input.setToolTip("Movie timescale")
        self.txt_timebase_input = QLabel("Movie timescale:")

        self.controller = None
        self.pos_info_input = QLineEdit("NA")
        self.pos_info_input.setFixedWidth(50)
        self.pos_info_input.setToolTip("Specify frames for positional/alignment info")
        self.txt_pos_info_input = QLabel("Pos. info frame #:")

        self.controller = None
        self.peak_rad_input = QLineEdit("2")
        self.peak_rad_input.setFixedWidth(50)
        self.peak_rad_input.setToolTip("Specify frames for positional/alignment info")
        self.txt_peak_rad_input = QLabel("Peak sel. radius:")

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        # --- Timescale input ---
        title_row = QHBoxLayout()
        genparam_row1 = QHBoxLayout()
        genparam_row2 = QHBoxLayout()
        title_row.addWidget(self.panel_label)
        genparam_row1.addWidget(self.txt_timebase_input)
        genparam_row1.addWidget(self.timebase_input)
        genparam_row1.addWidget(self.txt_pos_info_input)
        genparam_row1.addWidget(self.pos_info_input)
        genparam_row2.addWidget(self.txt_peak_rad_input)
        genparam_row2.addWidget(self.peak_rad_input)

        title_row.addStretch()
        title_row.setAlignment(Qt.AlignLeft)
        genparam_row1.addStretch()
        genparam_row1.setAlignment(Qt.AlignLeft)
        genparam_row2.addStretch()
        genparam_row2.setAlignment(Qt.AlignLeft)

        # --- Final Layout ---
        layout = QVBoxLayout()
        layout.addLayout(title_row)
        layout.addLayout(genparam_row1)
        layout.addLayout(genparam_row2)
        layout.addStretch()

        self.setLayout(layout)

    def get_timebase(self) -> float:
        return float(self.timebase_input.text())

    def get_pos_info(self) -> float:
        return self.pos_info_input.text()

    def get_peak_rad(self) -> float:
        return float(self.peak_rad_input.text())
