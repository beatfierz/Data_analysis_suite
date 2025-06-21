from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt
from utils.frame_skipping import parse_skip_frames

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
        self.pos_info_input = QLineEdit("N/A")
        self.pos_info_input.setFixedWidth(50)
        self.pos_info_input.setToolTip("Specify frames for positional/alignment info. Syntax: excluded frames: x1,x2, "
                                       "...; y1:y2; y2: is the spacing of all frames excluded starting with y1")
        self.txt_pos_info_input = QLabel("Pos. info frame #:")

        self.controller = None
        self.peak_rad_input = QLineEdit("3")
        self.peak_rad_input.setFixedWidth(50)
        self.peak_rad_input.setToolTip("Specify radius for peak selection in pixels")
        self.txt_peak_rad_input = QLabel("Peak sel. radius:")

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        #--- callbacks ---
        self.peak_rad_input.returnPressed.connect(self.update_circle_size)
        self.pos_info_input.returnPressed.connect(self.update_frame_skip)

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
        return int(self.peak_rad_input.text())

    def update_circle_size(self):
        self.controller.circle_radius = int(self.peak_rad_input.text())
        self.controller.canvas.set_circle_radius(int(self.peak_rad_input.text()))
        self.controller.canvas.update()

    def update_frame_skip(self):
        if self.controller.current_trace:
            skip = parse_skip_frames(self.pos_info_input.text(), self.controller.tiff_stack.shape[0])
        else:
            skip = []
        self.controller.skip_frames = skip
        return skip