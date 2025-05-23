from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import Qt

class FileControls(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.parent_window = None
        self.path_label = QLabel("No file loaded")
        self.path_label.setStyleSheet("font-style: italic; color: gray")
        self.load_button = QPushButton("Load TIFF")
        self.load_button.setMaximumWidth(100)
        self.quit_button = QPushButton("Quit")
        self.quit_button.setMaximumWidth(100)

    def init_with_controller(self, controller, parent):
        self.controller = controller
        self.parent_window = parent
        self.init_ui()

    def init_ui(self):
        # --- Button callbacks ---
        self.load_button.clicked.connect(self.controller.load_tiff)
        self.quit_button.clicked.connect(self.parent_window.close)

        button_row = QHBoxLayout()
        button_row.addWidget(self.load_button)
        button_row.addWidget(self.quit_button)
        button_row.addStretch()
        button_row.setAlignment(Qt.AlignLeft)

        # --- Final Layout ---
        layout = QVBoxLayout()
        layout.addLayout(button_row)
        layout.addWidget(self.path_label)
        layout.addStretch()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def set_path_label(self, path):
        self.path_label.setText(path)

    def get_path_label(self):
        return self.path_label.text()
