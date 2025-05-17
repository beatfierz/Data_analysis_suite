
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout, QLabel

def FileControls(tiff_viewer, parent):
    load_button = QPushButton("Load TIFF")
    load_button.setMaximumWidth(100)

    quit_button = QPushButton("Quit")
    quit_button.setMaximumWidth(100)
    quit_button.clicked.connect(parent.close)

    path_label = QLabel("No file loaded")
    path_label.setStyleSheet("font-style: italic; color: gray")
    tiff_viewer.file_path_label = path_label  # expose to TiffViewer for updates

    load_button.clicked.connect(tiff_viewer.load_tiff)

    button_row = QHBoxLayout()
    button_row.addWidget(load_button)
    button_row.addWidget(quit_button)
    button_row.addStretch()

    layout = QVBoxLayout()
    layout.addLayout(button_row)
    layout.addWidget(path_label)

    return layout
