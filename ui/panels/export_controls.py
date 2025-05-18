

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt

def ExportControls(tiff_viewer):
    # --- peak detection elements ---
    timebase_row = QHBoxLayout()
    timebase_input = QLineEdit("0.1")
    timebase_input.setFixedWidth(50)
    timebase_input.setToolTip("Movie timescale")
    timebase_row.addWidget(QLabel("Movie timescale:"))
    timebase_row.addWidget(timebase_input)
    timebase_row.addStretch()
    timebase_row.setAlignment(Qt.AlignLeft)

    # --- peaklist buttons ---
    sel_peak_button = QPushButton("Select peak")
    sel_peak_button.setMaximumWidth(120)
    sel_peak_button.clicked.connect(tiff_viewer.select_peaks)

    desel_peak_button = QPushButton("Deselect peak")
    desel_peak_button.setMaximumWidth(120)
    desel_peak_button.clicked.connect(tiff_viewer.deselect_peaks)

    selAll_peak_button = QPushButton("Select all peaks")
    selAll_peak_button.setMaximumWidth(120)
    selAll_peak_button.clicked.connect(tiff_viewer.select_all_peaks)

    clcAll_peak_button = QPushButton("Clear selection")
    clcAll_peak_button.setMaximumWidth(120)
    clcAll_peak_button.clicked.connect(tiff_viewer.clcAll_peaks)

    extract_button = QPushButton("Extract trace")
    extract_button.setMaximumWidth(120)
    extract_button.clicked.connect(tiff_viewer.extract_trace)

    peak_button_row = QHBoxLayout()
    peak_button_row.addWidget(sel_peak_button)
    peak_button_row.addWidget(desel_peak_button)
    peak_button_row.addWidget(selAll_peak_button)
    peak_button_row.addWidget(clcAll_peak_button)
    peak_button_row.addStretch()
    peak_button_row.setAlignment(Qt.AlignLeft)

    edit_button_row = QHBoxLayout()
    edit_button_row.addWidget(extract_button)
    edit_button_row.addStretch()
    edit_button_row.setAlignment(Qt.AlignLeft)

    layout = QVBoxLayout()
    layout.addLayout(timebase_row)
    layout.addLayout(peak_button_row)
    layout.addLayout(edit_button_row)
    layout.addStretch()
    return layout
