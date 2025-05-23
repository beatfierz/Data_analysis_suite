
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QLineEdit, QPushButton, QHBoxLayout

def PeakControls(tiff_viewer):
    # --- peak detection elements ---
    input_row = QHBoxLayout()
    peak_input = QLineEdit("1")
    peak_input.setFixedWidth(50)
    peak_input.setToolTip("Frame for peak detection")
    tiff_viewer.peak_frame_input = peak_input
    input_row.addWidget(QLabel("Frame for peak detection:"))
    input_row.addWidget(peak_input)
    input_row.addStretch()
    input_row.setAlignment(Qt.AlignLeft)

    # --- peaklist buttons ---
    peak_button = QPushButton("Detect Peaks")
    peak_button.setMaximumWidth(120)
    peak_button.clicked.connect(tiff_viewer.detect_peaks)

    save_button = QPushButton("Save Peaklist")
    save_button.setMaximumWidth(120)
    save_button.clicked.connect(tiff_viewer.save_peaklist)

    load_button = QPushButton("Load Peaklist")
    load_button.setMaximumWidth(120)
    load_button.clicked.connect(tiff_viewer.load_peaklist)

    add_button = QPushButton("Add Peak")
    add_button.setMaximumWidth(120)
    add_button.clicked.connect(tiff_viewer.activate_add_peak_mode)

    delete_button = QPushButton("Delete Peak")
    delete_button.setMaximumWidth(120)
    delete_button.clicked.connect(tiff_viewer.activate_delete_peak_mode)

    clear_button = QPushButton("Clear All Peaks")
    clear_button.setMaximumWidth(120)
    clear_button.clicked.connect(tiff_viewer.clear_all_peaks)

    peak_button_row = QHBoxLayout()
    peak_button_row.addWidget(peak_button)
    peak_button_row.addWidget(save_button)
    peak_button_row.addWidget(load_button)
    peak_button_row.addStretch()
    peak_button_row.setAlignment(Qt.AlignLeft)

    edit_button_row = QHBoxLayout()
    edit_button_row.addWidget(add_button)
    edit_button_row.addWidget(delete_button)
    edit_button_row.addWidget(clear_button)
    edit_button_row.addStretch()
    edit_button_row.setAlignment(Qt.AlignLeft)

    threshold_layout = QHBoxLayout()
    threshold_layout.addWidget(QLabel("Detection Threshold:"))
    tiff_viewer.threshold_slider.setMinimumWidth(200)
    tiff_viewer.threshold_slider.setMaximumWidth(200)
    threshold_layout.addWidget(tiff_viewer.threshold_slider, alignment=Qt.AlignLeft)

    layout = QVBoxLayout()
    layout.addLayout(input_row)
    layout.addLayout(threshold_layout)
    layout.addLayout(peak_button_row)
    layout.addLayout(edit_button_row)
    layout.addStretch()
    return layout
