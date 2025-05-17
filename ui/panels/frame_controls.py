
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QLineEdit

def FrameControls(tiff_viewer):
    frame_label = QLabel("Frame:")
    frame_input = QLineEdit("0")
    frame_input.setFixedWidth(50)
    frame_input.returnPressed.connect(lambda: update(tiff_viewer, frame_input))

    tiff_viewer.frame_slider.valueChanged.connect(
        lambda val: frame_input.setText(str(val))
    )

    layout = QHBoxLayout()
    layout.addWidget(frame_label)
    layout.addWidget(tiff_viewer.frame_slider)
    layout.addWidget(frame_input)
    return layout

def update(tiff_viewer, field):
    try:
        val = int(field.text())
        tiff_viewer.set_frame_value(val)
    except ValueError:
        pass
