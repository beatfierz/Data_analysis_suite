
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QLineEdit

def IntensityControls(tiff_viewer):
    intensity_label = QLabel("Max Intensity:")
    intensity_input = QLineEdit("1000")
    intensity_input.setFixedWidth(50)
    intensity_input.returnPressed.connect(lambda: update(tiff_viewer, intensity_input))

    tiff_viewer.intensity_slider.valueChanged.connect(
        lambda val: intensity_input.setText(str(val))
    )

    layout = QHBoxLayout()
    layout.addWidget(intensity_label)
    layout.addWidget(tiff_viewer.intensity_slider)
    layout.addWidget(intensity_input)
    return layout

def update(tiff_viewer, field):
    try:
        val = int(field.text())
        tiff_viewer.set_intensity_value(val)
    except ValueError:
        pass
