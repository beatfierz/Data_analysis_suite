from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QSlider
from PyQt5.QtCore import Qt

class IntensityControls(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.input_field = QLineEdit("1000")
        self.input_field.setFixedWidth(50)
        self.input_field.returnPressed.connect(self.apply_text_input)

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(1)
        slider.setMaximum(65280)
        slider.setValue(1000)
        slider.valueChanged.connect(self.sync_slider)
        slider.valueChanged.connect(self.controller.update_intensity)

        self.controller.intensity_slider = slider  # register with viewer

        layout = QHBoxLayout()
        layout.addWidget(QLabel("Max Intensity:"))
        layout.addWidget(slider)
        layout.addWidget(self.input_field)
        self.setLayout(layout)

    def sync_slider(self, val):
        self.input_field.setText(str(val))

    def apply_text_input(self):
        try:
            val = int(self.input_field.text())
            self.controller.set_intensity_value(val)
            self.controller.intensity_slider.setValue(val)
        except ValueError:
            pass
