from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QSlider
from PyQt5.QtCore import Qt

class IntensityControls(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.input_field = QLineEdit("1000")
        self.input_field.setFixedWidth(50)
        self.input_field.returnPressed.connect(self.apply_text_input)
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(1)
        self.intensity_slider.setMaximum(65280)
        self.intensity_slider.setValue(1)
        self.label = QLabel("Max Intensity:")

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        self.intensity_slider.valueChanged.connect(self.sync_slider)
        self.intensity_slider.valueChanged.connect(self.controller.update_intensity)

        # self.controller.intensity_slider = slider  # register with viewer

        layout = QHBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.intensity_slider)
        layout.addWidget(self.input_field)
        self.setLayout(layout)

    def sync_slider(self, val):
        self.input_field.setText(str(val))

    def apply_text_input(self):
        try:
            val = int(self.input_field.text())
            self.controller.update_intensity(val)
            self.intensity_slider.setValue(val)
        except ValueError:
            pass

    def set_intensity_range(self, val):
        self.intensity_slider.setMaximum(val)

    def set_intensity_value(self, val):
        self.intensity_slider.setValue(val)
        self.input_field.setText(str(val))

    def get_intensity_value(self):
        return self.intensity_slider.value()