from PyQt5.QtWidgets import QWidget, QHBoxLayout, QLabel, QLineEdit, QSlider
from PyQt5.QtCore import Qt

class FrameControls(QWidget):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.input_field = QLineEdit("0")
        self.input_field.setFixedWidth(50)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.setMinimum(0)
        self.slider.setSingleStep(1)

    def init_with_controller(self, controller):
        self.controller = controller
        self.init_ui()

    def init_ui(self):
        self.slider.valueChanged.connect(self.sync_slider)
        self.slider.valueChanged.connect(self.controller.update_frame)
        self.input_field.returnPressed.connect(self.apply_text_input)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Frame:"))
        layout.addWidget(self.slider)
        layout.addWidget(self.input_field)
        self.setLayout(layout)

    def set_maximum(self, val):
        self.slider.setMaximum(val)

    def set_slider(self, val):
        self.slider.setValue(val)
        self.sync_slider(val)

    def sync_slider(self, val):
        self.input_field.setText(str(val))

    def apply_text_input(self):
        try:
            val = int(self.input_field.text())
            self.controller.update_frame(val)
            self.set_slider(val)
        except ValueError:
            pass
