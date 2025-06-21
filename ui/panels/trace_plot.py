from PyQt5.QtNetwork import QHttpPart
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt

class TraceDisplayPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure()  # figsize=(1, 3) horizontal shape
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.tick_params(axis='both', labelsize=8)
        self.curr_trace = []
        self.curr_timeb = []
        self.curr_fit = []
        self.controller = None

        # --- label ---
        self.label = QLabel('---Selected trace---')

        # --- control for intensity ---
        self.max_int_txt = QLabel('Max int:')
        self.max_int_ed = QLineEdit('auto')
        self.max_int_ed.setMaximumWidth(50)
        self.min_int_txt = QLabel('Min int:')
        self.min_int_ed = QLineEdit('auto')
        self.min_int_ed.setMaximumWidth(50)

        # --- arrange layout
        title_ctr_row = QHBoxLayout()
        title_ctr_row.addWidget(self.label)
        title_ctr_row.addSpacing(100)
        title_ctr_row.addWidget(self.min_int_txt)
        title_ctr_row.addWidget(self.min_int_ed)
        title_ctr_row.addWidget(self.max_int_txt)
        title_ctr_row.addWidget(self.max_int_ed)
        title_ctr_row.addStretch()
        title_ctr_row.setAlignment(Qt.AlignLeft)

        layout = QVBoxLayout()
        layout.addLayout(title_ctr_row)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def init_with_controller(self, controller):
        self.controller = controller
        self.max_int_ed.returnPressed.connect(self.update_trace_int)
        self.min_int_ed.returnPressed.connect(self.update_trace_int)

    def update_trace_int(self):
        min_int = self.get_min_int()
        max_int = self.get_max_int()
        self.ax.set_ylim(min_int, max_int)
        self.canvas.draw()

    def plot_trace(self, x_values, y_values, fitted=None):
        self.ax.clear()
        self.curr_trace = y_values
        self.curr_timeb = x_values
        self.curr_fit = fitted
        self.ax.plot(self.curr_timeb, self.curr_trace, linewidth=0.5)
        self.ax.set_xlim(x_values[0], x_values[-1])

        if fitted is not None:
            self.ax.plot(self.curr_timeb, self.curr_fit, linewidth=0.75, color='red')

        # Plot baseline as black line
        baseline = self.controller.current_trace_bl
        self.ax.axhline(y=baseline, color='black', linestyle='--',
                        linewidth=0.75, label='Baseline')

        # Plot fixed threshold line (baseline + threshold)
        thrs = self.controller.export_controls.get_threshold()
        self.ax.axhline(y=baseline+thrs, color='green', linestyle='--', linewidth=0.75, label='Threshold')

        # --- set limits ---
        if self.min_int_ed == 'auto':
            min_int = min(y_values)
        else:
            min_int = self.get_min_int()

        if self.max_int_ed == 'auto':
            max_int = max(y_values)
        else:
            max_int = self.get_max_int()

        self.ax.set_ylim(min_int, max_int)

        # --- labeling ---
        self.ax.set_xlabel("Time", fontsize=8)
        self.ax.set_ylabel("Intensity", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def get_max_int(self):
        txt = self.max_int_ed.text()
        if txt == 'auto':
            return max(self.curr_trace)
        else:
            return int(self.max_int_ed.text())

    def get_min_int(self):
        txt = self.min_int_ed.text()
        if txt == 'auto':
            return min(self.curr_trace)
        else:
            return int(self.min_int_ed.text())

    def set_max_int(self,val):
        self.max_int_ed.setText(str(val))

    def set_min_int(self,val):
        self.min_int_ed.setText(str(val))


