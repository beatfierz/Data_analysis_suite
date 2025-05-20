from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class TracePlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fig = Figure(figsize=(2, 2))  # horizontal shape
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.tick_params(axis='both', labelsize=8)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_trace(self, y_values, fitted=None):
        self.ax.clear()
        self.ax.plot(y_values, linewidth=0.5)
        if fitted is not None:
            self.ax.plot(fitted, linewidth=0.75, color='red')
        self.ax.set_xlabel("Time", fontsize=8)
        self.ax.set_ylabel("Intensity", fontsize=8)
        self.fig.tight_layout()
        self.canvas.draw()
