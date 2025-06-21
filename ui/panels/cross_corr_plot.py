import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt

class CrossCorrDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # --- panel for x-crosscorr ---
        self.xcorr_chk = QCheckBox("Enable xcorr calculation")
        self.xcorr_chk.setChecked(False)
        self.xfig = Figure(figsize=(1,1))
        self.canvas = FigureCanvas(self.xfig)
        self.canvas.setMinimumSize(300,300)
        self.ax_x = self.xfig.add_subplot(211)
        self.ax_y = self.xfig.add_subplot(212)
        self.ax_x.tick_params(axis='both', labelsize=8)
        self.ax_y.tick_params(axis='both', labelsize=8)

        layout = QVBoxLayout()
        layout.addWidget(QLabel('---Cross correlation analysis---'))
        layout.addWidget(self.xcorr_chk)
        layout.addWidget(self.canvas)
        layout.addStretch()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

    def plot_trace(self, offset, xpeak, ypeak, xcorrmat):
        h, w = xcorrmat.shape
        scal = 0.4
        xlim_range = np.round(np.array([scal, 1 - scal]) * h).astype(int)
        ylim_range = np.round(np.array([scal, 1 - scal]) * w).astype(int)
        x_slice = xcorrmat[xlim_range[0]:xlim_range[1], xpeak]
        y_slice = xcorrmat[ypeak, ylim_range[0]:ylim_range[1]]
        self.plot_single_axis(self.ax_x, x_slice)
        self.plot_single_axis(self.ax_y, y_slice)

    def plot_single_axis(self, axis, y_values):
        axis.clear()
        axis.plot(y_values, linewidth=0.5)
        axis.set_xlabel("Time", fontsize=8)
        axis.set_ylabel("Intensity", fontsize=8)
        self.xfig.tight_layout()
        self.canvas.draw()

    def xcorr_enabled(self):
        return self.xcorr_chk.isChecked()