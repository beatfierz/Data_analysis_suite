# ui/widgets/tiff_canvas.py

from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPixmap, QImage, QPen, QColor, QCursor
from PyQt5.QtCore import Qt
import numpy as np

class TiffCanvas(QWidget):
    # Handles:
    # Displaying current frame of TIFF stack
    # Drawing overlays (e.g., peaks, circles)
    # Mouse events (if any interaction is required)
    # Takes:
    # Image data (numpy array)
    # Overlay data (peak positions etc.)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_stack = None
        self.current_frame = 0
        self.max_intensity = 65280
        self.min_intensity = 0
        self.peaks = []
        self.circle_radius = 5
        self.click_mode = None
        self.trace_callback = None  # assigned externally

    def set_stack(self, image_stack):
        self.image_stack = image_stack
        self.current_frame = 0
        self.update()

    def set_frame(self, index):
        self.current_frame = index
        self.update()

    def get_current_frame(self):
        return self.current_frame

    def set_peaks(self, peaks):
        self.peaks = peaks
        self.update()

    def set_intensity_range(self, min_val, max_val):
        self.min_intensity = min_val
        self.max_intensity = max_val
        self.update()

    def set_circle_radius(self, r):
        self.circle_radius = r

    def paintEvent(self, event):
        if self.image_stack is None:
            return
        frame = self.image_stack[self.current_frame]
        norm = np.clip((frame - self.min_intensity) / (self.max_intensity - self.min_intensity) * 255, 0, 255).astype(np.uint8)
        h, w = norm.shape
        img = QImage(norm.data, w, h, norm.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(img).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter = QPainter(self)
        x_off = (self.width() - pixmap.width()) // 2
        y_off = (self.height() - pixmap.height()) // 2
        painter.drawPixmap(x_off, y_off, pixmap)

        if self.peaks:
            scale_x = pixmap.width() / w
            scale_y = pixmap.height() / h
            for peak in self.peaks:
                color = QColor(0, 100, 255) if peak.get("selected", False) else QColor(255, 0, 0)
                painter.setPen(QPen(color, 2))
                cx = int(peak["x"] * scale_x) + x_off
                cy = int(peak["y"] * scale_y) + y_off
                r = self.circle_radius
                painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)

    def mousePressEvent(self, event):
        if self.image_stack is None or self.click_mode is None:
            return
        h, w = self.image_stack[self.current_frame].shape
        pixmap_width = self.width()
        pixmap_height = self.height()
        aspect_image = h / w
        aspect_canvas = pixmap_height / pixmap_width
        scale = h / pixmap_height if aspect_image >= aspect_canvas else w / pixmap_width
        offset_x = pixmap_width / 2 - w / scale / 2
        offset_y = pixmap_height / 2 - h / scale / 2
        click_x = (event.pos().x() - offset_x) * scale
        click_y = (event.pos().y() - offset_y) * scale

        if self.trace_callback:
            self.trace_callback(click_x, click_y)
        self.update()
