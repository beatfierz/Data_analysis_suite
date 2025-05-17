
import numpy as np
import os
import tifffile
import json
import csv
from PyQt5.QtWidgets import QWidget, QSlider, QFileDialog, QLineEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
from utils.peak_detection import fast_peak_find
from PyQt5.QtWidgets import QFileDialog
from utils.get_nearest_peak import get_nearest_peak

class TiffViewer(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.tiff_stack = None
        self.current_frame = 0
        self.min_intensity = 0
        self.max_intensity = 65280
        self.peak_list = []
        self.peak_frame_input = None
        self.peak_threshold = None
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(65535)
        self.threshold_slider.setValue(1000)
        self.threshold_slider.setEnabled(False)  # Disabled until TIFF is loaded
        self.threshold_slider.valueChanged.connect(self.update_peak_threshold)
        self.click_mode = None  # "add" or "delete"

        # --- set initial parameters ---
        # --- GUI params ---
        self.circle_radius = int(self.settings.get("circle_size", 5))

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(1)
        self.intensity_slider.setMaximum(65280)
        self.intensity_slider.setValue(1000)
        self.intensity_slider.valueChanged.connect(self.update_display)

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setSingleStep(1)
        self.frame_slider.valueChanged.connect(self.change_frame)

    def load_tiff(self):
        start_path = self.settings.get("last_path", ".")
        path, _ = QFileDialog.getOpenFileName(self, "Open TIFF File", start_path, "TIFF files (*.tif *.tiff)")
        if path:
            self.settings.set("last_path", os.path.dirname(path))
            self.tiff_stack = tifffile.imread(path)
            self.frame_slider.setMaximum(self.tiff_stack.shape[0] - 1)
            # --- set peak detection threshold ---
            d = self.tiff_stack[0]
            default_threshold = max(min(d.max(axis=0).min(), d.max(axis=1).min()), 1e-3)
            default_threshold = int(default_threshold)
            self.peak_threshold = default_threshold
            self.threshold_slider.setMaximum(int(d.max()))
            self.threshold_slider.setValue(default_threshold)
            self.threshold_slider.setEnabled(True)
            if hasattr(self, "file_path_label"):
                self.file_path_label.setText(path)
            self.update()

    def update_peak_threshold(self, value):
        self.peak_threshold = value

    def change_frame(self, value):
        self.current_frame = value
        self.update()

    def update_display(self, value):
        self.max_intensity = value
        self.update()

    def set_intensity_value(self, value):
        self.intensity_slider.setValue(value)
        self.max_intensity = value
        self.update()

    def set_frame_value(self, value):
        self.frame_slider.setValue(value)
        self.current_frame = value
        self.update()

    def detect_peaks(self):
        if self.tiff_stack is None:
            return
        try:
            idx = int(self.peak_frame_input.text())
        except:
            idx = self.current_frame
        idx = max(0, min(idx, self.tiff_stack.shape[0] - 1))
        frame = self.tiff_stack[idx]
        self.peak_list.clear()
        self.peak_list = fast_peak_find(frame, threshold=self.peak_threshold)
        print(f"Detected peaks in frame {idx}:")
        for (x, y) in self.peak_list:
            print(f"({x}, {y})")
        self.update()

    def save_peaklist(self):
        if not self.peak_list:
            print("No peaks to save.")
            return

        # Generate default filename based on TIFF filename
        tiff_path = self.file_path_label.text() if hasattr(self, "file_path_label") else None
        if tiff_path and os.path.isfile(tiff_path):
            base_name = os.path.splitext(os.path.basename(tiff_path))[0]
            suggested_name = base_name + "_lst.json"
            default_path = os.path.join(os.path.dirname(tiff_path), suggested_name)
        else:
            default_path = "peaklist_lst.json"

        path, _ = QFileDialog.getSaveFileName(self, "Save Peaklist (JSON)", default_path, "JSON files (*.json)")
        if not path:
            return

        metadata = {
            "filename": self.settings.get("last_path", "unknown"),
            "frame_count": int(self.tiff_stack.shape[0]),
            "frame_shape": list(self.tiff_stack.shape[1:]),
            "dtype": str(self.tiff_stack.dtype)
        }

        data = {
            "metadata": metadata,
            "peaklist": [[int(y), int(x)] for y, x in self.peak_list]
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved peaklist with metadata to {path}")

    def load_peaklist(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Peaklist (JSON)", "", "JSON files (*.json)")
        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        if "peaklist" not in data:
            print("Invalid peaklist file.")
            return

        self.peak_list = [tuple(p) for p in data["peaklist"]]
        meta = data.get("metadata", {})
        print(f"Loaded {len(self.peak_list)} peaks from {path}")
        if meta:
            print("Metadata:", json.dumps(meta, indent=2))

        self.update()

    def activate_add_peak_mode(self):
        self.click_mode = "add"
        self.setCursor(QCursor(Qt.CrossCursor))

    def activate_delete_peak_mode(self):
        self.click_mode = "delete"
        self.setCursor(QCursor(Qt.CrossCursor))

    def clear_all_peaks(self):
        self.peak_list.clear()
        self.update()

    def select_peaks(self):
        self.update()

    def deselect_peaks(self):
        self.update()

    def select_all_peaks(self):
        self.update()

    def clcAll_peaks(self):
        self.update()

    def extract_trace(self):
        self.update()

    def mousePressEvent(self, event):
        if self.tiff_stack is None or self.click_mode is None:
            return

        # Convert from widget coords to image coords
        frame = self.tiff_stack[self.current_frame]
        height, width = frame.shape
        scaled_pixmap_width = self.width()
        scaled_pixmap_height = self.height()
        # --- determine aspect ratios ---
        aspect_image = height / width
        aspect_canvas = scaled_pixmap_height / scaled_pixmap_width
        # --- compare aspect ratios ---
        if aspect_image >= aspect_canvas:
            # height canvas = height image
            scale = height / scaled_pixmap_height
        elif aspect_image < aspect_canvas:
            scale = width / scaled_pixmap_width

        # --- calculate offset ---
        offs_x = scaled_pixmap_width / 2 - width / scale / 2
        offs_y = scaled_pixmap_height / 2 - height / scale / 2

        click_x = int(event.pos().x() - offs_x)*scale
        click_y = int(event.pos().y() - offs_y)*scale
        print(f"Canvas width/height ({scaled_pixmap_width}, {scaled_pixmap_height})")
        print(f"Width/height ({width}, {height})")
        print(f"Unscaled click pos ({int(event.pos().x())}, {int(event.pos().y())})")
        print(f"Offsets X,Y ({offs_x}, {offs_y})")
        print(f"Scaling factor ({scale})")

        if self.click_mode == "add":
            self.peak_list.append((click_y, click_x))
            print(f"Added peak at ({click_x}, {click_y})")
        elif self.click_mode == "delete":
            idx, dist = get_nearest_peak(click_x, click_y, self.peak_list)
            if idx is not None and dist < 20:
                removed = self.peak_list.pop(idx)
                print(f"Deleted peak at {removed}")
        self.click_mode = None
        self.unsetCursor()
        self.update()

    def paintEvent(self, event):
        if self.tiff_stack is None:
            return
        frame = self.tiff_stack[self.current_frame]
        range_val = self.max_intensity - self.min_intensity
        if range_val == 0:
            range_val = 1
        norm = np.clip((frame - self.min_intensity) / range_val * 255, 0, 255).astype(np.uint8)
        height, width = norm.shape
        image = QImage(norm.data, width, height, norm.strides[0], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        painter = QPainter(self)
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        y_offset = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)

        # Overlay peaks
        if self.peak_list:
            scale_x = scaled_pixmap.width() / width
            scale_y = scaled_pixmap.height() / height
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            radius = self.circle_radius
            for py, px in self.peak_list:
                cx = int(px * scale_x) + x_offset
                cy = int(py * scale_y) + y_offset
                painter.drawEllipse(cx - radius, cy - radius, 2 * radius, 2 * radius)