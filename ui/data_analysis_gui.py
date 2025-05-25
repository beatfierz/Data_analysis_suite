# ui/data_analysis_gui.py

import os
import json
import tifffile
import numpy as np
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor
from utils.peak_detection import fast_peak_find
from utils.get_nearest_peak import get_nearest_peak
from utils.integrate_trace import get_data_from_stack
from utils.stepfit import stepfit
from utils.calc_xcorr_img import calcoffset
from pathlib import Path

class DataAnalysisGUI:
    def __init__(self, canvas, trace_plot, settings, file_controls,
                 intensity_controls, frame_controls, peak_controls,
                 export_controls, general_params, cross_corr):
        self.canvas = canvas
        self.trace_plot = trace_plot
        self.file_controls = file_controls
        self.frame_controls = frame_controls
        self.intensity_controls = intensity_controls
        self.peak_controls = peak_controls
        self.export_controls = export_controls
        self.general_params = general_params
        self.cross_corr = cross_corr
        self.settings = settings

        self.tiff_stack = None   # stores loaded Tiff images
        self.peak_list = []           # peaklist is stored here
        self.current_trace = {}   # stores values of current trace

        self.threshold = 1000
        self.circle_radius = int(settings.get("circle_size", 5))
        self.canvas.set_circle_radius(self.circle_radius)
        self.canvas.trace_callback = self.handle_mouse_click

        # --- initiate canvas and load placeholder image ---
        file_path = Path(__file__).parent / "startup_displ.tif"
        init_img = tifffile.imread(file_path)
        max_val = init_img.max()
        self.intensity_controls.set_intensity_value(max_val)
        self.canvas.set_intensity_range(max_val)
        init_img = init_img[np.newaxis, ...]
        self.canvas.set_stack(init_img)

    def load_tiff(self):
        start_path = self.settings.get("last_path", ".")
        path, _ = QFileDialog.getOpenFileName(None, "Open TIFF File", start_path, "TIFF files (*.tif *.tiff)")
        if not path:
            return
        self.settings.set("last_path", os.path.dirname(path))
        tiff_img = tifffile.imread(path)
        max_val = tiff_img.max()
        self.tiff_stack = tiff_img
        self.intensity_controls.set_intensity_value(max_val)
        self.canvas.set_intensity_range(max_val)

        # --- check if this is a single image, if so restructure and switch off frame controls
        if self.tiff_stack.ndim == 2:
            self.tiff_stack = self.tiff_stack[np.newaxis, ...]  # Convert to 3D stack with 1 frame
            self.frame_controls.slider.setEnabled(False)
        else:
            self.frame_controls.slider.setEnabled(True)

        self.canvas.set_stack(self.tiff_stack)
        self.frame_controls.set_maximum(self.tiff_stack.shape[0] - 1)
        self.threshold = int(max(1, self.tiff_stack[0].max() // 10))
        self.canvas.set_peaks([])
        self.file_controls.set_path_label(path)

        # --- set peak detection threshold ---
        d = self.tiff_stack[0]
        self.peak_controls.set_max_threshold(int(d.max()))
        self.peak_controls.set_threshold(max(min(d.max(axis=0).min(), d.max(axis=1).min()), 1e-3))

    def update_peak_threshold(self, value):
        self.threshold = value

    def update_frame(self, index):
        self.canvas.set_frame(index)
        self.peak_controls.set_pdet_frame(index)
        #--- update xcorr plot between first and current frame---
        offset, xpeak, ypeak, xcorrmat = calcoffset(self.tiff_stack[0], self.tiff_stack[index])
        self.cross_corr.plot_trace(offset, xpeak, ypeak, xcorrmat)

    def update_intensity(self, max_val):
        self.canvas.set_intensity_range(max_val)

    def activate_add_peak_mode(self):
        self.canvas.click_mode = "add"
        self.canvas.setCursor(QCursor(Qt.CrossCursor))

    def activate_delete_peak_mode(self):
        self.canvas.click_mode = "delete"
        self.canvas.setCursor(QCursor(Qt.CrossCursor))

    def clear_all_peaks(self):
        self.peak_list.clear()
        self.canvas.update()

    def select_peaks(self):
        if not self.peak_list:
            print("No peaks to select.")
            return

        self.canvas.setCursor(QCursor(Qt.CrossCursor))
        self.canvas.click_mode = "select"

    def deselect_peaks(self):
        if not self.peak_list:
            print("No peaks to deselect.")
            return

        self.canvas.setCursor(QCursor(Qt.CrossCursor))
        self.canvas.click_mode = "deselect"

    def select_all_peaks(self):
        for peak in self.peak_list:
            peak["selected"] = True
        self.canvas.update()

    def clcAll_peaks(self):
        for peak in self.peak_list:
            peak["selected"] = False
        self.canvas.update()

    def extract_trace(self):
        self.canvas.update()

    def detect_peaks(self):
        if self.tiff_stack is None:
            print("No image loaded.")
            return

        # --- detect peaks from designated frame ---
        frame_idx = self.peak_controls.get_pdet_frame()
        frame = self.tiff_stack[frame_idx]

        self.threshold = self.peak_controls.get_threshold() # just in case, update threshold value
        raw_peaks = fast_peak_find(frame, threshold=self.threshold)
        self.peak_list = [{"x": int(x), "y": int(y), "selected": False} for y, x in raw_peaks]
        self.canvas.set_peaks(self.peak_list)
        print(f"Detected {len(self.peak_list)} peaks in frame {frame_idx}.")

    def save_peaklist(self):
        if not self.peak_list:
            print("No peaks to save.")
            return

        # Generate default filename based on TIFF filename
        tiff_path = self.file_controls.get_path_label()
        if tiff_path and os.path.isfile(tiff_path):
            base_name = os.path.splitext(os.path.basename(tiff_path))[0]
            suggested_name = base_name + "_lst.json"
            default_path = os.path.join(os.path.dirname(tiff_path), suggested_name)
        else:
            default_path = "peaklist_lst.json"

        path, _ = QFileDialog.getSaveFileName(self.file_controls, "Save Peaklist (JSON)", default_path, "JSON files (*.json)")
        if not path:
            return

        metadata = {
            "frame_count": int(self.tiff_stack.shape[0]),
            "frame_shape": list(self.tiff_stack.shape[1:]),
            "dtype": str(self.tiff_stack.dtype)
        }

        data = {
            "metadata": metadata,
            "peaklist": self.peak_list
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved peaklist to {path}")

    def load_peaklist(self):
        start_path = self.settings.get("last_path", ".")
        path, _ = QFileDialog.getOpenFileName(None, "Load Peaklist", start_path, "JSON files (*.json)")
        if not path:
            return
        with open(path, "r") as f:
            data = json.load(f)
        self.peak_list = data.get("peaklist", [])
        self.canvas.set_peaks(self.peak_list)
        print(f"Loaded {len(self.peak_list)} peaks.")

    def clear_all_peaks(self):
        self.peak_list = []
        self.canvas.set_peaks([])

    def handle_mouse_click(self, x, y):
        if self.tiff_stack is None or not self.canvas.click_mode:
            return

        if self.canvas.click_mode == "add":
            self.peak_list.append({"x": x, "y": y, "selected": False})
            self.canvas.set_peaks(self.peak_list)
        elif self.canvas.click_mode == "delete":
            idx, dist = get_nearest_peak(x, y, [(p["y"], p["x"]) for p in self.peak_list])
            if idx is not None and dist < 20:
                self.peak_list.pop(idx)
                self.canvas.set_peaks(self.peak_list)
        elif self.canvas.click_mode == "select":
            idx, dist = get_nearest_peak(x, y, [(p["y"], p["x"]) for p in self.peak_list])
            if idx is not None and dist < 20:
                self.peak_list[idx]["selected"] = True
                trace = get_data_from_stack(self.tiff_stack, x, y, radius=2)
                if self.export_controls.stepfit_enabled():
                    fit = stepfit(trace, 'measnoise', 100, 'passes', 5, 'verbose', 0)
                else:
                    fit = None
                self.current_trace = {
                    "x": x,
                    "y": y,
                    "values": trace,
                    "fit": fit
                }
                if self.trace_plot:
                    self.trace_plot.plot_trace(trace, fitted=fit)
                self.canvas.set_peaks(self.peak_list)
        elif self.canvas.click_mode == "deselect":
            idx, dist = get_nearest_peak(x, y, [(p["y"], p["x"]) for p in self.peak_list])
            if idx is not None and dist < 20:
                self.peak_list[idx]["selected"] = False
                self.canvas.set_peaks(self.peak_list)

        self.canvas.click_mode = None
        self.canvas.unsetCursor()
