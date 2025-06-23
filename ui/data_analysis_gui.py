# ui/data_analysis_gui.py

import os
import json
import tifffile
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QCursor

from ui.panels import general_params
from utils.peak_detection import fast_peak_find
from utils.get_nearest_peak import get_nearest_peak
from utils.integrate_trace import get_data_from_stack
from utils.stepfit import stepfit
from utils.calc_xcorr_img import calcoffset
from utils.frame_skipping import process_skip_frames
from utils.background_noise import estimate_baseline_noise
from utils.trace_extraction import process_peak

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
        self.ref_img = None # stores the reference image
        self.showing_ref = False
        self.curr_frame_noise = 0
        self.curr_frame_bl = 0


        self.peak_list = []           # peaklist is stored here
        self.current_trace = {}   # stores values of current trace
        self.skip_frames = []
        self.current_trace_bl = 0
        self.current_trace_noise = 0

        self.threshold = 1000
        self.circle_radius = self.general_params.get_peak_rad() # int(settings.get("circle_size", 5))
        self.canvas.set_circle_radius(self.circle_radius)
        self.canvas.trace_callback = self.handle_mouse_click

        # --- initiate canvas and load placeholder image ---
        #get startup img
        startup_img = self.settings.get("startup_img", ".")
        file_path = Path(__file__).parent / startup_img
        init_img = tifffile.imread(file_path)
        max_val = init_img.max()
        self.intensity_controls.set_intensity_value(max_val)
        self.canvas.set_intensity_range(max_val)
        init_img = init_img[np.newaxis, ...]
        self.canvas.set_stack(init_img)

    def load_ref(self):
        print("loading reference tiff")
        ref_img, path = self.load_tiff_file(expect_2d=True)
        if ref_img is None:
            return

        self.file_controls.set_ref_path_label(path)
        max_val = ref_img.max()
        self.intensity_controls.set_intensity_value(max_val)
        self.frame_controls.slider.setEnabled(False)
        ref_img = ref_img[np.newaxis, ...]
        self.ref_img = ref_img
        self.canvas.set_stack(self.ref_img)
        self.showing_ref = True


    def load_tiff(self):
        tiff_img, path = self.load_tiff_file(expect_2d=False)
        if tiff_img is None:
            return

        self.tiff_stack = tiff_img
        max_val = tiff_img.max()
        self.intensity_controls.set_intensity_range(max_val)
        self.intensity_controls.set_intensity_value(int(max_val / 5))
        self.canvas.set_intensity_range(max_val)

        if self.tiff_stack.ndim == 2:
            self.tiff_stack = self.tiff_stack[np.newaxis, ...]
            self.frame_controls.slider.setEnabled(False)
        else:
            self.frame_controls.slider.setEnabled(True)

        self.canvas.set_stack(self.tiff_stack)
        self.frame_controls.set_maximum(self.tiff_stack.shape[0] - 1)
        self.canvas.set_peaks([])
        self.file_controls.set_path_label(path)
        self.showing_ref = False

        # initalize Frameskip
        self.general_params.update_frame_skip()

        # Estimate baseline and noise
        d = self.tiff_stack[0]
        self.curr_frame_bl, self.curr_frame_noise = estimate_baseline_noise(d)
        self.peak_controls.set_max_threshold(int(d.max()))
        thrs = max(min(d.max(axis=0).min(), d.max(axis=1).min()), 1e-3)
        self.peak_controls.set_threshold(thrs)
        self.threshold = self.curr_frame_bl + self.curr_frame_noise * 5

    def load_tiff_file(self, expect_2d=False):
            start_path = self.settings.get("last_path", ".")
            path, _ = QFileDialog.getOpenFileName(None, "Open TIFF File", start_path, "TIFF files (*.tif *.tiff)")
            if not path:
                return None, None
            self.settings.set("last_path", os.path.dirname(path))

            img = tifffile.imread(path)

            if expect_2d:
                if img.ndim != 2:
                    QMessageBox.critical(None, "Invalid Image", "Please select a single 2D image (not a stack).")
                    print("Error: loaded reference image is not 2D.")
                    return None, None

            return img, path

    def toggle_data_ref(self):
        print('Toggle data vs reference image')

        if not hasattr(self, "ref_img") or self.ref_img is None:
            print("No reference image loaded.")
            return

        if not hasattr(self, "tiff_stack") or self.tiff_stack is None:
            print("No TIFF stack loaded.")
            return

        self.showing_ref = not self.showing_ref
        self.file_controls.drtoggle_button.setText("Show Data" if self.showing_ref else "Show Ref")

        if self.showing_ref:
            print("Displaying reference image")
            self.canvas.set_stack(self.ref_img)
            self.frame_controls.slider.setEnabled(False)  # ref is a single frame

        else:
            print("Displaying data stack")
            frame_idx = self.peak_controls.get_pdet_frame()
            self.canvas.set_stack(self.tiff_stack)
            self.canvas.set_frame(frame_idx)
            self.frame_controls.slider.setEnabled(self.tiff_stack.shape[0] > 1)

        # Force canvas update
        self.canvas.update()

    def update_peak_threshold(self, value):
        self.threshold = value

    def update_frame(self, index):
        self.canvas.set_frame(index)
        self.peak_controls.set_pdet_frame(index)
        # --- update noise/bg levels and peak detection threshold
        curr_frame = self.tiff_stack[index]
        self.curr_frame_bl, self.curr_frame_noise = estimate_baseline_noise(curr_frame)
        thrs = max(min(curr_frame.max(axis=0).min(), curr_frame.max(axis=1).min()), 1e-3)
        self.peak_controls.set_threshold(thrs)
        # --- update xcorr plot between first and current frame, if checked---
        if self.cross_corr.xcorr_enabled():
            offset, xpeak, ypeak, xcorrmat = calcoffset(self.tiff_stack[0], curr_frame)
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

    def mouse_select_peaks(self):
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

    def txt_select_peaks(self):
        # --- get trace number ---
        idx = self.export_controls.get_trace_no()
        self.peak_sel(idx)
        self.canvas.update()

    def extract_trace(self):
        edge_len = int(self.settings.get("edge_len", "16"))
        pk_shift = float(self.settings.get("pk_shift", "5"))
        thrs = self.export_controls.get_threshold()
        timebase = self.general_params.get_timebase()
        self.skip_frames = self.general_params.update_frame_skip()
        selected_peaks = [p for p in self.peak_list if p.get("selected", False)]

        for i, peak in enumerate(selected_peaks):
            print(f"Processing selected peak at ({peak['x']}, {peak['y']})")
            process_peak(self, peak, i, edge_len, pk_shift, thrs, timebase)

        self.canvas.update()

    def detect_peaks(self):
        if self.tiff_stack is None:
            print("No image loaded.")
            return

        # --- detect peaks from designated frame ---
        # do this from current index if data is displayed, otherwise from reference image
        if self.showing_ref:
            frame = self.ref_img
            frame_idx = 0

        else:
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
                self.peak_sel(idx)
        elif self.canvas.click_mode == "deselect":
            idx, dist = get_nearest_peak(x, y, [(p["y"], p["x"]) for p in self.peak_list])
            if idx is not None and dist < 20:
                self.peak_list[idx]["selected"] = False
                self.canvas.set_peaks(self.peak_list)

        self.canvas.click_mode = None
        self.canvas.unsetCursor()

    def peak_sel(self, idx):
        self.export_controls.set_trace_no(idx)
        self.peak_list[idx]["selected"] = True
        x = self.peak_list[idx]["x"]
        y = self.peak_list[idx]["y"]
        # --- get trace data ---
        trace = get_data_from_stack(self.tiff_stack, x, y, self.general_params.get_peak_rad())
        self.current_trace_bl, self.current_trace_noise = estimate_baseline_noise(trace)
        # --- process skipframes ---
        self.skip_frames = self.general_params.update_frame_skip()
        pr_trace = process_skip_frames(trace, self.skip_frames)
        # --- generate timebase ---
        timebase = self.general_params.get_timebase()
        timepoints = timebase * np.arange(len(pr_trace))
        if self.export_controls.stepfit_enabled():
            fit = stepfit(pr_trace, 'measnoise', 100, 'passes', 5, 'verbose', 0)
        else:
            fit = None
        self.current_trace = {
            "x": x,
            "y": y,
            "times": timepoints,
            "values": pr_trace,
            "fit": fit
        }
        if self.trace_plot:
            self.trace_plot.plot_trace(timepoints, pr_trace, fitted=fit)
        self.canvas.set_peaks(self.peak_list)