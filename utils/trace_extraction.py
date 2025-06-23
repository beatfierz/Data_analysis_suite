import os
import numpy as np
from scipy.optimize import curve_fit
from numba import njit
import json
from utils.integrate_trace import get_data_from_stack
from utils.frame_skipping import process_skip_frames
from utils.background_noise import estimate_baseline_noise


def extract_subimg(img, coord, length, tim_start, tim_end, frame_skip=None):
    """
    Extracts a centered subimage from a 3D image stack and averages it over selected time frames.

    Parameters:
    - img: 3D numpy array of shape (x, y, time)
    - coord: tuple or list (x, y) center of the subimage
    - length: side length of the square subimage
    - tim_start: starting frame index (inclusive)
    - tim_end: ending frame index (inclusive)
    - frame_skip: list or array of frame indices to skip (optional)

    Returns:
    - subimg: 2D numpy array, averaged subimage
    """
    _, height_img, width_img = img.shape

    # attention: the first value in coord (assumed to be 'x') corresponds to the height value in the tiff
    # the second value in coord (assumed as 'y'), corresponds to the width value in the tiff

    height0 = round(coord[1])
    width0 = round(coord[0])
    half_len = round(length / 2)

    # Ensure boundaries do not exceed image dimensions
    height_start = max(0, height0 - half_len)
    heigth_end = min(height_img, height0 + half_len)
    width_start = max(0, width0 - half_len)
    widht_end = min(width_img, width0 + half_len)

    # Adjust to ensure subimage size
    if (heigth_end - height_start + 1) < length + 1:
        if height_start == 0:
            heigth_end = min(height_start + length, height_img - 1)
        else:
            height_start = max(heigth_end - length, 0)
    if (widht_end - width_start + 1) < length + 1:
        if width_start == 0:
            widht_end = min(width_start + length, width_img - 1)
        else:
            width_start = max(widht_end - length, 0)

    height_range = slice(height_start, heigth_end)
    width_range = slice(width_start, widht_end)

    # Generate list of frames to include
    time_indices = list(range(tim_start, tim_end + 1))
    if frame_skip:
        time_indices = [t for t in time_indices if t not in frame_skip]

    # Preallocate and accumulate
    subimg = np.zeros((heigth_end - height_start, widht_end - width_start), dtype=float)
    frame_count = 0

    for t in time_indices:
        if 0 <= t < img.shape[0]:
            subimg += img[t, height_range, width_range].astype(float)
            frame_count += 1

    if frame_count > 0:
        subimg /= frame_count
    else:
        subimg = np.zeros_like(subimg)

    return subimg

def gaussian_2d(coords, A, x0, sigma_x, y0, B):
    x, y = coords
    return gaussian_2d_numba(x, y, A, x0, sigma_x, y0, B)

@njit
def gaussian_2d_numba(x, y, A, x0, sigma_x, y0, B):
    return A * np.exp(-(((x - x0)**2 + (y - y0)**2) / (2 * sigma_x**2))) + B

def fit_psf(subimg, edge_len, pk_shift):
    # %fits a 2D gaussian to the center peak in the supplied image (subimg) of
    # edge length edge_len. The peak cannot move further from the center as
    # specified in pk_shift (from the parameters.ini file)
    # returns popt: optimized parameters [A, x0, sigma, y0, B]
    # corresponding to Z_fit = A * exp(-((X - x0)^2 + (Y - y0)^2) / (2*sigma^2)) + B
    # returns pcov: Covariance matrix of the fit
    # p0: initial guess used for debugging
    # returns 0 if fit fails.

    cent = edge_len / 2
    x = np.arange(edge_len)
    y = np.arange(edge_len)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack((X.ravel(), Y.ravel()))

    Z = subimg.ravel()

    # Initial guess: [Amplitude, x0, sigma, y0, offset]
    A0 = subimg[int(cent), int(cent)]
    x0 = cent
    y0 = cent
    sigma0 = 1
    B0 = np.mean(subimg)
    p0 = [A0, x0, sigma0, y0, B0]

    # Bounds (like lb and ub in MATLAB)
    lower_bounds = [0, x0 - pk_shift, 0, y0 - pk_shift, 0]
    upper_bounds = [1e8, x0 + pk_shift, 100, y0 + pk_shift, 1e8]

    try:
        popt, pcov = curve_fit(gaussian_2d, coords, Z, p0=p0, bounds=(lower_bounds, upper_bounds))
        return popt, pcov, p0
    except RuntimeError as e:
        print("Fit did not converge:", e)
        return None, None, p0

def process_peak(self, peak, peak_idx, edge_len, pk_shift, thrs, timebase):
    x, y = peak["x"], peak["y"]
    trace = get_data_from_stack(self.tiff_stack, x, y, self.general_params.get_peak_rad())
    pr_trace = process_skip_frames(trace, self.skip_frames)

    timepoints = timebase * np.arange(len(pr_trace))
    self.trace_plot.plot_trace(timepoints, pr_trace)

    self.current_trace_bl, self.current_trace_noise = estimate_baseline_noise(trace)
    threshold = self.current_trace_bl + thrs

    regions = detect_threshold_regions(trace, threshold)
    x_ref, y_ref = get_reference_fit(self, x, y, edge_len, pk_shift)

    region_data = []
    for start_idx, end_idx in regions:
        region_entry = analyze_region(self, x, y, start_idx, end_idx, x_ref, y_ref, edge_len, pk_shift)
        if region_entry is not None:
            region_data.append(region_entry)

    peak_entry = {
        "peak_x": float(x),
        "peak_y": float(y),
        "ref_fit_x": float(x_ref),
        "ref_fit_y": float(y_ref),
        "timebase": float(timebase),
        "trace_baseline": float(self.current_trace_bl),
        "threshold": float(threshold),
        "trace": [float(val) for val in pr_trace],
        "regions": region_data
    }

    tiff_path = self.file_controls.get_path_label()
    if tiff_path and os.path.isfile(tiff_path):
        base_name = os.path.splitext(os.path.basename(tiff_path))[0]
        f_name = base_name + f"_tracen_{peak_idx}.json"
        save_path = os.path.join(os.path.dirname(tiff_path), f_name)
    else:
        save_path = f"tracen_{peak_idx}.json"

    with open(save_path, "w") as f:
        json.dump(peak_entry, f, indent=2)

    print(f"  → Saved to {save_path}")

def detect_threshold_regions(trace, threshold):
    above = trace > threshold
    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if above[0]:
        starts = np.insert(starts, 0, 0)
    if above[-1]:
        ends = np.append(ends, len(trace))

    return list(zip(starts, ends))

def get_reference_fit(self, x, y, edge_len, pk_shift):
    try:
        if self.ref_img is not None:
            ref_source = self.ref_img
            frame_idx = 0
        else:
            frame_idx = self.peak_controls.get_pdet_frame()
            ref_source = self.tiff_stack[frame_idx]

        ref_subimg = extract_subimg(ref_source, [x, y], edge_len, 0, 0, frame_skip=None)
        ref_fit, _, _ = fit_psf(ref_subimg, edge_len, pk_shift)
        if ref_fit is not None:
            _, x_ref, _, y_ref, _ = ref_fit
        else:
            print("Warning: Reference fit failed. Using center of subimage.")
            x_ref, y_ref = edge_len / 2, edge_len / 2
    except Exception as e:
        print(f"Failed to extract or fit reference image: {e}")
        x_ref, y_ref = edge_len / 2, edge_len / 2

    return x_ref, y_ref

def analyze_region(self, x, y, start_idx, end_idx, x_ref, y_ref, edge_len, pk_shift):
    print(f"  Region {start_idx}–{end_idx - 1}")
    try:
        subimg = extract_subimg(self.tiff_stack, [x, y], edge_len, start_idx, end_idx - 1, frame_skip=None)
        popt, pcov, p0 = fit_psf(subimg, edge_len, pk_shift)

        if popt is None:
            return {"start_idx": int(start_idx), "end_idx": int(end_idx), "fit_failed": True}

        A, x0, sigma, y0, B = popt
        distance = np.sqrt((x0 - x_ref) ** 2 + (y0 - y_ref) ** 2)

        return {
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "distance_from_center": float(distance),
            "sigma": float(sigma),
            "fit_params": [float(p) for p in popt],
            "fit_guess": [float(p) for p in p0]
        }

    except Exception as e:
        print(f"    Failed to extract or fit subimage: {e}")
        return None

