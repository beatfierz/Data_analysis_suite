
import numpy as np
from scipy.ndimage import median_filter, gaussian_filter

def fast_peak_find(d, threshold=None, filt_size=7, sigma=1.0, edg=3):
    if d.ndim > 2:
        d = d[0, :, :]
    d = d.astype(np.float32)
    if threshold is None:
        threshold = max(min(d.max(axis=0).min(), d.max(axis=1).min()), 1e-3)

    d = median_filter(d, size=(2, 2))
    d[d < threshold] = 0
    if d.max() == 0:
        return []

    d = gaussian_filter(d, sigma=sigma)
    d[d < 0.9 * threshold] = 0
    h, w = d.shape
    x_list, y_list = np.where(d[edg:h-edg, edg:w-edg] > 0)
    x_list += edg
    y_list += edg

    peaks = []
    for x, y in zip(x_list, y_list):
        val = d[x, y]
        nb = d[x-1:x+2, y-1:y+2].flatten()
        if val > nb[[1, 3, 5, 7]].max() and val >= nb[[0, 2, 6, 8]].max():
            peaks.append((x, y))
    return peaks
