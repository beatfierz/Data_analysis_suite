import numpy as np
from scipy.ndimage import gaussian_filter1d


def estimate_baseline_noise(data, bins=256, smooth_sigma=1.5):
    """
    Estimate baseline (background) and noise from a 1D or 2D array.
    Uses histogram peak detection and width estimation.

    Parameters:
    - data: np.ndarray (1D trace or 2D image)
    - bins: int, number of histogram bins
    - smooth_sigma: float, smoothing width for histogram

    Returns:
    - baseline: float, estimated background level
    - noise: float, estimated noise (std dev)
    """
    data = np.asarray(data)
    flat = data.ravel()

    # Build histogram
    hist, bin_edges = np.histogram(flat, bins=bins, range=(np.min(flat), np.max(flat)))

    # Smooth histogram to suppress noise
    hist_smooth = gaussian_filter1d(hist, sigma=smooth_sigma)

    # Find peak in histogram (mode-like estimate of background)
    peak_idx = np.argmax(hist_smooth)
    baseline = 0.5 * (bin_edges[peak_idx] + bin_edges[peak_idx + 1])

    # Estimate width around the peak to get noise (FWHM-like)
    half_max = hist_smooth[peak_idx] / 2
    indices = np.where(hist_smooth >= half_max)[0]

    if len(indices) >= 2:
        left, right = indices[0], indices[-1]
        fwhm = bin_edges[right] - bin_edges[left]
        noise = fwhm / 2.355  # convert FWHM to std dev assuming Gaussian
    else:
        noise = np.std(flat)  # fallback if histogram is degenerate

    return baseline, noise