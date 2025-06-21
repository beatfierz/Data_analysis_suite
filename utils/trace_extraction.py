import numpy as np
from scipy.optimize import curve_fit
from numba import njit


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
    sigma0 = 10
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


