import numpy as np

def get_data_from_stack(tiff_stack, x_center, y_center, radius):
    """
    Extracts an intensity trace by integrating pixel values in a circular
    region around (x_center, y_center) across all frames in the TIFF stack.

    Parameters:
        tiff_stack : 3D numpy array (frames, height, width)
        x_center   : float - X coordinate of peak center
        y_center   : float - Y coordinate of peak center
        radius     : int - integration radius in pixels

    Returns:
        intensity_trace : list of integrated intensity values per frame
    """
    num_frames, height, width = tiff_stack.shape
    Y, X = np.ogrid[:height, :width]
    mask = (X - x_center) ** 2 + (Y - y_center) ** 2 <= radius ** 2

    intensity_trace = [np.sum(tiff_stack[i][mask]) for i in range(num_frames)]
    return intensity_trace
