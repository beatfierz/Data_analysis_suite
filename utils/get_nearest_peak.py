
import numpy as np

def get_nearest_peak(click_x, click_y, peak_list):
    if not peak_list:
        return None, float('inf')
    distances = [(i, np.hypot(px - click_x, py - click_y))
                 for i, (py, px) in enumerate(peak_list)]
    nearest_index, min_dist = min(distances, key=lambda x: x[1])
    return nearest_index, min_dist
