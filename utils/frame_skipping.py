import numpy as np

def parse_skip_frames(skip_str, total_frames):
    """
    Parses a skip frame string and returns an array of frame indices to skip.

    Parameters:
    - skip_str: string in the format "x1,x2,...;y1:y2" or similar
    - total_frames: total number of frames in the movie

    Returns:
    - skip_frames: sorted numpy array of unique frame indices to skip
    """
    skip_frames = set()

    if skip_str.strip().upper() == "N/A" or skip_str.strip() == "":
        return np.array([], dtype=int)

    parts = skip_str.split(";")

    try:
        for part in parts:
            part = part.strip()
            if ":" in part:
                # parse y1:y2 (interval pattern)
                y1, y2 = map(int, part.split(":"))
                if y2 == 0:
                    raise ValueError("Step size cannot be zero in y1:y2")
                skip_frames.update(range(y1, total_frames, y2))
            elif "," in part:
                # parse list of individual frame numbers
                indices = map(int, part.split(","))
                skip_frames.update(indices)
            elif part.isdigit():
                skip_frames.add(int(part))
            else:
                raise ValueError(f"Unrecognized skip pattern: '{part}'")
    except Exception as e:
        print(f"[Error] Failed to parse skip string '{skip_str}': {e}")
        return np.array([], dtype=int)

    return np.array(sorted(i for i in skip_frames if 0 <= i < total_frames), dtype=int)

def process_skip_frames(trace, skip_frames):
    """
    Overwrites values in `trace` at positions in `skip_frames` by averaging neighbors.

    Parameters:
    - trace: 1D array-like of numeric values
    - skip_frames: array-like of frame indices to be smoothed

    Returns:
    - new_trace: a copy of the trace with specified frames replaced
    """
    trace = np.asarray(trace, dtype=float)  # ensure it's a float array
    new_trace = trace.copy()
    length = len(trace)

    for i in skip_frames:
        if i <= 0:
            new_trace[i] = trace[i + 1]
        elif i >= length - 1:
            new_trace[i] = trace[i - 1]
        else:
            new_trace[i] = 0.5 * (trace[i - 1] + trace[i + 1])

    return new_trace
