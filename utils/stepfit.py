import numpy as np
from scipy.signal import freqz
from scipy.signal import convolve, lfilter, butter
from scipy.signal.windows import gaussian
from scipy.ndimage import minimum_filter1d, maximum_filter1d
# from scipy.interpolate import CubicSpline
from scipy.ndimage import uniform_filter1d
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from numba import njit

# Define a context object to carry "formerly global" parameters
class StepfitContext:
    def __init__(self):
        self.userargs = None
        self.Fs = 1
        self.noise = None
        self.outputnoise = None
        self.measnoise = 0
        self.resolution = 0.1
        self.pass_idx = 0
        self.passes = 10
        self.verbose = 0
        self.b = np.array([1])
        self.a = np.array([1, 0])
        self.envelopeflag = 0
        self.y = None
        self.endbounding = 0
        self.topthickness = None
        self.bottomthickness = None

# parallel wrapper
def run_stepfit_parallel(data, use_processes=False, *args):
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    with Executor() as executor:
        return stepfit(data, *args, executor=executor)

# Main function
def stepfit(data, *args):#, executor=None):
    ctx = StepfitContext()  # Localized context instead of global vars

    # Parse name-value pairs
    arg_iter = iter(args)
    for key in arg_iter:
        try:
            val = next(arg_iter)
        except StopIteration:
            print(f"Warning: No value provided for key '{key}'")
            break
        key_lower = key.lower()
        if key_lower == 'fs':
            ctx.Fs = val
        elif key_lower == 'outputnoise':
            ctx.outputnoise = val
        elif key_lower == 'measnoise':
            ctx.measnoise = val
        elif key_lower == 'passes':
            ctx.passes = val
        elif key_lower == 'verbose':
            ctx.verbose = val
        else:
            print(f"Warning: Unknown parameter: {key}")
        if ctx.verbose > 0:
            print(f"{key_lower} = {val}")

    # Ensure row vector
    y = np.asarray(data).flatten()
    ctx.y = y
    N = len(y)

    # Default output noise if not provided
    if ctx.outputnoise is None:
        ctx.outputnoise = noise_std(y)

    # Transfer function coefficients (fixed)
    ctx.b = np.array([1])
    ctx.a = np.array([1, 0])

    # Display model coefficients
    if ctx.verbose > 0:
        print("Model coefficients:")
        print(f"  b = {ctx.b}")
        print(f"  a = {ctx.a}")

    # Estimate filtered noise amplitude
    w, h = freqz(ctx.b, ctx.a, worN=1000)
    noiseamp = np.sqrt(np.mean(np.abs(h) ** 2))
    ctx.noise = max(0, (ctx.outputnoise - ctx.measnoise) / noiseamp)

    # Initial estimate
    est = y.copy()

    for pass_idx in range(ctx.passes):
        ctx.pass_idx = pass_idx + 1
        est = vitpass(y, est, ctx) #, executor=executor)
        if ctx.verbose > 0:
            numsteps = np.sum(np.diff(est) != 0)
            print(f"Pass {ctx.pass_idx}: {numsteps} steps detected")

    return est

def vitpass(y, estin, ctx): #, executor=None):
    N = len(y)

    # Filter impulse response and window size
    imp = np.zeros(1000)
    imp[0] = 1
    impf = lfilter(ctx.b, ctx.a, imp)
    bw = np.where(impf > 0.2 * np.max(impf))[0]
    window = max(10, min(N, bw[-1] if bw.size > 0 else 0))

    # Re-estimate effective noise
    if ctx.pass_idx >= 1:
        _, h = freqz(ctx.b, ctx.a, worN=10000)
        noiseamp = np.sqrt(np.mean(np.abs(h) ** 2))
        ctx.noise = (ctx.outputnoise - ctx.measnoise) / noiseamp

    # Mean dwell time and block size
    steps = np.diff(estin)
    ind = np.where(steps != 0)[0]
    if ctx.pass_idx == 1:
        meandwell = 500
    else:
        meandwell = max(500, int(np.round(np.mean(np.diff(ind)))) if len(ind) > 1 else 500)
    blocksize = max(10 * meandwell, int(np.ceil(N / np.ceil(N / 10 / meandwell))))
    r = int(np.floor((N - 1) / blocksize))
    overlap = max(int(0.1 * blocksize), 2 * meandwell)

    # Initialize output
    est = np.zeros(N)

    # Optional noise re-estimation
    if ctx.pass_idx > 1 and (not ctx.userargs or len(ctx.userargs) < 3 or ctx.userargs[2] is None):
        residual = y - lfilter(ctx.b, ctx.a, estin)
        ctx.outputnoise = noise_std(residual)

    # Compute envelope bounds
    miny = None
    maxy = None
    if ctx.pass_idx == 1:
        low, high, maxstep, minstep, miny, maxy = findenvelope2(estin, window)
        ctx.lowsave = low
        ctx.highsave = high
        ctx.resolution = np.mean(high - low) / 50
    else:
        low, high, maxstep, minstep, miny, maxy, ctx.resolution = findenvelope3(
            estin, window, ctx.resolution, ctx.lowsave, ctx.highsave, ctx)
        ctx.lowsave = low
        ctx.highsave = high

    # Normalize bounds
    low = np.round((low - miny) / ctx.resolution).astype(int) + 1
    high = np.round((high - miny) / ctx.resolution).astype(int) + 1

    if ctx.verbose:
        print(f"Pass: {ctx.pass_idx}, Resolution = {ctx.resolution:.4f}")

    # Compute transition weight
    if ctx.pass_idx <= 1:
        range_vals = np.arange(minstep - 1, maxstep + ctx.resolution, ctx.resolution)
        zeroindex = np.argmin(np.abs(range_vals))
        range_vals -= range_vals[zeroindex]
        ctx.wt = {
            'zeroindex': zeroindex,
            'pdf': np.abs(np.sign(range_vals))
        }
    else:
        ctx.wt = computeweight(estin, minstep, maxstep, ctx)

    # Define block boundaries
    start = np.zeros(r + 1, dtype=int)
    stop = np.zeros(r + 1, dtype=int)
    eststart = np.zeros(r + 1, dtype=int)
    eststop = np.zeros(r + 1, dtype=int)
    yme, lowtmp, hightmp = [], [], []

    for j in range(r + 1):
        i = j
        start[j] = max(blocksize * i - overlap, 0)
        stop[j] = min(blocksize * (i + 1) + overlap, N)
        eststart[j] = blocksize * i
        eststop[j] = min(N, blocksize * (i + 1))
        yme.append(y[start[j]:stop[j]])
        lowtmp.append(low[start[j]:stop[j]])
        hightmp.append(high[start[j]:stop[j]])

    # Run Viterbi decoding
    param = {
        'noise': ctx.noise,
        'measnoise': ctx.measnoise,
        'outputnoise': ctx.outputnoise,
        'wt': ctx.wt,
        'resolution': ctx.resolution,
        'b': ctx.b,
        'a': ctx.a,
        'pass': ctx.pass_idx
    }

    # no parallelization
    myest = []
    for j in range(r + 1):
        est_block, _ = viterbistepdetector(yme[j], miny, lowtmp[j], hightmp[j], param)
        myest.append(est_block)

    # Run Viterbi decoding in parallel
    # block_args = [
    #    (yme[j], miny, lowtmp[j], hightmp[j], param) for j in range(r + 1)
    #]

    #if executor is not None:
    #    results = list(executor.map(run_block, block_args))
    # else:
    #    results = [run_block(args) for args in block_args]

    # myest = [res[0] for res in results]

    # Merge overlapping blocks
    for j in range(r + 1):
        idx1 = eststart[j]
        idx2 = eststop[j]
        rel_start = idx1 - start[j]
        rel_stop = idx2 - start[j]
        est[idx1:idx2] = myest[j][rel_start:rel_stop]

    return est

def run_block(args):
    return viterbistepdetector(*args)

def computeweight(est, minstep, maxstep, ctx):
    """
    Compute transition weights for step size changes.
    Translated from MATLAB computeweight() with full logic preservation.
    """
    resolution = ctx.resolution
    noise = ctx.noise
    pass_idx = ctx.pass_idx
    passes = ctx.passes

    # Smoothing factor based on noise and current pass
    redfactor = 1 - 0.8 * pass_idx / passes
    smooth = max(3, round(redfactor * noise / resolution))

    # Step size differences
    steps = np.diff(est)

    # Bin centers: extend range to accommodate Gaussian filter width
    s = np.arange(minstep - 2 * smooth * resolution,
                  maxstep + 2 * smooth * resolution + resolution,
                  resolution)

    # Index of bin closest to zero (to center)
    zindex = np.argmin(np.abs(s))
    s = s - s[zindex]  # Shift to center bins at zero

    # Histogram of step sizes using bin edges
    bin_edges = s - resolution / 2
    bin_edges = np.append(bin_edges, bin_edges[-1] + resolution)
    N, _ = np.histogram(steps, bins=bin_edges)

    # Apply Gaussian smoothing
    if smooth > 0:
        t = N[zindex]  # preserve center value
        Ntmp = N.copy()

        # Zero out center and neighbors
        if zindex > 0:
            Ntmp[zindex] = 0
        if zindex + 1 < len(Ntmp):
            Ntmp[zindex + 1] = 0
        if zindex - 1 >= 0:
            Ntmp[zindex - 1] = 0

        d = smooth
        width = 2 * d + 1
        H = gaussian(M=width, std=1/d)  # Gaussian FIR filter (like gaussfir)
        H /= np.sum(H)  # Normalize

        # Pad and filter right side
        right = np.pad(Ntmp[zindex + 1:], (0, width), mode='constant')
        right_filt = convolve(right, H, mode='full')
        right_filt = right_filt[d:len(right_filt) - d]
        Ntmpright = right_filt[:len(Ntmp) - zindex - 1]

        # Pad and filter left side
        left = np.pad(Ntmp[:zindex], (0, width), mode='constant')
        left_filt = convolve(left, H, mode='full')
        left_filt = left_filt[d:len(left_filt) - d]
        Ntmpleft = left_filt[:zindex]

        # Combine smoothed histogram
        Ntmp = np.concatenate([Ntmpleft, [0], Ntmpright])
        Ntmp[zindex] = t * np.max(H)  # Restore peak at center
        N = Ntmp + 1e-50  # Avoid log(0)

    # Suppress neighbors of zero step
    if zindex + 1 < len(N):
        N[zindex + 1] = 0
    if zindex - 1 >= 0:
        N[zindex - 1] = 0

    # Normalize and convert to log-likelihood
    # Final fix before log
    N[N == 0] = 1e-50  # prevent log(0)
    N = N / np.sum(N)
    N = -np.log(N)

    wt = {
        'zeroindex': zindex,
        'pdf': N
    }

    return wt

def viterbistepdetector(y, miny, low, high, param):
    # Unpack parameters from param dictionary
    noise = param['noise']
    measnoise = param['measnoise']
    wt = param['wt']
    resolution = param['resolution']
    b = param['b']
    a = param['a']
    pass_idx = param['pass']

    T = len(y)

    # Compute variance scale
    if pass_idx > 1:
        sigmabar = 2 * (np.sum(b ** 2) * noise ** 2 + np.sum(a ** 2) * measnoise ** 2)
    else:
        sigmabar = 9 * (noise ** 2 + measnoise ** 2 + resolution ** 2)

    maxstep = np.max(high[1:] - low[:-1])
    minstep = np.min(low[1:] - high[:-1])
    maxband = np.max(high - low + 1)

    # Initialize S and cost
    band_range = np.arange(low[0], low[0] + maxband)
    S = np.tile(band_range[:, np.newaxis], (1, T))
    cost = ((S[:, 0] - 1) * resolution + miny - y[0]) ** 2

    # Transition cost from pdf
    levelsup = np.arange(0, maxstep + 1)
    levelsdown = np.arange(minstep, 0)
    levels = np.concatenate((levelsdown, levelsup)) + wt['zeroindex']
    levels = np.clip(levels, 0, len(wt['pdf']) - 1)
    pdfcost1 = wt['pdf'][levels]
    pdfcostoffset1 = len(levelsdown)

    # Filtered output memory
    zhat = np.full_like(S, y[0], dtype=float)
    costtrack = 0  # Placeholder

    # Viterbi forward pass
    cost, S, zhat = viterbi_cost_loop(
        S, zhat, cost, y, b, a, low, high, miny, resolution,
        pdfcost1, pdfcostoffset1, sigmabar, pass_idx, T
    )

    # Backtrack
    end_idx = np.argmin(cost[:high[-1] - low[-1] + 1])
    est = reconstruction(S, low, end_idx-1, resolution, miny)
    return est, costtrack

from numba import njit
import numpy as np

@njit(cache=True)
def viterbi_cost_loop(S, zhat, cost, y, b, a, low, high, miny, resolution,
                      pdfcost1, pdfcostoffset1, sigmabar, pass_idx, T):
    """
    JIT-compiled Viterbi forward pass loop for step detection.

    Inputs:
        S          : state transition matrix (to be updated)
        zhat       : estimated signal matrix (to be updated)
        cost       : cost vector for each state (to be updated)
        y          : noisy signal (1D)
        b, a       : FIR/IIR coefficients
        low, high  : envelope bounds (int arrays)
        miny       : signal base offset
        resolution : quantization step
        pdfcost1   : transition cost (array of log-likelihoods)
        pdfcostoffset1 : center offset for step size PDF
        sigmabar   : total variance scale
        pass_idx   : current pass index (1-based)
        T          : number of timepoints

    Returns:
        Updated cost, S, and zhat arrays (in-place modification)
    """

    max_a_b = max(len(a), len(b))
    maxband = S.shape[0]

    for t in range(max_a_b - 1, T):
        low_prev = low[t - 1]
        high_prev = high[t - 1]
        low_t = low[t]
        high_t = high[t]

        llastrow = high_prev - low_prev + 1
        lrow = high_t - low_t + 1

        z = np.zeros((llastrow, lrow))
        dx = np.zeros((llastrow, lrow), dtype=np.int32)

        # Step 1: compute dx and initial z
        for i in range(llastrow):
            for j in range(lrow):
                dx[i, j] = - (low_prev + i) + (low_t + j) + pdfcostoffset1
                z[i, j] = b[0] * ((low_prev + i - 1) * resolution + miny)

        # Step 2: dynamic convolution filtering (discrete transfer function)
        lastrowtmp = np.zeros(llastrow, dtype=np.int32)
        for i in range(llastrow):
            lastrowtmp[i] = low_prev + i

        for i in range(1, max_a_b):
            if t - i >= 1:
                for p in range(llastrow):
                    idx = lastrowtmp[p] - low[t - i]
                    if idx < 0 or idx >= maxband:
                        continue

                    if i < len(b):
                        z[p, :] += b[i] * ((lastrowtmp[p] - 1) * resolution + miny)

                    if i < len(a):
                        if pass_idx == 1:
                            z[p, :] -= a[i] * zhat[idx, t - i]
                        else:
                            z[p, :] -= a[i] * y[t - i]

                    lastrowtmp[p] = low[t - i] - 1 + S[idx, t - i]

        dx = np.clip(dx, 0, len(pdfcost1) - 1)

        # Step 3: compute noise and measurement cost
        noisecost = np.zeros((llastrow, lrow))
        for i in range(llastrow):
            for j in range(lrow):
                noisecost[i, j] = pdfcost1[dx[i, j]] * sigmabar

        measurementcost = np.zeros((llastrow, lrow))
        for i in range(llastrow):
            for j in range(lrow):
                measurementcost[i, j] = (y[t] - z[i, j]) ** 2

        # Step 4: total cost and best step
        tempcost = np.zeros((llastrow, lrow))
        for i in range(llastrow):
            for j in range(lrow):
                tempcost[i, j] = cost[i] + measurementcost[i, j] + noisecost[i, j]

        for j in range(lrow):
            mincost = tempcost[0, j]
            best_i = 0
            for i in range(1, llastrow):
                if tempcost[i, j] < mincost:
                    mincost = tempcost[i, j]
                    best_i = i
            cost[j] = mincost
            S[j, t] = best_i + 1  # 1-based indexing (like MATLAB)
            zhat[j, t] = z[best_i, j]

    return cost, S, zhat


def findenvelope2(signal, window):
    """
    Estimate lower and upper envelopes and compute step-fitting bounds.

    Parameters:
        signal : 1D numpy array
        window : int, half-width of the window for filtering

    Returns:
        low      : lower envelope (expanded)
        high     : upper envelope (expanded)
        maxstep  : max step size across signal
        minstep  : min step size across signal
        miny     : min value of expanded lower envelope
        maxy     : max value of expanded upper envelope
    """
    # Compute local envelopes using min/max filters
    low = minfilter(signal, window)
    high = maxfilter(signal, window)

    # Compute midpoint and expand envelopes
    mid = (low + high) / 2
    low = 2 * low - mid - 2
    high = 2 * high - mid + 2

    # Step constraints
    maxstep = np.max(high - low)
    minstep = np.min(low - high)
    miny = np.min(low)
    maxy = np.max(high)

    return low, high, maxstep, minstep, miny, maxy

def findenvelope3(signal, window, resolution, low, high, ctx):
    """
    Refines signal envelopes for Viterbi step detection.

    Parameters:
        signal     : 1D numpy array, the input signal
        window     : int, half-width of filter window
        resolution : float, current resolution estimate
        low        : numpy array, previous lower envelope
        high       : numpy array, previous upper envelope
        ctx        : StepfitContext object

    Returns:
        low        : updated lower envelope
        high       : updated upper envelope
        maxstep    : maximum step size across envelope
        minstep    : minimum step size across envelope
        miny       : min value of lower envelope
        maxy       : max value of upper envelope
        resolution : possibly refined resolution
    """

    # Adjust top and bottom thickness during early passes
    if ctx.pass_idx <= 2:
        ctx.topthickness = max(5 * resolution, np.mean(high - signal))
        ctx.bottomthickness = max(5 * resolution, np.mean(signal - low))

    # If envelope too loose, recalculate tightly
    if not (np.min(np.abs(high - signal)) <= resolution or
            np.min(np.abs(low - signal)) <= resolution):
        ctx.envelopeflag = 1
        ctx.topthickness /= 2
        ctx.bottomthickness /= 2
        low = signal - ctx.bottomthickness
        high = signal + ctx.topthickness
    else:
        ctx.envelopeflag = 0

        # Extend only where signal touches envelope
        topper = ctx.topthickness * (np.abs(high - signal) <= 6 * resolution)
        lower = ctx.bottomthickness * (np.abs(signal - low) <= 6 * resolution)
        low -= lower
        high += topper

    # Filter envelopes to enforce smoothness
    low = np.minimum(low, minfilter(signal - 2 * resolution, window))
    high = np.maximum(high, maxfilter(signal + 2 * resolution, window))

    # Update resolution
    finelevel = max(100, 1)
    resolution = min(resolution, 0.5 * (np.min(high - low) / 10 + np.max(high - low) / finelevel))

    # Enforce minimum separation
    high = np.maximum(low + 2 * resolution, high)

    # Compute step bounds
    maxstep = np.max(high[1:] - low[:-1])
    minstep = np.min(low[1:] - high[:-1])
    miny = np.min(low)
    maxy = np.max(high)

    return low, high, maxstep, minstep, miny, maxy, resolution


def noise_std(y):
    """Estimate standard deviation of noise in signal y."""
    y = np.asarray(y)
    y = y[:min(len(y), 10000)]

    N = len(y) // 10
    w = np.unique(np.floor(np.logspace(np.log10(1), np.log10(N), 50)).astype(int))
    m = np.zeros(len(w))

    for i, win in enumerate(w):
        v, _ = movingvar(y, win)
        if len(v) >= 2 * win:
            m[i] = np.mean(v[win:-win])  # Central part only

    st = np.sqrt(m)
    dst = np.abs(np.diff(st))

    # Butterworth filter (1st order, 0.6 cutoff)
    b, a = butter(1, 0.6)

    dst1 = dst.copy()
    for _ in range(3):
        dst1 = 0.5 * lfilter(b, a, dst1, axis=0, zi=[dst1[0]])[0] + \
               0.5 * np.flip(lfilter(b, a, np.flip(dst1), axis=0, zi=[dst1[-1]])[0])

    _, peaks = lmax(-dst1)
    if len(peaks) == 0:
        return np.min(st)  # fallback
    noiseD = np.min(st[peaks + 1])
    return noiseD

@njit(cache=True)
def movingvar(x, m):
    N = len(x)
    offset = m // 2

    # 1. Moving mean (centered)
    s = np.zeros(N)
    for i in range(N):
        start = max(0, i - offset)
        end = min(N, i + offset + 1)
        s[i] = np.mean(x[start:end])

    # 2. Detrend and square
    x_detrended = x - s
    x_squared = x_detrended ** 2

    # 3. Moving average of squared values
    v = np.zeros(N)
    for i in range(N):
        start = max(0, i - offset)
        end = min(N, i + offset + 1)
        v[i] = np.mean(x_squared[start:end])

    return v, s

def movingmean(x, m):
    """
    Compute a centered moving average using a uniform kernel.
    Parameters:
        x : 1D numpy array
        m : int, window size
    Returns:
        xm : 1D numpy array of moving averages
    """
    x = np.asarray(x).flatten()
    return uniform_filter1d(x, size=m, mode='nearest')

@njit(cache=True)
def lmax(x):
    der = np.diff(x)
    sign_der = np.sign(der)
    second_der = np.diff(sign_der)
    i = np.where(second_der < 0)[0] + 1
    y = x[i]
    return y, i

def minfilter(signal, window):
    """Local minimum filter with window size (2*window + 1)."""
    size = 2 * window + 1
    return minimum_filter1d(signal, size=size, mode='nearest')

def maxfilter(signal, window):
    """Local maximum filter with window size (2*window + 1)."""
    size = 2 * window + 1
    return maximum_filter1d(signal, size=size, mode='nearest')

@njit(cache=True)
def reconstruction(S, low, lastindex, resolution, miny):
    """
    Reconstruct step signal from Viterbi backtracking matrix.

    Parameters:
        S         : 2D numpy array (states x time) with backtracking indices
        low       : 1D numpy array of lower envelope values per timepoint
        lastindex : int, final index of the minimum-cost state (0-based)
        resolution: float, signal resolution
        miny      : float, base signal offset

    Returns:
        est : 1D numpy array of reconstructed step values
    """
    N = S.shape[1]
    est = np.zeros(N)

    # Last time point (adjust for 0-based indexing)
    est[N - 1] = (low[N - 1] + lastindex - 1) * resolution + miny
    ind = lastindex

    # Backtrack
    for k in range(N - 1, 0, -1):
        ind = S[ind, k] - 1
        est[k - 1] = (ind + low[k - 1] - 1) * resolution + miny

    return est