
import numpy as np
from scipy.signal import correlate2d
from scipy.fft import fft2, ifft2
from scipy.signal import fftconvolve

def normxcorr2(template, image):
    """Perform normalized cross-correlation of template over image."""
    template = (template - np.mean(template)) / (np.std(template) + 1e-10)
    image = (image - np.mean(image)) / (np.std(image) + 1e-10)

    return correlate2d(image, template, mode='full')

def calcoffset(img1, img2):
    """Python equivalent of MATLAB's calcoffset using normxcorr2."""
    xcorrmat = normxcorr2_fft(img1, img2)
    ypeak, xpeak = np.unravel_index(np.argmax(xcorrmat), xcorrmat.shape)
    offset = [-(ypeak - img2.shape[0]), -(xpeak - img2.shape[1])]
    return offset, xpeak, ypeak, xcorrmat

def normxcorr2_fft(template, image):
    """Fast normalized cross-correlation using FFT"""
    template = template - np.mean(template)
    image = image - np.mean(image)

    # Flip the template
    template_flipped = np.flipud(np.fliplr(template))

    # Compute raw cross-correlation
    corr = fftconvolve(image, template_flipped, mode='full')

    # Compute normalization terms
    template_norm = np.sqrt(np.sum(template ** 2))

    kernel = np.ones(template.shape)
    image_sq = image ** 2
    image_sum = fftconvolve(image, kernel, mode='full')
    image_sq_sum = fftconvolve(image_sq, kernel, mode='full')
    image_norm = np.sqrt(image_sq_sum - image_sum ** 2 / np.prod(template.shape))

    with np.errstate(invalid='ignore', divide='ignore'):
        ncc = corr / (image_norm * template_norm)

    ncc[np.isnan(ncc)] = 0
    return ncc
