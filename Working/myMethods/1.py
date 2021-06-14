import numpy as np
from PIL import Image
from scipy.signal import convolve2d


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    miu1 = filter2(im1, window, 'valid')
    miu2 = filter2(im2, window, 'valid')
    miu1_sq = miu1 * miu1
    miu2_sq = miu2 * miu2
    mean12 = miu1 * miu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - miu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - miu2_sq
    sigma12 = filter2(im1 * im2, window, 'valid') - mean12

    ssim = ((2 * mean12 + C1) * (2 * sigma12 + C2)) / ((miu1_sq + miu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim))


if __name__ == "__main__":
    Image1 =Image.open('img.png')
    Image1 = np.array(Image1)
    Image2 = np.random.rand(28, 28)

    print(compute_ssim(Image1, Image1))
