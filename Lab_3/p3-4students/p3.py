#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Act as a computer vision expert. As you know, the mean filter is separable. What does this imply? Well, we can obtain the same result by applying sequentially two 1D filters (one over rows and the other over columns) that by applying one 2D filter, with the benefit that the cost of applying two onedimensional filters is smaller than that of applying a single 2D filter. Write the separable version averageFilterSep(), of the mean filter given in averageFilter().
Function: '''def averageFilter(im, filterSize):
    mask = np.ones((filterSize, filterSize))
    mask = np.divide(mask, np.sum(mask))
    return filters.convolve(im, mask)
'''"""

from PIL import Image
from scipy.ndimage import filters
import numpy.fft as fft
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import time
import requests
import io

sys.path.append("../../p1/code") # set the path for visualPercepUtils.py
import visualPercepUtils as vpu

# ----------------------
# Fourier Transform
# ----------------------

def FT(im):
    # https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
    return fft.fftshift(fft.fft2(im))  # perform also the shift to have lower frequencies at the center
    #return fft.fft2(im)


def IFT(ft):
    return fft.ifft2(fft.ifftshift(ft))  # assumes ft is shifted and therefore reverses the shift before IFT
    #return fft.ifft2(ft)


def testFT(im, params=None):
    ft = FT(im)
    #print(ft.shape)
    phase = np.angle(ft)
    magnitude = np.log(np.absolute(ft))
    bMagnitude = True
    if bMagnitude:
        im2 = np.absolute(IFT(ft))  # IFT consists of complex number. When applied to real-valued data the imaginary part should be zero, but not exactly for numerical precision issues
    else:
        im2 = np.real(IFT(ft)) # with just the module we can't appreciate the effect of a shift in the signal (e.g. if we use fftshift but not ifftshift, or viceversa)
        # Important: one case where np.real() is appropriate but np.absolute() is not is where the sign in the output is relevant
    return [magnitude, phase, im2]


def exercise1(result):
    # (a) Compute the lowest and highest magnitudes
    lowest_magnitude = np.min(result[0])
    highest_magnitude = np.max(result[0])

    # (b) Display a boxplot with the magnitude values
    plt.boxplot(result[0].flatten())
    plt.title('Magnitude Boxplot')
    plt.xlabel('Magnitude')
    plt.show()

    # (c) Plot a histogram of the phase values
    plt.hist(result[1].flatten(), bins=50, edgecolor='black')
    plt.title('Phase Histogram')
    plt.xlabel('Phase')
    plt.ylabel('Frequency')
    plt.show()

    print(f'(a) Lowest Magnitude: {lowest_magnitude}')
    print(f'(a) Highest Magnitude: {highest_magnitude}')


# -----------------------
# Convolution theorem
# -----------------------

# the mask corresponding to the average (mean) filter
def avgFilter(filterSize):
    mask = np.ones((filterSize, filterSize))
    return mask/np.sum(mask)


# apply average filter in the spatial domain
def averageFilterSpace(im, filterSize):
    return filters.convolve(im, avgFilter(filterSize))


# apply average filter in the frequency domain
def averageFilterFrequency(im, filterSize):
    filterMask = avgFilter(filterSize)  # the usually small mask
    filterBig = np.zeros_like(im, dtype=float)  # as large as the image (dtype is important here!)

    # Now, place filter (the "small" filter mask) at the center of the "big" filter

    ## First, get sizes
    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2  # half width and height of the "small" mask
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2  # half width and height of the "big" mask

    ## Then, paste the small mask at the center using the sizes computed before as an aid
    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask

    # FFT of the big filter
    filterBig = fft.ifftshift(filterBig)  # shift origin at upper-left corner

    # Finally, IFT of the element-wise product of the FT's
    imFiltFreq = np.absolute(IFT(FT(im) * FT(filterBig)))
    FG = np.absolute(FT(im) * FT(filterBig))
    filterFT = FT(filterBig)

    return imFiltFreq, FG, filterFT  # both '*' and multiply() perform elementwise product


def testConvTheo(im, params=None):
    #im = imfiles[0]
    filterSize = params['filterSize']
    filterSizes = [3, 5, 7, 9]
    sigma = 1.2
    sigmas = [1, 2, 3, 4]
    n = 3

    # image filtered with a convolution in spatial domain
    imFiltSpace = averageFilterSpace(im, filterSize)
    #imFiltSpace = gaussianFilterSpace(im, filterSize, sigma)

    # image filtered in frequency domain
    imFiltFreq, FG, filterFT = averageFilterFrequency(im, filterSize)
    #imFiltFreq, FG, filterFT = gaussianFilterFrequency(im, filterSize, sigma)
    
    #gaussianPlot(im, imFiltSpace, imFiltFreq, FG, filterFT)
    #plot_mean_filter_frequency_response(im, filterSizes)
    #plot_gaussian_filter_frequency_response(im, filterSize, sigmas)
    #plot_time_mean_filter_frequency_response(imfiles, filterSizes)
    img_filterd = my_filter(im, n)
    plot_myfilter(im, img_filterd)

    # How much do they differ?
    # To quantify the difference, we use the Root Mean Square Measure (https://en.wikipedia.org/wiki/Root_mean_square)
    margin = 5  # exclude some outer pixels to reduce the influence of border effects
    rms = np.linalg.norm(imFiltSpace[margin:-margin, margin:-margin] - imFiltFreq[margin:-margin, margin:-margin], 2) / np.prod(im.shape)
    print("Images filtered in space and frequency differ in (RMS):", rms)
    
    return [imFiltSpace, imFiltFreq]


def my_mask(n):
    matrix = np.zeros((n, n), dtype=int)
    matrix[np.triu_indices(n, k=1)] = -1
    matrix[np.tril_indices(n, k=-1)] = 1
    return matrix


def my_filter(im, n):
    mask = my_mask(n) # Get the mask from my_mask function      
    im_fft = fft.fft2(im) # FFT of the image
    mask_fft = fft.fft2(mask, s=im.shape) # FFT of the mask
    im_filtered_fft = im_fft * mask_fft # Apply the filter in the frequency domain
    im_filtered = np.real(fft.ifft2(im_filtered_fft)) # Inverse FFT to obtain the filtered image
    return im_filtered


def plot_myfilter(image, filtered_image):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image with n=3')

    plt.show()
    

def gaussianFilter(filterSize, sigma):
    x = np.arange(-filterSize // 2 + 1., filterSize // 2 + 1.)
    y = np.arange(-filterSize // 2 + 1., filterSize // 2 + 1.)
    xx, yy = np.meshgrid(x, y, sparse=True)
    filterMask = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return filterMask / np.sum(filterMask)

def gaussianFilterSpace(im, filterSize, sigma):
    return filters.convolve(im, gaussianFilter(filterSize, sigma))

def gaussianFilterFrequency(im, filterSize, sigma):
    filterMask = gaussianFilter(filterSize, sigma)
    filterBig = np.zeros_like(im, dtype=float)

    w, h = filterMask.shape
    w2, h2 = w / 2, h / 2
    W, H = filterBig.shape
    W2, H2 = W / 2, H / 2

    filterBig[int(W2 - w2):int(W2 + w2), int(H2 - h2):int(H2 + h2)] = filterMask
    filterBig = fft.ifftshift(filterBig)
    
    imFiltFreq = np.absolute(IFT(FT(im) * FT(filterBig)))
    FG = np.absolute(FT(im) * FT(filterBig))
    filterFT = FT(filterBig)

    return imFiltFreq, FG, filterFT


def gaussianPlot(im, imFiltSpace, imFiltFreq, FG, filterFT):
    # Display the results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(im, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 3, 2)
    plt.imshow(imFiltSpace, cmap='gray')
    plt.title('Filtered in Space')

    plt.subplot(2, 3, 3)
    plt.imshow(imFiltFreq, cmap='gray')
    plt.title('Filtered in Frequency')

    plt.subplot(2, 3, 4)
    plt.imshow(np.log1p(np.abs(filterFT)), cmap='gray')
    plt.title('Fourier Transform of Filter')

    plt.subplot(2, 3, 6)
    plt.imshow(np.log1p(np.abs(FG)), cmap='gray')
    plt.title('Element-wise Product of Fourier Transforms of the image and the filter')

    plt.show()
    
    
def plot_mean_filter_frequency_response(im, filter_sizes):
    filterFT_lst = []
    for size in filter_sizes:
        imFiltFreq, FG, filterFT = averageFilterFrequency(im, size)
        filterFT_lst.append(filterFT)
    
    # Display the results
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(np.log1p(np.abs(filterFT_lst[0])), cmap='gray')
    plt.title('size: 3x3')

    plt.subplot(2, 2, 2)
    plt.imshow(np.log1p(np.abs(filterFT_lst[1])), cmap='gray')
    plt.title('size: 5x5')

    plt.subplot(2, 2, 3)
    plt.imshow(np.log1p(np.abs(filterFT_lst[2])), cmap='gray')
    plt.title('size: 7x7')

    plt.subplot(2, 2, 4)
    plt.imshow(np.log1p(np.abs(filterFT_lst[3])), cmap='gray')
    plt.title('size: 9x9')

    plt.show()

    
    
    
def plot_gaussian_filter_frequency_response(im, filterSize,sigmas):
    filterFT_lst = []
    for sigma in sigmas:
        imFiltFreq, FG, filterFT = gaussianFilterFrequency(im, filterSize, sigma)
        filterFT_lst.append(filterFT)
    
    # Display the results
    plt.figure(figsize=(8, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(np.log1p(np.abs(filterFT_lst[0])), cmap='gray')
    plt.title('sigma: 1')

    plt.subplot(2, 2, 2)
    plt.imshow(np.log1p(np.abs(filterFT_lst[1])), cmap='gray')
    plt.title('sigma: 2')

    plt.subplot(2, 2, 3)
    plt.imshow(np.log1p(np.abs(filterFT_lst[2])), cmap='gray')
    plt.title('sigma: 3')

    plt.subplot(2, 2, 4)
    plt.imshow(np.log1p(np.abs(filterFT_lst[3])), cmap='gray')
    plt.title('sigma: 4')

    plt.show()
    
    
def plot_time_mean_filter_frequency_response(imfiles, filterSizes):
    time_lst_filters = []
    time_lst_im = []
    im = imfiles[0] # 255x255
    size = filterSizes[1] #7x7
    im_sizes = ['255x255', '512x512']
    n = 1000
    
    
    time_mean_filters = []
    for filterSize in filterSizes:
        for i in range(n):
            initTime = time.time()
            imFiltFreq, FG, filterFT = averageFilterFrequency(im, filterSize)
            endTime = time.time()
            time_lst_filters.append(endTime - initTime)
        time_mean_filters.append(np.mean(time_lst_filters))
    
    time_mean_im = []   
    for imfile in imfiles:
        for i in range(n):
            initTime = time.time()
            imFiltFreq, FG, filterFT = averageFilterFrequency(imfile, size)
            endTime = time.time()
            time_lst_im.append(endTime - initTime)
        time_mean_im.append(np.mean(time_lst_im))
    
    # Crear el primer gráfico
    plt.subplot(2, 1, 1)  # 2 filas, 1 columna, primer gráfico
    plt.plot(filterSizes, time_mean_filters, marker='o', linestyle='-')
    plt.title('image size fixed (255x255)')
    plt.xlabel('Filter Sizes')
    plt.ylabel('Time')
    
    plt.tight_layout()

    plt.subplot(2, 1, 2)  # 2 filas, 1 columna, segundo gráfico
    plt.plot(im_sizes, time_mean_im, marker='o', linestyle='-')
    plt.title('filter size fixed (7x7)')
    plt.xlabel('Image sizes')
    plt.ylabel('Time')

    plt.show()
    
        
        
    



# -----------------------------------
# High-, low- and band-pass filters
# -----------------------------------

# generic band-pass filter (both, R and r, given) which includes the low-pass (r given, R not)
# and the high-pass (R given, r not) as particular cases
def bandPassFilter(shape, r=None, R=None):
    n, m = shape
    m2, n2 = np.floor(m / 2.0), np.floor(n / 2.0)
    [vx, vy] = np.meshgrid(np.arange(-m2, m2 + 1), np.arange(-n2, n2 + 1))
    distToCenter = np.sqrt(vx ** 2.0 + vy ** 2.0)
    if R is None:  # low-pass filter assumed
        assert r is not None, "at least one size for filter is expected"
        filter = distToCenter<r # same as np.less(distToCenter, r)
    elif r is None:  # high-pass filter assumed
        filter = distToCenter>R # same as np.greater(distToCenter, R)
    else:  # both, R and r given, then band-pass filter
        if r > R:
            r, R = R, r  # swap to ensure r < R (alternatively, warn the user, or throw an exception)
        filter = np.logical_and(distToCenter<R, distToCenter>r)
    filter = filter.astype('float')  # convert from boolean to float. Not strictly required

    bDisplay = True
    if bDisplay:
        plt.imshow(filter, cmap='gray')
        plt.show()
        plt.title("The filter in the frequency domain")
        # Image.fromarray((255*filter).astype(np.uint8)).save('filter.png')

    return filter


def testBandPassFilter(im, params=None):
    r, R = params['r'], params['R']
    filterFreq = bandPassFilter(im.shape, r, R)  # this filter is already in the frequency domain
    filterFreq = fft.ifftshift(filterFreq)  # shifting to have the origin as the FT(im) will be
    return [np.absolute(fft.ifft2(filterFreq * fft.fft2(im)))]  # the filtered image


# -----------------
# Test image files
# -----------------
path_input = './imgs-P3/'
path_output = './imgs-out-P3/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'lena255.pgm']  # lena255, habas, mimbre

# --------------------
# Tests to perform
# --------------------
bAllTests = False
if bAllTests:
    tests = ['testFT', 'testConvTheo', 'testBandPassFilter']
else:
    #tests = ['testFT']
    tests = ['testConvTheo']
    #tests = ['testBandPassFilter']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testFT': '2D Fourier Transform',
             'testConvTheo': 'Convolution Theorem (tested on mean filter)',
             'testBandPassFilter': 'Frequency-based filters ("high/low/band-pass")'
             }

bSaveResultImgs = False

testsUsingPIL = []  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


def load_image(url):
    response = requests.get(url)
    img = Image.open(io.BytesIO(response.content)).convert('L')  # Convert to grayscale
    return np.array(img).astype(float)

url_stp1 = 'https://homepages.inf.ed.ac.uk/rbf/HIPR2/images/stp1.gif'
url_stp2 = 'https://homepages.inf.ed.ac.uk/rbf/HIPR2/images/stp2.gif'


def inverse_transform_combination(F1, F2, lmbda):
    return np.real(fft.ifft2(lmbda * F1 + (1 - lmbda) * F2))

def exercise5(im1, im2):
    F1 = fft.fft2(im1)
    F2 = fft.fft2(im2)
    
    lambda_values = np.linspace(0, 1, 5)
    
    plt.figure(figsize=(15, 10))

    for i, lmbda in enumerate(lambda_values, 1):
        combined_image = inverse_transform_combination(F1, F2, lmbda)

        plt.subplot(2, 3, i)
        plt.imshow(combined_image, cmap='gray')
        plt.title(f'λ = {lmbda:.2f}')

    plt.show()



# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def doTests():
    print("Testing on", files)
    
    image1 = load_image(url_stp1)
    image2 = load_image(url_stp2)
    
    exercise5(image1, image2)
    
    #imfiles = []
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array
        #imfiles.append(im)
        #size = im_pil.size
    
        # Imprime el nombre del archivo y el tamaño de la imagen
        #print(f"Archivo: {imfile}, Tamaño: {size}")
        
        
        for test in tests:
            if test is "testFT":
                params = {}
                subTitle = ": I, |F|, ang(F), IFT(F)"
            elif test is "testConvTheo":
                params = {}
                params['filterSize'] = 7 #7
                subTitle = ": I, I*M, IFT(FT(I).FT(M))"
            else:
                params = {}
                r,R = 5,None # for low-pass filter
                # 5,30 for band-pass filter
                # None, 30 for high-pass filter
                params['r'], params['R'] = r,R
                # let's assume r and R are not both None simultaneously
                if r is None:
                    filter="high pass" + " (R=" + str(R) + ")"
                elif R is None:
                    filter="low pass" + " (r=" + str(r) + ")"
                else:
                    filter="band pass" + " (r=" + str(r) + ", R=" + str(R) + ")"
                subTitle = ", " + filter + " filter"

            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
                #exercise1(outs_np)
            print("# images", len(outs_np))
            print(len(outs_np))

            #vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)
        

if __name__ == "__main__":
    doTests()
