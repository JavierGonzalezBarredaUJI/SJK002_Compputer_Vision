#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
from scipy.ndimage import filters, convolve1d, convolve
from scipy.signal import medfilt2d, convolve2d
import numpy as np
import matplotlib.pyplot as plt
import math as math
import glob
import os
import sys
import time
from scipy import signal

sys.path.append("../../p1/code") # set the path for visualPercepUtils.py
import visualPercepUtils as vpu

# -----------------------
# Salt & pepper noise
# -----------------------

def addSPNoise(im, percent):
    # Now, im is a PIL image (not a NumPy array)
    # percent is in range 0-100 (%)

    # convert image it to numpy 2D array and flatten it
    im_np = np.array(im)
    im_shape = im_np.shape  # keep shape for later use (*)
    im_vec = im_np.flatten()  # this is a 1D array # https://www.geeksforgeeks.org/differences-flatten-ravel-numpy/

    # generate random locations
    N = im_vec.shape[0]  # number of pixels
    m = int(math.floor(percent * N / 100.0)) # number of pixels corresponding to the given percentage
    locs = np.random.randint(0, N, m)  # generate m random positions in the 1D array (index 0 to N-1)

    # generate m random S/P values (salt and pepper in the same proportion)
    s_or_p = np.random.randint(0, 2, m)  # 2 random values (0=salt and 1=pepper)

    # set the S/P values in the random locations
    im_vec[locs] = 255 * s_or_p  # values after the multiplication will be either 0 or 255

    # turn the 1D array into the original 2D image
    im2 = im_vec.reshape(im_shape) # (*) here is where we use the shape that we saved earlier

    # convert Numpy array im2 back to a PIL Image and return it
    return Image.fromarray(im2)


def testSandPNoise(im, percents):
    imgs = []
    for percent in percents:
        imgs.append(addSPNoise(im, percent))
    return imgs


# -----------------
# Gaussian noise
# -----------------
"""
def addGaussianNoise(im, sd=5):
    return im + np.random.normal(loc=0, scale=sd, size=im.shape)
"""

def addGaussianNoise(im, sd=5):
    noise = np.random.normal(loc=0, scale=sd, size=im.shape)
    noisy_im = im + noise
    # Clip thee values to the valid range [0, 255]
    noisy_im = np.clip(noisy_im, 0, 255)
    
    return noisy_im.astype(np.uint8)

def testGaussianNoise(im, sigmas):
    imgs = []
    for sigma in sigmas:
        print('testing sigma:', sigma)
        imgs.append(addGaussianNoise(im, sigma))
        print(len(imgs))
    return imgs


# -------------------------
# Average (or mean) filter
# -------------------------
def plotRunningTimes(runtimes_2d, runtimes_sep):
    plt.figure(figsize=(10, 6))
    plt.plot(runtimes_2d, label='2D Filter')
    plt.plot(runtimes_sep, label='Separable Filter')
    plt.xlabel('Experiment')
    plt.ylabel('Running Time (s)')
    plt.title('Comparison of 2D and Separable Mean Filter Running Times')
    plt.legend()
    plt.show()
    
    
def plotResults(results):
    for idx, result in enumerate(results):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(result['Original with Noise'], cmap='gray')
        plt.title('Original with Noise')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(result['2D Filtered Image'], cmap='gray')
        plt.title('2D Filtered Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(result['Separable Filtered Image'], cmap='gray')
        plt.title('Separable Filtered Image')
        plt.axis('off')
        
        plt.show()


def averageFilter(im, filterSize):
    mask = np.ones((filterSize, filterSize))
    mask = np.divide(mask, np.sum(mask)) # can you think of any alternative for np.sum(mask)?
    return filters.convolve(im, mask)


def averageFilterSep(im, filterSize):
    # Create a 1D filter along rows
    row_filter = np.ones((filterSize,)) / filterSize
    
    # Convolve along rows
    im_filtered_rows = convolve1d(im, row_filter, axis=0, mode='constant', cval=0.0)
    
    # Convolve along columns using the transposed result from the previous step
    im_filtered_sep = convolve1d(im_filtered_rows, row_filter, axis=1, mode='constant', cval=0.0)
    
    return im_filtered_sep


def testAverageFilter(im_clean, params):
    runtimes_2d = []
    runtimes_sep = []
    imgs = []
    imgsep = []
    results = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg) # salt and pepper noise
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            start_time = time.time()
            im = averageFilter(im_dirty, filterSize)
            imgs.append(im)
            elapsed_time = time.time() - start_time
            runtimes_2d.append(elapsed_time)
            
            imgsep.append(np.array(im_dirty))
            start_time = time.time()
            imsep = averageFilterSep(im_dirty, filterSize)
            imgsep.append(imsep)
            elapsed_time = time.time() - start_time
            runtimes_sep.append(elapsed_time)
                
            results.append({
                'Original with Noise': im_dirty,
                '2D Filtered Image': im,
                'Separable Filtered Image': imsep
            }) 
    
    plotRunningTimes(runtimes_2d, runtimes_sep)
    plotResults(results)
    
    return imgs




# -----------------
# Gaussian filter
# -----------------

def gaussianFilter(im, sigma=5):
    # im is PIL image
    return filters.gaussian_filter(im, sigma)


def gaussianFilterExplicit(im, sigma=5):
    # (a) Generate 1D Gaussian vector
    n = 101  # Size of the vector
    gv1d = signal.windows.gaussian(n, sigma)
    
    # (b) Generate 2D Gaussian matrix
    gv2d = np.outer(gv1d, gv1d)
    
    # (c) Convolve the image using gv2d as a mask
    im_convolved_2d = convolve(im, gv2d, mode='constant', cval=0.0)
    
    # original function of gaussian filter
    im_gaussian_filter = filters.gaussian_filter(im, sigma)
    
    # (d) Apply convolution with two 1D masks
    im_convolved_sep = convolve2d(convolve2d(im, gv1d.reshape(-1, 1), mode='same', boundary='fill', fillvalue=0.0),
                               gv1d.reshape(1, -1), mode='same', boundary='fill', fillvalue=0.0)
    
    # plots
    # a
    plt.figure()
    plt.imshow(gv1d.reshape(-1, 1), cmap='gray', interpolation='none')
    plt.title('1D Gaussian Vector')
    plt.show()
    
    #b
    plt.figure()
    plt.imshow(gv2d, cmap='gray', interpolation='none')
    plt.title('2D Gaussian Matrix')
    plt.show()
    
    # Display the original image with noise
    plt.figure()
    plt.imshow(im, cmap='gray', interpolation='none')
    plt.title('Original Image with Noise')
    plt.show()
    
    #c
    plt.figure()
    plt.imshow(im_convolved_2d, cmap='gray', interpolation='none')
    plt.title('Convolution with 2D Gaussian Mask')
    plt.show()
    
    # original gaussian filter function
    plt.figure()
    plt.imshow(im_gaussian_filter, cmap='gray', interpolation='none')
    plt.title('Original gaussianFilter Function')
    plt.show()
    
    #d
    plt.figure()
    plt.imshow(im_convolved_sep, cmap='gray', interpolation='none')
    plt.title('Convolution with Separable Gaussian Masks')
    plt.show()
    
    
    return filters.gaussian_filter(im, sigma)


def gaussian_kernel(size, sigma):
    #Generate a 2D Gaussian kernel.
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / g.sum()

def quotientImage(im, sigma):
    #Compute the quotient image by dividing the image by its blurred version.
    # Create a 2D Gaussian kernel
    kernel = gaussian_kernel(6*sigma, sigma)

    # Blur the image using the Gaussian kernel
    blurred_im = convolve(im, kernel, mode='constant', cval=0.0)

    # Compute the quotient image by pixel-wise division
    quotient_im = im / (blurred_im + 1e-8)  # Adding a small constant to avoid division by zero

    return quotient_im
    


def testGaussianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sigma in params['sd_gauss_noise']:
        im_dirty = addGaussianNoise(im_clean, sigma)
        for filterSize in params['sd_gauss_filter']:
            imgs.append(np.array(im_dirty))
            imgs.append(gaussianFilterExplicit(im_dirty, filterSize))
    return imgs


# -----------------
# Median filter
# -----------------

def medianFilter(im, filterSize):
    return medfilt2d(im, filterSize)

def testMedianFilter(im_clean, params):
    # This function turned out to be too similar to testAverageFilter
    # This is a good sign that code factorization is called for :)
    imgs = []
    for sp_pctg in params['sp_pctg']:
        im_dirty = addSPNoise(im_clean, sp_pctg)
        for filterSize in params['filterSizes']:
            imgs.append(np.array(im_dirty))
            imgs.append(medianFilter(im_dirty, filterSize))
    return imgs


# -----------------
# Test image files
# -----------------

path_input = './imgs-P2/'
path_output = './imgs-out-P2/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.pgm")
else:
    files = [path_input + 'lena256.pgm']  # lena256, lena512

# --------------------
# Tests to perform
# --------------------

testsNoises = ['testSandPNoise', 'testGaussianNoise']
testsFilters = ['testAverageFilter', 'testAverageFilterSep']
bAllTests = False
if bAllTests:
    tests = testsNoises + testsFilters
else:
    tests = ['testGaussianNoise']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testGaussianNoise': 'Gaussian noise',
             'testSandPNoise': 'Salt & Pepper noise',
             'testAverageFilter': 'Mean filter',
             'testAverageFilterSep': 'Mean filter separate',
             'testGaussianFilter': 'Gaussian filter',
             'testMedianFilter': 'Median filter'}

suffixFiles = {'testGaussianNoise': '_GN',
             'testSandPNoise': '_SPN',
             'testAverageFilter': '_mean',
             'testAverageFilterSep': '_meanSep',
             'testGaussianFilter': '_gaussian',
             'testMedianFilter': '_median'}

bSaveResultImgs = True

# -----------------------
# Parameters of noises
# -----------------------
percentagesSandP = [3]  # ratio (%) of image pixes affected by salt and pepper noise
gauss_sigmas_noise = [3, 5, 10]  # standard deviation (for the [0,255] range) for Gaussian noise

# -----------------------
# Parameters of filters
# -----------------------

gauss_sigmas_filter = [1.2]  # standard deviation for Gaussian filter
avgFilter_sizes = [3, 7, 15]  # sizes of mean (average) filter
medianFilter_sizes = [3, 7, 15]  # sizes of median filter

testsUsingPIL = ['testSandPNoise']  # which test(s) uses PIL images as input (instead of NumPy 2D arrays)


# -----------------------------------------
# Apply defined tests and display results
# -----------------------------------------

def saveImg(imfile, im2, test):
    dirname,basename = os.path.dirname(imfile), os.path.basename(imfile)
    fname, fext = os.path.splitext(basename)
    #print(dname,basename)
    pil_im = Image.fromarray(im2.astype(np.uint8))  # from array to Image
    pil_im.save(path_output+'//'+fname + suffixFiles[test] + fext)

def doTests():
    print("Testing on", files)
    for imfile in files:
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array
        
        #im = np.array(Image.open(imfile))

        for test in tests:

            if test == "testGaussianNoise":
                params = gauss_sigmas_noise
                subTitle = r", $\sigma$: " + str(params)
            elif test == "testSandPNoise":
                params = percentagesSandP
                subTitle = ", %: " + str(params)
            elif test == "testAverageFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testMedianFilter":
                params = {}
                params['filterSizes'] = avgFilter_sizes
                params['sp_pctg'] = percentagesSandP
                subTitle = ", " + str(params)
            elif test == "testGaussianFilter":
                params = {}
                params['sd_gauss_noise'] = gauss_sigmas_noise
                params['sd_gauss_filter'] = gauss_sigmas_filter
                subTitle = r", $\sigma_n$ (noise): " + str(gauss_sigmas_noise) + ", $\sigma_f$ (filter): " + str(gauss_sigmas_filter)
            
            if test in testsUsingPIL:
                outs_pil = eval(test)(im_pil, params)
                outs_np = vpu.pil2np(outs_pil)
            else:
                # apply test to given image and given parameters
                outs_np = eval(test)(im, params)
                print("num images", len(outs_np))
            print(len(outs_np))
            # display original image, noisy images and filtered images
            vpu.showInGrid([im] + outs_np, title=nameTests[test] + subTitle)
            
            if bSaveResultImgs:
                im2 = outs_np[0]
                saveImg(imfile, im2, test)


if __name__ == "__main__":
    doTests()
    
    #run()
    