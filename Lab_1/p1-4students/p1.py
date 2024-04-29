#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import glob
import os
import visualPercepUtils as vpu


# functions
def histeq(im, nbins=256):
    imhist, bins = np.histogram(im.flatten(), list(range(nbins)), density=False)
    cdf = imhist.cumsum() # cumulative distribution function (CDF) = cummulative histogram
    factor = 255 / cdf[-1]  # cdf[-1] = last element of the cummulative sum = total number of pixels)
    im2 = np.interp(im.flatten(), bins[:-1], factor*cdf)
    return im2.reshape(im.shape), cdf


def darkenImg(im,p=2):
    return (im ** float(p)) / (255 ** (p - 1)) # try without the float conversion and see what happens


def brightenImg(im,p=2):
    return np.power(255.0 ** (p - 1) * im, 1. / p)  # notice this NumPy function is different to the scalar math.pow(a,b)


def brightenImg2(im, p=2):
    print('SHAPE: ', im.shape)
    if len(im.shape) == 3:
        result = np.zeros_like(im, dtype=np.float64)
        
        for channel in range(im.shape[2]):
            result[:, :, channel] = np.power(255.0 ** (p - 1) * im[:, :, channel], 1. / p)
            
    else:
        result = np.power(255.0 ** (p - 1) * im, 1. / p)
    
    return result


def darkenImg2(im, p=2):
    if len(im.shape) == 3:
        result = np.zeros_like(im, dtype=np.float64)
        
        for channel in range(im.shape[2]):
            result[:, :, channel] = (im[:, :, channel] ** float(p)) / (255 ** (p - 1))
            
    else:
        result = (im ** float(p)) / (255 ** (p - 1))
    
    return result


def checkBoardImg(im, m, n):
    # Ensure the image is 2D (grayscale)
    if len(im.shape) == 2:
        # Get image dimensions
        height, width = im.shape

        # Create an empty image with the same size and data type
        checkerboard_im = np.zeros_like(im)

        # Calculate the size of each cell
        cell_height = height // m
        cell_width = width // n

        # Create a boolean mask for the checkerboard pattern
        checkerboard_mask = np.fromfunction(lambda i, j: (i // cell_height + j // cell_width) % 2, (height, width), dtype=int)

        # Use the mask to invert the pixels in alternate cells
        checkerboard_im = np.where(checkerboard_mask, 255 - im, im)

        return checkerboard_im
    
    else:
        return im
    
    
def multiHist(im, n, bins):
    histograms = []

    def recursive_histogram(image, level, bins):
        # Calculate histogram at the current level
        histogram = np.histogram(image.flatten(), bins=bins)[0]
        histograms.append(histogram)

        # Base case: stop recursion when the desired level is reached
        if level == n:
            return

        # Recursive case: partition the image into four quadrants
        height, width = image.shape
        half_height, half_width = height // 2, width // 2

        # Top-left quadrant
        recursive_histogram(image[:half_height, :half_width], level + 1, bins)
        # Top-right quadrant
        recursive_histogram(image[:half_height, half_width:], level + 1, bins)
        # Bottom-left quadrant
        recursive_histogram(image[half_height:, :half_width], level + 1, bins)
        # Bottom-right quadrant
        recursive_histogram(image[half_height:, half_width:], level + 1, bins)

    recursive_histogram(im, 1, bins)

    return im, histograms


# test functions
def testHistEq(im):
    im2, cdf = histeq(im)
    return [im2, cdf]

def testDarkenImg(im):
    im2 = darkenImg(im,p=2) #  Is "p=2" different here than in the function definition? Can we remove "p=" here?
    return [im2]

def testDarkenImg2(im):
    im2 = darkenImg2(im,p=2) 
    return [im2]

def testBrightenImg(im):
    p=2
    im2=brightenImg(im,p)
    return [im2]

def testBrightenImg2(im):
    p=2
    im2=brightenImg2(im,p)
    return [im2]

def testcheckBoardImg(im):
    # cells size
    m = 5 
    n = 3
    im2=checkBoardImg(im, m, n)
    return [im2]

def testmultiHist(im):
    n = 2   # levels
    bins = 3
    im2, histograms=multiHist(im, n, bins)
    print('Histograms list: ', histograms)
    return [im2]


def saveImg(imfile, im2, test):
    dirname,basename = os.path.dirname(imfile), os.path.basename(imfile)
    fname, fext = os.path.splitext(basename)
    #print(dname,basename)
    pil_im = Image.fromarray(im2.astype(np.uint8))  # from array to Image
    pil_im.save(path_output+'//'+fname + suffixFiles[test] + fext)
    


path_input = './imgs-P1/'
path_output = './imgs-out-P1/'

bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*")
else:
    files = [path_input + 'peppers.ppm'] # iglesia,huesos

bAllTests = False
if bAllTests:
    tests = ['testDarkenImg2', 'testBrightenImg2', 'testcheckBoardImg', 'testmultiHist']
else:
    tests = ['testBrightenImg2']
    
nameTests = {'testHistEq': "Histogram equalization",
             'testBrightenImg': 'Brighten image',
             'testBrightenImg2': 'Brighten image 2',
             'testDarkenImg2': 'Darken image 2',
             'testDarkenImg': 'Darken image',
             'testcheckBoardImg': 'Board image',
             'testmultiHist': 'Multi histogram'}

suffixFiles = {'testHistEq': '_heq',
               'testBrightenImg': '_br',
               'testBrightenImg2': '_br2',
               'testDarkenImg2': '_dk2',
               'testDarkenImg': '_dk',
               'testcheckBoardImg': '_board',
               
               'testmultiHist': '_multihist'}

bSaveResultImgs = True


def doTests():
    print("Testing on", files)
    for imfile in files:
        im = np.array(Image.open(imfile))  # from Image to array
        
        #im = np.array(Image.open(imfile).convert('L'))  # from Image to array
        for test in tests:
            out = eval(test)(im)
            im2 = out[0]
            vpu.showImgsPlusHists(im, im2, title=nameTests[test])
            if len(out) > 1:
                vpu.showPlusInfo(out[1],"cumulative histogram" if test=="testHistEq" else None)
            if bSaveResultImgs:
                saveImg(imfile, im2, test)

if __name__== "__main__":
    doTests()

