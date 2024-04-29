# !/usr/bin/env python
# -*- coding: utf-8 -*-

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import glob
import sys

from skimage import measure

from skimage.morphology import disk, square, closing, opening # for the mathematically morphology part

sys.path.append("../../p1/code")
import visualPercepUtils as vpu

bStudentVersion=True
if not bStudentVersion:
    import p5e

def testOtsu(im, params=None):
    nbins = 256
    th = threshold_otsu(im)
    hist = np.histogram(im.flatten(), bins=nbins, range=[0, 255])[0]
    return [th, im > th, hist]  # threshold, binarized image (using such threshold), and image histogram


def fillGaps(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['closing-radius'])
    return [binIm, closing(binIm, sElem)]

# Don't worry about this function
def removeSmallRegions(im, params=None):
    binIm = im > threshold_otsu(im)
    sElem = disk(params['opening-radius'])
    return [binIm, opening(binIm, sElem)]

# Don't worry about this function
def fillGapsThenRemoveSmallRegions(im, params=None):
    binIm, closeIm = fillGaps(im, params)  # first, fill gaps
    sElem = disk(params['opening-radius'])
    return [binIm, opening(closeIm, sElem)]

def labelConnectedComponents(im, params=None):
    binIm = im > threshold_otsu(im, params)
    binImProc = fillGapsThenRemoveSmallRegions(im, params)[1]
    return [binIm, binImProc,
            measure.label(binIm, background=0), measure.label(binImProc, background=0)]
"""
def reportPropertiesRegions(labelIm,title):
    print("* * "+title)
    regions = measure.regionprops(labelIm)
    for r, region in enumerate(regions):  # enumerate() is often handy: it provides both the index and the element
        print("Region", r + 1, "(label", str(region.label) + ")")
        print("\t area: ", region.area)
        print("\t perimeter: ", round(region.perimeter, 1))  # show only one decimal place
"""

def plotCoins(label_im, total_money):
    # Display and report properties of the labeled image
    plt.figure(figsize=(8, 8))
    plt.plot()
    plt.imshow(label_im, cmap='viridis')
    plt.title('Total money: ' + str(total_money) + 'cents')

    plt.show()    

      
def reportPropertiesRegions(im, labelIm, title):
    print("* * " + title)
    regions = measure.regionprops(labelIm)
    coin_count = 0  # Counter for the number of coins
    euro_coin = 0
    cent_coin = 0
    
    # Create a new image with colors for circular and non-circular regions
    result_image = np.zeros_like(im, dtype=np.uint8)
    
    for r, region in enumerate(regions):
        print("Region", r + 1, "(label", str(region.label) + ")")
        print("\t area:", region.area)
        print("\t perimeter:", round(region.perimeter, 1))
        
        # Determine coin type based on circularity and diameter
        diameter_pixels = np.sqrt(region.area / np.pi) * 2  # Diameter in pixels
        diameter_inches = diameter_pixels / 50.0  # Convert pixels to inches
        diameter_mm = diameter_inches * 25.4  # Convert inches to mm
        
        print("\t diameter:", round(diameter_mm, 1))

        # Check circularity based on the ratio of perimeter to the square root of area
        circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)
        print("\t circularity:", round(circularity, 3))

        # If circularity is above the threshold, consider it a coin
        if circularity >= 0.9 and circularity <= 1.1:
            print("\t Coin detected!")
            coin_count += 1
            
            # Set color based on circularity
            #color = 255 #(255, 0, 0)  
            
            if 22 < diameter_mm < 24:  # Assuming 1 euro coin diameter = 23 mm
                euro_coin += 1
                color = 255

            elif 18.5 < diameter_mm < 20.5:  # Assuming 10 cents coin diameter = 19.5 mm
                cent_coin += 1
                color = 128
            
        else:
            color = 0 #(0, 255, 0)

        # Assign color to the region in the result image
        result_image[labelIm == region.label] = color
    
    total_money = euro_coin * 100 + cent_coin * 10
    plotCoins(result_image, total_money)
    print("Total coins detected:", coin_count)
    print("Total 1 euro coins detected:", euro_coin)
    print("Total 10 cents coins detected:", cent_coin)
    print("Total money:", total_money, "cents")
    
    


# -----------------
# Test image files
# -----------------
path_input = './imgs-P5/'
path_output = './imgs-out-P5/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'monedas.pgm']

# --------------------
# Tests to perform
# --------------------
bAllTests = True
if bAllTests:
    tests = ['testOtsu', 'labelConnectedComponents']
else:
    #tests = ['testOtsu']
    #tests = ['fillGaps']
    tests = ['labelConnectedComponents']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testOtsu': "thresholding with Otsu's method",
             'labelConnectedComponents': 'Labelling conected components'}

myThresh = 100  # use your own value here
diskSizeForClosing = 2  # don't worry about this
diskSizeForOpening = 5  # don't worry about this

def doTests():
    print("Testing ", tests, "on", files)
    nFiles = len(files)
    nFig = None
    for i, imfile in enumerate(files):
        im_pil = Image.open(imfile).convert('L')
        im = np.array(im_pil)  # from Image to array

        for test in tests:
            title = nameTests[test]
            print(imfile, test)
            if test is "testOtsu":
                params = {}
            elif test is "labelConnectedComponents":
                params = {}
                params['closing-radius'] = diskSizeForClosing
                params['opening-radius'] = diskSizeForOpening
                subtitles = ["original image", "binarized image", "Processed binary image", "Connected components", "Connected componentes from processed binary image"]

            outs_np = eval(test)(im, params)

            if test is "testOtsu":
                outs_np_plot = [outs_np[2]] + [outs_np[1]] + [im > myThresh]
                subtitles = ["original image", "Histogram", "Otsu with threshold=" + str(outs_np[0]),
                             "My threshold: " + str(myThresh)]
                m = n = 2
            else:
                outs_np_plot = outs_np
            print(len(outs_np_plot))
            vpu.showInGrid([im] + outs_np_plot, m=m, n=n, title=title, subtitles=subtitles)
            if test is 'labelConnectedComponents':
                plt.figure()
                labelImOriginalBinaryImage = outs_np_plot[2]
                labelImProcessedBinaryImage = outs_np_plot[3]
                vpu.showImWithColorMap(labelImOriginalBinaryImage,'jet') # the default color map, 'spectral', does not work in lab computers
                plt.show(block=True)
                titleForBinaryImg = "From unprocessed binary image"
                titleForProcesImg = "From filtered binary image"
                reportPropertiesRegions(im=im, labelIm=labelImOriginalBinaryImage,title=titleForBinaryImg)
                reportPropertiesRegions(im=im, labelIm=labelImProcessedBinaryImage,title=titleForProcesImg)

                if not bStudentVersion:
                    p5e.displayImageWithCoins(im,labelIm=labelImOriginalBinaryImage,title=titleForBinaryImg)
                    p5e.displayImageWithCoins(im,labelIm=labelImProcessedBinaryImage,title=titleForProcesImg)

    plt.show(block=True)  # show pending plots


if __name__ == "__main__":
    doTests()
