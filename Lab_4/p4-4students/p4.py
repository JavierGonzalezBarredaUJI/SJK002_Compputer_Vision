from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import filters
from scipy import signal
import glob
import sys

from skimage import feature
from skimage.transform import hough_line, hough_line_peaks  # , probabilistic_hough_line
from skimage.draw import line


from scipy import ndimage as ndi
from copy import deepcopy

sys.path.append("../../p1/code")
import visualPercepUtils as vpu

bLecturerVersion=False
# try:
#     import p4e
#     bLecturerVersion=True
# except ImportError:
#     pass # file only available to lecturers

def plotSobel(im, filt_im):
    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Original image
    axes[0].imshow(im, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Sobel filter
    axes[1].imshow(filt_im, cmap='gray')
    axes[1].set_title('Sobel Filter')
    axes[1].axis('off')

    plt.show()
    
def sobel_convolution(im):
    # Sobel filter kernels for x and y directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Perform convolutions
    gx = signal.convolve2d(im, sobel_x, mode='same', boundary='symm', fillvalue=0)
    gy = signal.convolve2d(im, sobel_y, mode='same', boundary='symm', fillvalue=0)

    # Compute gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    return magnitude

def testSobel(im, params=None):
    gx = filters.sobel(im, 1)
    gy = filters.sobel(im, 0)
    magnitude = np.sqrt(gx**2 + gy**2)
    #magnitude = sobel_convolution(im)
    plotSobel(im, magnitude)
    return [magnitude]

def plotCanny(im, edge, sigma):
    # Display the images
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Original image
    axes[0].imshow(im, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Canny edge detection
    axes[1].imshow(edge, cmap='gray')
    axes[1].set_title(f'Canny (σ={sigma})')
    axes[1].axis('off')

    plt.show()

def testCanny(im, params=None):
    sigma = params['sigma']
    edge = feature.canny(im, sigma=sigma, low_threshold=0.2 * 255, high_threshold=0.25 * 255, use_quantiles=False)
    return [edge]


def testHough(im, params=None):
    edges = testCanny(im, params)[0]
    numThetas = 200
    H, thetas, rhos = hough_line(edges, np.linspace(-np.pi/2, np.pi/2, numThetas))
    print("# angles:", len(thetas))
    print("# distances:", len(rhos))
    print("rho[...]",rhos[:5],rhos[-5:])
    return [np.log(H+1), (H, thetas, rhos)] # log of Hough space for display purpose


def findPeaks(H, thetas, rhos, nPeaksMax=None):
    if nPeaksMax is None:
        nPeaksMax = np.inf
    return hough_line_peaks(H, thetas, rhos, num_peaks=nPeaksMax, threshold=0.15 * np.max(H), min_angle=20, min_distance=15)


def drawHoughLines(im, peaks, thetas, rhos):
    # Create a copy of the original image to draw lines on
    im_with_lines = np.copy(im)

    # Draw lines based on Hough peaks
    for peak_row, peak_col in zip(peaks[0], peaks[1]):
        theta = thetas[peak_col]
        rho = rhos[peak_row]

        # Calculate endpoint coordinates for the line
        y0 = int(rho / np.sin(theta))
        y1 = int((rho - im.shape[1] * np.cos(theta)) / np.sin(theta))
        x0 = 0
        x1 = im.shape[1]

        # Draw the line on the image
        rr, cc = line(y0, x0, y1, x1)
        im_with_lines[rr, cc] = 255  # Assuming white lines on a black background
        
    plotHF(im, im_with_lines)

    return im_with_lines


def plotHF(im, image_with_contours):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(image_with_contours, cmap='gray')
    plt.title('Image with Contours')
    plt.axis('off')

    plt.show()


# -----------------
# Test image files
# -----------------
path_input = './imgs-P4/'
path_output = './imgs-out-P4/'
bAllFiles = False
if bAllFiles:
    files = glob.glob(path_input + "*.p??")
else:
    files = [path_input + 'cuadros.png']  # cuadros, lena

# --------------------
# Tests to perform
# --------------------
bAllTests = False
if bAllTests:
    tests = ['testSobel', 'testCanny', 'testHough']
else:
    #tests = ['testSobel']
    #tests = ['testCanny']
    tests = ['testHough']

# -------------------------------------------------------------------
# Dictionary of user-friendly names for each function ("test") name
# -------------------------------------------------------------------

nameTests = {'testSobel': 'Detector de Sobel',
             'testCanny': 'Detector de Canny',
             'testHough': 'Transformada de Hough'}

bAddNoise = False
bRotate = True


def doTests():
    print("Testing on", files)
    nFiles = len(files)
    nFig = None
    alfa = [0, 15, 30, 45, 90]
    for test in tests:
        if test is "testSobel":
            params = {}
        elif test in ["testCanny", "testHough"]:
            params = {}
            params['sigma'] = 5  # 15
        if test is "testHough":
            pass  # params={}

        for i, imfile in enumerate(files):
            print("testing", test, "on", imfile)

            im_pil = Image.open(imfile).convert('L')
            im = np.array(im_pil)  # from Image to array
            
            for angle in alfa:
                if bRotate:
                    im = ndi.rotate(im, angle, mode='nearest')

                if bAddNoise:
                    im = im + np.random.normal(loc=0, scale=5, size=im.shape)

                outs_np = eval(test)(im, params)
                print("num ouputs", len(outs_np))
                if test is "testHough":
                    outs_np_plot = outs_np[0:1]
                else:
                    outs_np_plot = outs_np
                nFig = vpu.showInFigs([im] + outs_np_plot, title=nameTests[test], nFig=nFig, bDisplay=True)  # bDisplay=True for displaying *now* and waiting for user to close

                if test is "testHough":
                    H, thetas, rhos = outs_np[1]  # second output is not directly displayable
                    peaks_values, peaks_thetas, peaks_rhos = findPeaks(H, thetas, rhos, nPeaksMax=None)
                    #image_with_contours = drawHoughLines(im, peaks_values, thetas, rhos)
                    vpu.displayHoughPeaks(H, peaks_values, peaks_thetas, peaks_rhos, thetas, rhos)
                    if bLecturerVersion:
                        p4e.displayLines(im, peaks_thetas, peaks_rhos, peaks_values) # exercise
                        plt.show(block=True)
                    # displayLineSegments(...) # optional exercise

    plt.show(block=True)  # show pending plots (useful if we used bDisplay=False in vpu.showInFigs())


def add_noise(im):
    im = im + np.random.normal(loc=0, scale=5, size=im.shape)
    return im

def main():
    # Load images
    cuadros_path = path_input + 'cuadros.png'
    lena_path = path_input + 'lena.pgm'

    cuadros = np.array(Image.open(cuadros_path).convert('L'))
    lena = np.array(Image.open(lena_path).convert('L'))

    # Parameters
    sigma_values = [1, 2, 3]  # Adjust the values as needed

    for sigma in sigma_values:
        #cuadros = add_noise(cuadros)
        #lena = add_noise(lena)

        params = {'sigma': sigma}

        # Test Canny with cuadros
        edge_cuadros = testCanny(cuadros, params)[0]
        plotCanny(cuadros, edge_cuadros, sigma)

        # Test Canny with lena
        edge_lena = testCanny(lena, params)[0]
        plotCanny(lena, edge_lena, sigma)

if __name__ == "__main__":
    #main()
    doTests()
