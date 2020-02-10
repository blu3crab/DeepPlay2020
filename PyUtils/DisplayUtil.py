###############################################################################
__author__ = "M.A.Tucker"
__license__ = "Apache License 2.0"
__version__ = "0.0.1"
#
# import
#   from PyUtils.DisplayUtil import displayGrayImageAtIndex
# invoke
#   displayGrayImageAtIndex(x_train, 1)
###############################################################################

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# purpose
#   display MNIST like image from dataset at index
# import
#   from PyUtils.DisplayUtil import displayGrayImageAtIndex
# invoke
#   displayGrayImageAtIndex(x_train, 1)
#
def displayGrayImageAtIndex(imageData, imageIndex):
    image = imageData[imageIndex]
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()

def displayColorImageAtIndex(imageData, imageIndex):
    plt.figure()
    plt.imshow(imageData[imageIndex])
    plt.colorbar()
    plt.grid(True)
    plt.show()

def displayImageLabelRange(imageData, imageLabels, classNames, startIndex, endIndex):
    plt.figure(figsize=(10, 10))
    for i in range(startIndex, endIndex):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imageData[i], cmap=plt.cm.binary)
        plt.xlabel(classNames[imageLabels[i]])
    plt.show()