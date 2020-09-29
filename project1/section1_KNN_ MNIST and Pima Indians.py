from numpy import *
import operator
# mnist
import struct
import array
import numpy
#https://github.com/sorki/python-mnist/blob/master/mnist/loader.py

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # print sortedClassCount
    return sortedClassCount[0][0]


def handwritingClassTest():
    trainingMat, hwLabels, size = load_mnist('D:/electrical/temp/MNIST_data/train-images-idx3-ubyte',
                                                 'D:/electrical/temp/MNIST_data/train-labels-idx1-ubyte')
    dataUnderTest, classNumStr, size = load_mnist('D:/electrical/temp\MNIST_data/t10k-images-idx3-ubyte',
                                                      'D:/electrical/temp/MNIST_data/t10k-labels-idx1-ubyte')
    errorCount = 0.0
    for i in range(size):
        classifierResult = classify0(dataUnderTest[i, :], trainingMat, hwLabels, 3)
        print("the NO.%d classifier came back with: %d, the real answer is: %d, error count is: %d" % (
        i, classifierResult, classNumStr[i], errorCount))
        if (classifierResult != classNumStr[i]): errorCount += + 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(size)))

# load mnist
def load_mnist(path_img, path_lbl):
    labels = []
    with open(path_lbl, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, expected 2049,'
                             'got {}'.format(magic))

        label_data = array.array("B", file.read())
        for i in range(size):
            labels.append(label_data[i])
    with open(path_img, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051,'
                             'got {}'.format(magic))
        image_data = array.array("B", file.read())
        images = numpy.zeros((size, rows * cols))

        for i in range(size):
            if ((i % 2000 == 0) or (i + 1 == size)):
                print("%d numbers imported" % (i))
            images[i, :] = image_data[i * rows * cols: (i + 1) * rows * cols]
    return images, labels, size

handwritingClassTest