"""
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
"""

from numpy import *


def load_data_set():
    _data_mat = []
    _label_mat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        _data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        _label_mat.append(int(line_arr[2]))
    return _data_mat, _label_mat


def sigmoid(in_x):
    return 1.0/(1 + exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)                      # convert to NumPy matrix
    _label_mat = mat(class_labels).transpose()          # convert to NumPy matrix
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    _weights = ones((n, 1))
    for k in range(max_cycles):                          # heavy on matrix operations
        h = sigmoid(data_matrix * _weights)              # matrix multiply
        error = (_label_mat - h)                         # vector subtraction
        _weights = _weights + alpha * data_matrix.transpose() * error   # matrix multiply
    return _weights


def plot_best_fit(arr_weights):
    import matplotlib.pyplot as plt
    _data_mat, _label_mat = load_data_set()
    _data_arr = array(_data_mat)
    n = shape(_data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(_label_mat[i]) == 1:
            xcord1.append(_data_arr[i, 1])
            ycord1.append(_data_arr[i, 2])
        else:
            xcord2.append(_data_arr[i,1])
            ycord2.append(_data_arr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-arr_weights[0] - arr_weights[1] * x) / arr_weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def stoc_grad_ascent0(data_matrix, class_labels):
    m, n = shape(data_matrix)
    alpha = 0.01
    _weights = ones(n)   # initialize to all ones
    for i in range(m):
        h = sigmoid(sum(data_matrix[i] * _weights))
        error = class_labels[i] - h
        _weights = _weights + alpha * error * data_matrix[i]
    return _weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)       # initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    # apha decreases with iteration, does not
            randIndex = int(random.uniform(0, len(dataIndex)))   # go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests)))


if __name__ == '__main__':
    data_arr, label_mat = load_data_set()
    # weights = grad_ascent(data_arr, label_mat).getA()
    weights = stoc_grad_ascent0(array(data_arr), label_mat)
    print(type(weights))
    print(weights)
    plot_best_fit(weights)
    print("Run Logistic finish")
