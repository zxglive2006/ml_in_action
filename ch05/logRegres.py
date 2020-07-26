"""
Created on Oct 27, 2010
Logistic Regression Working Module
@author: Peter
"""
from numpy import *
import matplotlib.pyplot as plt


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
        error = _label_mat - h                           # vector subtraction
        _weights = _weights + alpha * data_matrix.transpose() * error   # matrix multiply
    return _weights


def plot_best_fit(arr_weights):
    _data_mat, _label_mat = load_data_set()
    _data_arr = array(_data_mat)
    n = shape(_data_arr)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(_label_mat[i]) == 1:
            x_cord1.append(_data_arr[i, 1])
            y_cord1.append(_data_arr[i, 2])
        else:
            x_cord2.append(_data_arr[i,1])
            y_cord2.append(_data_arr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, c='red', marker='s')
    ax.scatter(x_cord2, y_cord2, s=30, c='green')
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


def stoc_grad_ascent1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    _weights = ones(n)       # initialize to all ones
    for j in range(num_iter):
        data_index = [x for x in range(m)]
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001    # alpha decreases with iteration, does not
            rand_index = int(random.uniform(0, len(data_index)))   # go to 0 because of the constant
            h = sigmoid(sum(data_matrix[rand_index] * _weights))
            error = class_labels[rand_index] - h
            _weights = _weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return _weights


def classify_vector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('horseColicTraining.txt')
    fr_test = open('horseColicTest.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr =[]
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    train_weights = stoc_grad_ascent1(array(training_set), training_labels, 1000)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr =[]
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(array(line_arr), train_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = float(error_count) / num_test_vec
    print("the error rate of this test is: %f" % error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("after %d iterations the average error rate is: %f" % (num_tests, error_sum/float(num_tests)))


if __name__ == '__main__':
    data_arr, label_mat = load_data_set()
    # weights = grad_ascent(data_arr, label_mat).getA()
    # weights = stoc_grad_ascent0(array(data_arr), label_mat)
    # weights = stoc_grad_ascent1(array(data_arr), label_mat)
    # print(weights)
    # plot_best_fit(weights)
    multi_test()
    print("Run Logistic finish")
