"""
Created on Nov 28, 2010
Adaboost is short for Adaptive Boosting
@author: Peter
"""
from numpy import array, ones, shape, mat, inf, zeros, multiply, exp, sign
from math import log
import matplotlib.pyplot as plt


def load_simple_data():
    dat_mat = array([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.]
    ])
    class_labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dat_mat, class_labels


# general function to parse tab -delimited floats
def load_data_set(file_name):
    # get number of fields
    num_feat = len(open(file_name).readline().split('\t'))
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = []
        cur_line = line.strip().split('\t')
        for i in range(num_feat-1):
            line_arr.append(float(cur_line[i]))
        data_mat.append(line_arr)
        label_mat.append(float(cur_line[-1]))
    return data_mat, label_mat


# just classify the data
def stump_classify(data_matrix, dimension, thresh_val, thresh_inequal):
    ret_array = ones((shape(data_matrix)[0], 1))
    if thresh_inequal == 'lt':
        ret_array[data_matrix[:, dimension] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimension] > thresh_val] = -1.0
    return ret_array
    

def build_stump(dataArr, class_labels, D):
    data_matrix = mat(dataArr)
    label_mat = mat(class_labels).T
    m, n = shape(data_matrix)
    num_steps = 10.0
    best_stump = {}
    best_class_est = mat(zeros((m, 1)))
    # init error sum to +infinity
    min_error = inf
    for i in range(n):  # loop over all dimensions
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        step_size = (range_max-range_min)/num_steps
        # loop over all range in current dimension
        for j in range(-1, int(num_steps)+1):
            for inequal in ['lt', 'gt']:    # go over less than and greater than
                thresh_val = range_min + float(j) * step_size
                # call stump classify with i, j, lessThan
                predicted_vals = stump_classify(data_matrix, i, thresh_val, inequal)
                err_arr = mat(ones((m, 1)))
                err_arr[predicted_vals == label_mat] = 0
                weighted_error = D.T*err_arr  # calc total error multiplied by D
                # print("split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f"
                #       % (i, thresh_val, inequal, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = inequal
    return best_stump, min_error, best_class_est


def ada_boost_train_ds(data_arr, class_labels, num_it=40):
    """
    使用单层决策树作为弱分类器的AdaBoost算法
    :param data_arr:
    :param class_labels:
    :param num_it: 迭代次数
    :return:
    """
    weak_class_arr = []
    m = shape(data_arr)[0]
    D = mat(ones((m, 1))/m)   # init D to all equal
    agg_class_est = mat(zeros((m, 1)))
    for i in range(num_it):
        best_stump, error, class_est = build_stump(data_arr, class_labels, D)   # build Stump
        # print("D:", D.T)
        # calc alpha, throw in max(error,eps) to account for error=0
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        best_stump['alpha'] = alpha
        # store Stump Params in Array
        weak_class_arr.append(best_stump)
        # print("class_est: ", class_est.T)
        # exponent for D calc, getting messy
        exponent = multiply(-1 * alpha * mat(class_labels).T, class_est)
        # Calc New D for next iteration
        D = multiply(D, exp(exponent))
        D = D/D.sum()
        # calc training error of all classifiers, if this is 0 quit for loop early (use break)
        agg_class_est += alpha*class_est
        # print("agg_class_est: ", agg_class_est.T)
        agg_errors = multiply(sign(agg_class_est) != mat(class_labels).T, ones((m, 1)))
        error_rate = agg_errors.sum()/m
        # print("total error: ", error_rate)
        if error_rate == 0.0:
            break
    return weak_class_arr, agg_class_est


def ada_classify(datToClass, classifierArr):
    # do stuff similar to last aggClassEst in ada_boost_train_ds
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        # call stump classify
        classEst = stump_classify(
            dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print(aggClassEst)
    return sign(aggClassEst)


def plot_roc(predStrengths, classLabels):
    # cursor
    cur = (1.0, 1.0)
    # variable to calculate AUC
    ySum = 0.0
    numPosClass = sum(array(classLabels) == 1.0)
    yStep = 1/float(numPosClass)
    xStep = 1/float(len(classLabels)-numPosClass)
    # get sorted index, it's reverse
    sortedIndices = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    # loop through all the values, drawing a line segment at each point
    for index in sortedIndices.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        # draw line from cur to (cur[0]-delX,cur[1]-delY)
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)


def test_simple_data():
    my_data_mat, my_class_labels = load_simple_data()
    print("my_class_labels")
    print(my_class_labels)
    my_d = mat(ones((5, 1)) / 5)
    # print(build_stump(my_data_mat, my_class_labels, my_d))
    classifierArray = ada_boost_train_ds(my_data_mat, my_class_labels, 9)
    print("classifierArray")
    print(classifierArray)
    print(ada_classify([0, 0], classifierArray))
    print(ada_classify([[5, 5], [0, 0]], classifierArray))


def test_horse_data():
    datArr, labelArr = load_data_set("horseColicTraining2.txt")
    testArr, testLabelArr = load_data_set("horseColicTest2.txt")
    test_size = len(testLabelArr)
    classifier_count = 50
    classifierArray = ada_boost_train_ds(datArr, labelArr, classifier_count)
    prediction10 = ada_classify(testArr, classifierArray)
    errArr = mat(ones((test_size, 1)))
    errCount = errArr[prediction10 != mat(testLabelArr).T].sum()
    print("classifier count:{}, error count:{}, error rate:{:.2f}".format(
        classifier_count, errCount, errCount/test_size))


if __name__ == '__main__':
    # test_simple_data()
    # test_horse_data()
    datArr, labelArr = load_data_set("horseColicTraining2.txt")
    classifierArray, aggClassEst = ada_boost_train_ds(datArr, labelArr, 10)
    plot_roc(aggClassEst.T, labelArr)
    print("Run adaboost finish")
