# coding=utf-8
"""
kNN: k Nearest Neighbors
Input:      in_x: vector to compare to existing data set (1xN)
            data_set: size m data set of known vectors (NxM)
            _labels: data set _labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
Output:     the most popular class label
"""
from numpy import *
import operator
from os import listdir
import matplotlib.pyplot as plt


def create_data_set():
    """
    创建数据集和标签
    :return:
    """
    _group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    _labels = ['A', 'A', 'B', 'B']
    return _group, _labels


def classify0(in_x, data_set, _labels, k):
    """
    k-近邻分类
    :param in_x: 用于分类的输入向量
    :param data_set: 输入的训练样本集
    :param _labels: 标签向量
    :param k: 用于选择最近邻居的数目
    :return: 发生频率最高的元素标签
    """
    data_set_size = data_set.shape[0]
    diff_mat = tile(in_x, (data_set_size, 1)) - data_set
    sq_diff_mat = diff_mat**2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances**0.5
    sorted_dist_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        vote_i_label = _labels[sorted_dist_indices[i]]
        class_count[vote_i_label] = class_count.get(vote_i_label, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    """
    将文本记录转换为Numpy的解析程序
    :param filename:
    :return:
    """
    love_dictionary = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    array_of_lines = fr.readlines()
    number_of_lines = len(array_of_lines)           # get the number of lines in the file
    return_mat = zeros((number_of_lines, 3))        # prepare matrix to return
    class_label_vector = []                         # prepare _labels return
    index = 0
    for line in array_of_lines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        if list_from_line[-1].isdigit():
            class_label_vector.append(int(list_from_line[-1]))
        else:
            class_label_vector.append(love_dictionary.get(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector

    
def auto_norm(_data_set):
    _min_vals = _data_set.min(0)
    _max_vals = _data_set.max(0)
    _ranges = _max_vals - _min_vals
    m = _data_set.shape[0]
    norm_data_set = _data_set - tile(_min_vals, (m, 1))
    norm_data_set = norm_data_set / tile(_ranges, (m, 1))   # element wise divide
    return norm_data_set, _ranges, _min_vals


def dating_class_test():
    ho_ratio = 0.10      # hold out 10%
    _dating_data_mat, _dating_labels = file2matrix('datingTestSet.txt')       # load data set from file
    norm_mat, ranges, min_vals = auto_norm(_dating_data_mat)
    m = norm_mat.shape[0]
    num_test_vecs = int(m * ho_ratio)
    error_count = 0.0
    for i in range(num_test_vecs):
        classifier_result = classify0(
            norm_mat[i, :], norm_mat[num_test_vecs:m, :], _dating_labels[num_test_vecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifier_result, _dating_labels[i]))
        if classifier_result != _dating_labels[i]:
            error_count += 1.0
    print("the total error rate is: %f" % (error_count / float(num_test_vecs)))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input("percentage of time spent playing video games?"))
    ff_miles = float(input("frequent flier miles earned per year?"))
    ice_cream = float(input("liters of ice cream consumed per year?"))
    dating_data_mat, dating_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(dating_data_mat)
    inArr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((inArr - min_vals)/ranges, norm_mat, dating_labels, 3)
    print("You will probably like this person: %s" % result_list[classifier_result - 1])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d"
              % (classifierResult, classNumStr))
        if classifierResult != classNumStr:
            errorCount += 1.0
    print("\n")
    print("the total number of errors is: %d" % errorCount)
    print("the total error rate is: %f" % (errorCount/float(mTest)))


if __name__ == '__main__':
    # group, labels = create_data_set()
    # print(group)
    # print(labels)
    # print(classify0([0, 0], group, labels, 3))
    # dating_data_mat, dating_labels = file2matrix(r"datingTestSet.txt")
    # print(dating_data_mat)
    # print(dating_labels[0:20])
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(dating_data_mat[:, 0], dating_data_mat[:, 1],
    #            15.0*array(dating_labels), 15.0*array(dating_labels))
    # plt.show()
    dating_class_test()
    print("Run kNN finish")
