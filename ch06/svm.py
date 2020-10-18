# coding=utf-8
"""
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
"""
from numpy import *


def load_data_set(file_name):
    data_mat = []
    label_mat = []
    fr = open(file_name)
    for line in fr.readlines():
        line_arr = line.strip().split('\t')
        data_mat.append([float(line_arr[0]), float(line_arr[1])])
        label_mat.append(float(line_arr[2]))
    return data_mat, label_mat


def select_j_rand(i, m):
    j = i
    # we want to select any j not equal to i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, high, low):
    if aj > high:
        aj = high
    if low > aj:
        aj = low
    return aj


def smo_simple(data_mat_in, class_labels, c, tolerance, max_iter):
    """
    SMO算法简化版本
    :param data_mat_in:
    :param class_labels:
    :param c:
    :param tolerance:
    :param max_iter:
    :return:
    """
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iterator = 0
    while iterator < max_iter:
        alpha_pairs_changed = 0
        for i in range(m):
            f_xi = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[i, :].T)) + b
            ei = f_xi - float(label_mat[i])  # if checks if an example violates KKT conditions
            if ((label_mat[i] * ei < -tolerance) and (alphas[i] < c)) \
                    or ((label_mat[i] * ei > tolerance) and (alphas[i] > 0)):
                j = select_j_rand(i, m)
                f_xj = float(multiply(alphas, label_mat).T * (data_matrix * data_matrix[j, :].T)) + b
                ej = f_xj - float(label_mat[j])
                alpha_i_old = alphas[i].copy()
                alpha_j_old = alphas[j].copy()
                if label_mat[i] != label_mat[j]:
                    low = max(0, alphas[j] - alphas[i])
                    high = min(c, c + alphas[j] - alphas[i])
                else:
                    low = max(0, alphas[j] + alphas[i] - c)
                    high = min(c, alphas[j] + alphas[i])
                if low == high:
                    print("low==high")
                    continue
                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T \
                    - data_matrix[i, :] * data_matrix[i, :].T \
                    - data_matrix[j, :] * data_matrix[j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                alphas[j] -= label_mat[j] * (ei - ej) / eta
                alphas[j] = clip_alpha(alphas[j], high, low)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print("j not moving enough")
                    continue
                # update i by the same amount as j, the update is in the opposite direction
                alphas[i] += label_mat[j] * label_mat[i] * (alpha_j_old - alphas[j])
                b1 = b - ei - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[i, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - ej - label_mat[i] * (alphas[i] - alpha_i_old) * data_matrix[i, :] * data_matrix[j, :].T - \
                     label_mat[j] * (alphas[j] - alpha_j_old) * data_matrix[j, :] * data_matrix[j, :].T
                if (0 < alphas[i]) and (c > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (c > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                alpha_pairs_changed += 1
                print("iter: %d i:%d, pairs changed %d" % (iterator, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            iterator += 1
        else:
            iterator = 0
        print("iteration number: %d" % iterator)
    return b, alphas


class optStruct:
    # Initialize the structure with the parameters
    def __init__(self, data_mat_in, class_labels, c, toler):
        self.X = data_mat_in
        self.labelMat = class_labels
        self.C = c
        self.tol = toler
        self.m = shape(data_mat_in)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag


def calc_ek(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJ(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if (len(validEcacheList)) > 1:
        # loop through valid Ecache values and find the one that maximizes delta E
        for k in validEcacheList:
            if k == i:
                continue  # don't calc for i, waste of time
            Ek = calc_ek(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = select_j_rand(i, oS.m)
        Ej = calc_ek(oS, j)
    return j, Ej


def updateEk(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calc_ek(oS, k)
    oS.eCache[k] = [1, Ek]


def innerL(i, oS):
    Ei = calc_ek(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (
            oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from selectJrand
        alpha_i_old = oS.alphas[i].copy()
        alpha_j_old = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            low = max(0, oS.alphas[j] - oS.alphas[i])
            high = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            low = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            high = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if low == high:
            print("low==high")
            return 0
        # changed for kernel
        eta = 2.0*oS.X[i, :]*oS.X[j, :].T - oS.X[i, :]*oS.X[i, :].T - oS.X[j, :]*oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], high, low)
        updateEk(oS, j)  # added this for the Ecache
        if abs(oS.alphas[j] - alpha_j_old) < 0.00001:
            print("j not moving enough")
            return 0
        # update i by the same amount as j
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alpha_j_old - oS.alphas[j])
        # added this for the Ecache, the update is in the opposite direction
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[i, :].T \
            - oS.labelMat[j] * (oS.alphas[j] - alpha_j_old) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[i, :].T \
            - oS.labelMat[j] * (oS.alphas[j] - alpha_j_old) * oS.X[i, :] * oS.X[j, :].T
        if 0 < oS.alphas[i] < oS.C:
            oS.b = b1
        elif 0 < oS.alphas[j] < oS.C:
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smo_p(data_mat_in, class_labels, C, toler, maxIter):
    """
    SMO算法完整版本
    :param data_mat_in: 
    :param class_labels: 
    :param C: 
    :param toler: 
    :param maxIter: 
    :return: 
    """
    oS = optStruct(mat(data_mat_in), mat(class_labels).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while iter < maxIter and (alphaPairsChanged > 0 or entireSet):
        alphaPairsChanged = 0
        if entireSet:  # go over all
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False  # toggle entire set loop
        elif alphaPairsChanged == 0:
            entireSet = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def calcWs(alphas, data_arr, class_labels):
    X = mat(data_arr)
    label_mat = mat(class_labels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_mat[i], X[i, :].T)
    return w


# calc the kernel or transform data to a higher dimensional space
def kernel_trans(X, A, k_tup):
    m, n = shape(X)
    K = mat(zeros((m, 1)))
    if k_tup[0] == 'lin':
        K = X * A.T  # linear kernel
    elif k_tup[0] == 'rbf':
        for j in range(m):
            delta_row = X[j, :] - A
            K[j] = delta_row * delta_row.T
        K = exp(K / (-1 * k_tup[1] ** 2))  # divide in NumPy is element-wise not matrix like Matlab
    else:
        raise NameError('Houston We Have a Problem -- That Kernel is not recognized')
    return K


class optStructK:
    # Initialize the structure with the parameters
    def __init__(self, data_mat_in, class_labels, C, toler, kTup):
        self.X = data_mat_in
        self.labelMat = class_labels
        self.C = C
        self.tol = toler
        self.m = shape(data_mat_in)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.eCache = mat(zeros((self.m, 2)))  # first column is valid flag
        self.K = mat(zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = kernel_trans(self.X, self.X[i, :], kTup)


def calcEkK(oS, k):
    fXk = float(multiply(oS.alphas, oS.labelMat).T * (oS.X * oS.X[k, :].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek


def selectJK(i, oS, Ei):  # this is the second choice -heurstic, and calcs Ej
    maxK = -1
    maxDeltaE = 0
    Ej = 0
    oS.eCache[i] = [1, Ei]  # set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:, 0].A)[0]
    if len(validEcacheList) > 1:
        for k in validEcacheList:  # loop through valid Ecache values and find the one that maximizes delta E
            if k == i:
                continue  # don't calc for i, waste of time
            Ek = calc_ek(oS, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeltaE:
                maxK = k
                maxDeltaE = deltaE
                Ej = Ek
        return maxK, Ej
    else:  # in this case (first time around) we don't have any valid eCache values
        j = select_j_rand(i, oS.m)
        Ej = calc_ek(oS, j)
    return j, Ej


def updateEkK(oS, k):  # after any alpha has changed update the new value in the cache
    Ek = calc_ek(oS, k)
    oS.eCache[k] = [1, Ek]


def innerLK(i, oS):
    Ei = calc_ek(oS, i)
    if (oS.labelMat[i] * Ei < -oS.tol and oS.alphas[i] < oS.C) or (
            oS.labelMat[i] * Ei > oS.tol and oS.alphas[i] > 0):
        j, Ej = selectJ(i, oS, Ei)  # this has been changed from select J rand
        alpha_i_old = oS.alphas[i].copy()
        alpha_j_old = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            low = max(0, oS.alphas[j] - oS.alphas[i])
            high = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            low = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            high = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if low == high:
            print("low==high")
            return 0
        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - oS.X[j, :] * oS.X[j, :].T
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j] * (Ei - Ej) / eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], high, low)
        updateEk(oS, j)  # added this for the Ecache
        if abs(oS.alphas[j] - alpha_j_old) < 0.00001:
            print("j not moving enough")
            return 0
        # update i by the same amount as j
        oS.alphas[i] += oS.labelMat[j] * oS.labelMat[i] * (alpha_j_old - oS.alphas[j])
        # added this for the Ecache, the update is in the opposite direction
        updateEk(oS, i)
        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[i, :].T \
            - oS.labelMat[j] * (oS.alphas[j] - alpha_j_old) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alpha_i_old) * oS.X[i, :] * oS.X[j, :].T \
            - oS.labelMat[j] * (oS.alphas[j] - alpha_j_old) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0


def smoPK(data_mat_in, class_labels, C, toler, maxIter, kTup):
    """
    使用核函数的完整SMO算法
    :param data_mat_in:
    :param class_labels:
    :param C:
    :param toler:
    :param maxIter:
    :param kTup:
    :return:
    """
    oS = optStructK(mat(data_mat_in), mat(class_labels).transpose(), C, toler, kTup)
    iter = 0
    entire_set = True
    alpha_pairs_changed = 0
    while iter < maxIter and (alpha_pairs_changed > 0 or entire_set):
        alpha_pairs_changed = 0
        if entire_set:  # go over all
            for i in range(oS.m):
                alpha_pairs_changed += innerLK(i, oS)
                print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
            iter += 1
        else:  # go over non-bound (railed) alphas
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alpha_pairs_changed += innerLK(i, oS)
                print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alpha_pairs_changed))
            iter += 1
        if entire_set:
            entire_set = False  # toggle entire set loop
        elif alpha_pairs_changed == 0:
            entire_set = True
        print("iteration number: %d" % iter)
    return oS.b, oS.alphas


def test_rbf(k1=1.3):
    data_arr, label_arr = load_data_set('testSetRBF.txt')
    # c=200 important
    b, alphas = smoPK(data_arr, label_arr, 200, 0.0001, 10000, ('rbf', k1))
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = data_mat[svInd]  # get matrix of only support vectors
    labelSV = label_mat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(data_mat)
    error_count = 0
    for i in range(m):
        kernel_eval = kernel_trans(sVs, data_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(label_arr[i]):
            error_count += 1
    print("the training error rate is: %f" % (float(error_count) / m))
    data_arr, label_arr = load_data_set('testSetRBF2.txt')
    error_count = 0
    data_mat = mat(data_arr)
    label_mat = mat(label_arr).transpose()
    m, n = shape(data_mat)
    for i in range(m):
        kernel_eval = kernel_trans(sVs, data_mat[i, :], ('rbf', k1))
        predict = kernel_eval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(label_arr[i]): error_count += 1
    print("the test error rate is: %f" % (float(error_count) / m))


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)  # load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i, :] = img2vector('%s/%s' % (dirName, fileNameStr))
    return trainingMat, hwLabels


def testDigits(kTup=('rbf', 10)):
    dataArr, labelArr = loadImages('trainingDigits')
    b, alphas = smo_p(dataArr, labelArr, 200, 0.0001, 10000)
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    svInd = nonzero(alphas.A > 0)[0]
    sVs = datMat[svInd]
    labelSV = labelMat[svInd]
    print("there are %d Support Vectors" % shape(sVs)[0])
    m, n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernel_trans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f" % (float(errorCount) / m))
    dataArr, labelArr = loadImages('testDigits')
    errorCount = 0
    datMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(datMat)
    for i in range(m):
        kernelEval = kernel_trans(sVs, datMat[i, :], kTup)
        predict = kernelEval.T * multiply(labelSV, alphas[svInd]) + b
        if sign(predict) != sign(labelArr[i]): errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))


if __name__ == '__main__':
    # data_arr, labelArr = load_data_set("testSet.txt")
    # print(labelArr)
    # # my_b, my_alphas = smo_simple(data_arr, labelArr, 0.6, 0.001, 40)
    # my_b, my_alphas = smo_p(data_arr, labelArr, 0.6, 0.001, 40)
    # print("b:{}".format(my_b))
    # print("alphas shape:{}".format(shape(my_alphas)))
    # print("alphas>0:{}".format(my_alphas[my_alphas > 0]))
    # for index in range(100):
    #     if my_alphas[index] > 0.0:
    #         print(my_alphas[index], data_arr[index], labelArr[index])
    # ws = calcWs(my_alphas, data_arr, labelArr)
    # print(ws)
    # data_mat = mat(data_arr)
    # print(data_mat[0] * mat(ws) + my_b)
    # print(labelArr[0])
    test_rbf()
    print("Run svm finish")
