# encoding: UTF-8
"""
Created on Jun 1, 2011
@author: Peter Harrington
数据集: https://archive.ics.uci.edu/ml/index.php
"""
from numpy import *
import matplotlib.pyplot as plt


def load_data_set(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return mat(datArr)


def pca(data_mat, top_N_feat=9999999):
    meanVals = mean(data_mat, axis=0)
    meanRemoved = data_mat - meanVals    # remove mean
    covMat = cov(meanRemoved, rowvar=False)
    eig_vals, eig_vectors = linalg.eig(mat(covMat))
    eigValInd = argsort(eig_vals)            # sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(top_N_feat + 1):-1]  # cut off unwanted dimensions
    red_eig_vectors = eig_vectors[:, eigValInd]       # reorganize eig vectors largest to smallest
    lowDDataMat = meanRemoved * red_eig_vectors     # transform data into new dimensions
    reconMat = (lowDDataMat * red_eig_vectors.T) + meanVals
    return lowDDataMat, reconMat


def replace_nan_with_mean():
    datMat = load_data_set('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])    # values that are not NaN (a number)
        datMat[nonzero(isnan(datMat[:, i].A))[0], i] = meanVal  # set NaN values to mean
    return datMat


def hello_pca():
    dataMat = load_data_set('testSet.txt')
    print(shape(dataMat))
    lowDMat, reconMat = pca(dataMat, 1)
    # lowDMat, reconMat = pca(dataMat, 2)
    print(shape(lowDMat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=90, c='red')
    plt.show()


def hello_secom():
    dataMat = replace_nan_with_mean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved, rowvar=False)
    eigVals, eig_vectors = linalg.eig(mat(covMat))
    print(eigVals)


if __name__ == '__main__':
    # hello_pca()
    hello_secom()
    print("Run PCA finish")
