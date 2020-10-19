"""
Created on Mar 8, 2011
@author: Peter
"""
from numpy import *
from numpy import linalg as la


def load_ex_data():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]


def load_ex_data2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def euclid_sim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))


def pears_sim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar=False)[0][1]


def cos_sim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)


def stand_est(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = nonzero(logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


def svd_est(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    # arrange Sig4 into a diagonal matrix
    Sig4 = mat(eye(4) * Sigma[:4])
    # create transformed items
    x_formed_items = dataMat.T * U[:, :4] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(x_formed_items[item, :].T, x_formed_items[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal


def recommend(dataMat, user, N=3, simMeas=cos_sim, estMethod=stand_est):
    # find unrated items
    unratedItems = nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def print_mat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1,)
            else:
                print(0,)
        print('')


def img_compress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print("****original matrix******")
    print_mat(myMat, thresh)
    U, Sigma, VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    # construct diagonal matrix from vector
    for k in range(numSV):
        SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon*VT[:numSV, :]
    print("****reconstructed matrix using %d singular values******" % numSV)
    print_mat(reconMat, thresh)


def hello_svd():
    U, Sigma, VT = la.svd([[1, 1], [7, 7]])
    print(U)
    print(Sigma)
    print(VT)
    Data = load_ex_data()
    U, Sigma, VT = la.svd(Data)
    print(Sigma)
    Sig3 = mat([[Sigma[0], 0, 0], [0, Sigma[1], 0], [0, 0, Sigma[2]]])
    print(U[:, :3] * Sig3 * VT[:3, :])


def hello_sim():
    myMat = mat(load_ex_data())
    print("euclid_sim")
    print(euclid_sim(myMat[:, 0], myMat[:, 4]))
    print(euclid_sim(myMat[:, 0], myMat[:, 0]))
    print("cos_sim")
    print(cos_sim(myMat[:, 0], myMat[:, 4]))
    print(cos_sim(myMat[:, 0], myMat[:, 0]))
    print("pears_sim")
    print(pears_sim(myMat[:, 0], myMat[:, 4]))


if __name__ == '__main__':
    # hello_svd()
    hello_sim()
    print("Run svd finish")
