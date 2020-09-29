"""
Created'' on Feb 4, 2011
Tree-Based Regression Methods
@author: Peter Harrington
"""
from numpy import *


def load_data_set(fileName):
    """
    General function to parse tab -delimited floats
    :param fileName:
    :return:
    """
    # assume last column is target value
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # map all elements to float()
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def bin_split_data_set(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def reg_leaf(dataSet):
    # returns the value used for each leaf
    return mean(dataSet[:, -1])


def reg_err(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]


def linear_solve(dataSet):   # helper function used in two places
    dataSet = mat(dataSet)
    m, n = shape(dataSet)
    # create a copy of data with 1 in 0th position
    X = mat(ones((m, n)))
    Y = mat(ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n-1]
    Y = dataSet[:, -1]   # and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


def model_leaf(dataSet):
    # create linear model and return coefficient
    ws, X, Y = linear_solve(dataSet)
    return ws


def model_err(dataSet):
    ws, X, Y = linear_solve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat, 2))


def choose_best_split(dataSet, leafType=reg_leaf, errType=reg_err, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].tolist())) == 1:  # exit cond 1
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = bin_split_data_set(dataSet, featIndex, splitVal)
            if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if S - bestS < tolS:
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = bin_split_data_set(dataSet, bestIndex, bestValue)
    if shape(mat0)[0] < tolN or shape(mat1)[0] < tolN:  # exit cond 3
        return None, leafType(dataSet)
    # returns the best feature to split on and the value used for that split
    return bestIndex, bestValue


def create_tree(dataSet, leafType=reg_leaf, errType=reg_err, ops=(1, 4)):
    # assume dataSet is NumPy Mat so we can array filtering
    # choose the best split
    feat, val = choose_best_split(dataSet, leafType, errType, ops)
    # if the splitting hit a stop condition return val
    if feat is None:
        return val
    retTree = {'spInd': feat, 'spVal': val}
    lSet, rSet = bin_split_data_set(dataSet, feat, val)
    retTree['left'] = create_tree(lSet, leafType, errType, ops)
    retTree['right'] = create_tree(rSet, leafType, errType, ops)
    return retTree  


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['left']+tree['right'])/2.0


def prune(tree, testData):
    # if we have no test data collapse the tree
    if shape(testData)[0] == 0:
        return get_mean(tree)
    # if the branches are not trees try to prune them
    if is_tree(tree['right']) or is_tree(tree['left']):
        lSet, rSet = bin_split_data_set(testData, tree['spInd'], tree['spVal'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # if they are now both leaves, see if we can merge them
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        lSet, rSet = bin_split_data_set(testData, tree['spInd'], tree['spVal'])
        errorLeft = sum(power(lSet[:, -1] - tree['left'], 2))
        errorRight = sum(power(rSet[:, -1] - tree['right'], 2))
        errorNoMerge = errorLeft + errorRight
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge: 
            print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


def reg_tree_eval(model, inDat):
    return float(model)


def model_tree_eval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:, 1:n+1]=inDat
    return float(X*model)


def tree_fore_cast(tree, inData, modelEval=reg_tree_eval):
    if not is_tree(tree):
        return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if is_tree(tree['left']):
            return tree_fore_cast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if is_tree(tree['right']):
            return tree_fore_cast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def create_fore_cast(tree, testData, modelEval=reg_tree_eval):
    m = len(testData)
    yHat = mat(zeros((m, 1)))
    for i in range(m):
        yHat[i, 0] = tree_fore_cast(tree, mat(testData[i]), modelEval)
    return yHat


def hello_bin_split():
    testMat = mat(eye(4))
    print(testMat)
    my_mat0, my_mat1 = bin_split_data_set(testMat, 1, 0.5)
    print("mat0")
    print(my_mat0)
    print("mat1")
    print(my_mat1)


def hello_create_tree():
    myDat = load_data_set("ex00.txt")
    myMat = array(myDat)
    print(create_tree(myMat))
    myDat1 = load_data_set("ex0.txt")
    myMat1 = array(myDat1)
    print(create_tree(myMat1))
    # print(create_tree(myMat1, ops=(0, 1)))
    myDat2 = load_data_set("ex2.txt")
    myMat2 = array(myDat2)
    # print(create_tree(myMat2))
    # print(create_tree(myMat2, ops=(10000, 4)))
    myTree = create_tree(myMat2, ops=(0, 1))
    print(myTree)
    myDatTest = load_data_set("ex2test.txt")
    myMat2Test = array(myDatTest)
    print(prune(myTree, myMat2Test))
    myHat2 = array(load_data_set("exp2.txt"))
    print(create_tree(myHat2, model_leaf, model_err, (1, 10)))


def hello_tree_fore_cast():
    trainMat = array(load_data_set("bikeSpeedVsIq_train.txt"))
    testMat = array(load_data_set("bikeSpeedVsIq_test.txt"))
    myTree = create_tree(trainMat, ops=(1, 20))
    yHat = create_fore_cast(myTree, testMat[:, 0])
    print(corrcoef(yHat, testMat[:, 1], rowvar=False)[0, 1])
    myTree = create_tree(trainMat, model_leaf, model_err, ops=(1, 20))
    yHat = create_fore_cast(myTree, testMat[:, 0], model_tree_eval)
    print(corrcoef(yHat, testMat[:, 1], rowvar=False)[0, 1])
    ws, X, Y = linear_solve(trainMat)
    print(ws)
    for i in range(shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    print(corrcoef(yHat, testMat[:, 1], rowvar=False)[0, 1])


if __name__ == '__main__':
    # hello_bin_split()
    # hello_create_tree()
    hello_tree_fore_cast()
    print("Run regTress finish")
