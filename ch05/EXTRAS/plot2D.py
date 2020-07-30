"""
Created on Oct 6, 2010
@author: Peter
"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ch05.logRegres as logRegres


def load_data_set():
    _data_mat = []
    _label_mat = []
    fr = open(r'..\testSet.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        _data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        _label_mat.append(int(line_arr[2]))
    return _data_mat, _label_mat


dataMat, labelMat = load_data_set()
dataArr = array(dataMat)
weights = logRegres.stoc_grad_ascent0(dataArr, labelMat)

n = shape(dataArr)[0]       # number of points to create
xcord1 = []
ycord1 = []
xcord2 = []
ycord2 = []

markers =[]
colors =[]
for i in range(n):
    if int(labelMat[i])== 1:
        xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
    else:
        xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])

fig = plt.figure()
ax = fig.add_subplot(111)
# ax.scatter(xcord,ycord, c=colors, s=markers)
type1 = ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
type2 = ax.scatter(xcord2, ycord2, s=30, c='green')
x = arange(-3.0, 3.0, 0.1)
#arr_weights = [-2.9, 0.72, 1.29]
#arr_weights = [-5, 1.09, 1.42]
# weights = [13.03822793,   1.32877317,  -1.96702074]
weights = [4.12,   0.48,  -0.6168]
y = (-weights[0]-weights[1]*x)/weights[2]
type3 = ax.plot(x, y)
# ax.legend([type1, type2, type3], ["Did Not Like", "Liked in Small Doses", "Liked in Large Doses"], loc=2)
# ax.axis([-5000,100000,-2,25])
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()