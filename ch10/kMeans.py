# encoding: UTF-8
"""
Created on Feb 16, 2011
k Means Clustering for Ch10 of Machine Learning in Action
@author: Peter Harrington
"""
from numpy import *
import matplotlib.pyplot as plt


def load_data_set(file_name):      # general function to parse tab -delimited floats
    data_mat = []                    # assume last column is target value
    fr = open(file_name)
    for line in fr.readlines():
        cur_line = line.strip().split('\t')
        flt_line = list(map(float, cur_line)) # map all elements to float()
        data_mat.append(flt_line)
    return data_mat


def dist_euler(vec_a, vec_b):
    return sqrt(sum(power(vec_a - vec_b, 2)))     # la.norm(vecA-vecB)


def rand_cent(data_set, k):
    n = shape(data_set)[1]
    centroids = mat(zeros((k, n)))   # create centroid mat
    for j in range(n):              # create random cluster centers, within bounds of each dimension
        min_j = min(data_set[:, j])
        max_j = max(data_set[:, j])
        range_j = float(max_j - min_j)
        centroids[:, j] = mat(min_j + range_j * random.rand(k, 1))
    return centroids


def k_means(data_set, k, dist_meas=dist_euler, create_cent=rand_cent):
    m = shape(data_set)[0]
    cluster_assment = mat(zeros((m,2)))
    # create mat to assign data points to a centroid, also holds SE of each point
    centroids = create_cent(data_set, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):  # for each data point assign it to the closest centroid
            minDist = inf;
            minIndex = -1
            for j in range(k):
                distJI = dist_meas(centroids[j, :], data_set[i, :])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if cluster_assment[i,0] != minIndex: cluster_changed = True
            cluster_assment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):   # recalculate centroids
            # get all the point in this cluster
            ptsInClust = data_set[nonzero(cluster_assment[:, 0].A == cent)[0]]
            centroids[cent,:] = mean(ptsInClust, axis=0)    # assign centroid to mean
    return centroids, cluster_assment


def biKmeans(dataSet, k, distMeas=dist_euler):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2)))
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #create a list with one centroid
    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            centroidMat, splitClustAss = k_means(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:,1])#compare the SSE to the currrent minimum
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids 
        centList.append(bestNewCents[1,:].tolist()[0])
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment

import urllib
import json
def geoGrab(stAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'  #create a dict and constants for the goecoder
    params = {}
    params['flags'] = 'J'#JSON return type
    params['appid'] = 'aaa0VN6k'
    params['location'] = '%s %s' % (stAddress, city)
    url_params = urllib.urlencode(params)
    yahooApi = apiStem + url_params      #print url_params
    print(yahooApi)
    c=urllib.urlopen(yahooApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt', 'w')
    for line in open(fileName).readlines():
        line = line.strip()
        lineArr = line.split('\t')
        retDict = geoGrab(lineArr[1], lineArr[2])
        if retDict['ResultSet']['Error'] == 0:
            lat = float(retDict['ResultSet']['Results'][0]['latitude'])
            lng = float(retDict['ResultSet']['Results'][0]['longitude'])
            print("%s\t%f\t%f" % (lineArr[0], lat, lng))
            fw.write('%s\t%f\t%f\n' % (line, lat, lng))
        else:
            print("error fetching")
        sleep(1)
    fw.close()


def distSLC(vecA, vecB):    # Spherical Law of Cosines
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0     # pi is imported with numpy


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1,0.1,0.8,0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0=fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    print("ch10")
    datMat = mat(load_data_set('testSet.txt'))
    print(min(datMat[:, 0]))
    print(min(datMat[:, 1]))
    print(max(datMat[:, 0]))
    print(max(datMat[:, 1]))
    print(rand_cent(datMat, 2))
    print(dist_euler(datMat[0], datMat[1]))
