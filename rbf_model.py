#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 15:03:49 2018

@author: yi-chun
"""

import math
from numpy import *
import random


###### K-Means classification ########
inf=open('vowel-train.tab','r')  #input: dataSet
inputData=inf.read()
inf.close



dataSet_float=map(float, inputData) # string turns to float type
dataSetArray =  np.array(dataSet_float) # turns to array
n_inputData, Dim= dataSetArray.shape

for i in range(n_inputData-3):
    for j in range(Dim-3):
        dataSet=dataSetArray[i+3][j+3] 
    

for i in range(n_inputData-3):
    Output=dataSetArray[i+3][2] 



k=dataSet.shape[0]/2 #initial assumption of k, k is the number of centroids

# calculate Euclidean distance
def euclDistance(vector1, vector2):
	return sqrt(sum(power(vector2 - vector1, 2)))
 
# init centroids with random samples
def initCentroids(dataSet, k):
    # choose k numbers of data
	numSamples, dim = dataSet.shape
	centroids = zeros((k, dim))
	for i in range(k):
		index = int(random.uniform(0, numSamples))
		centroids[i, :] = dataSet[index, :] # choose data randomly
	return centroids
 
# k-means cluster
def kmeans(dataSet, k):
    # k: numbers of centers
    numSamples = dataSet.shape[0]
	
    clusterAssment = mat(zeros((numSamples, 2)))
    # first column stores which cluster this sample belongs to,
	# second column stores the error between this sample and its centroid
    clusterChanged = True
 
	## step 1: init centroids
    centroids = initCentroids(dataSet, k)
 
    while clusterChanged:
            clusterChanged = False
            ## for each sample
            for i in range(numSamples):
                minDist  = 100000.0
                minIndex = 0 #initalize
			## for each centroid
			## step 2: find the centroid who is closest
            for j in range(k):
                distance = euclDistance(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist  = distance
                    minIndex = j
			
			## step 3: update its cluster
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist**2
 
            ## step 4: update centroids
            for j in range(k):
                pointsInCluster = dataSet[nonzero(clusterAssment[:, 0].A == j)[0]]
                centroids[j, :] = mean(pointsInCluster, axis = 0)
                M[j]=pointsInCluster.shape
            
	
    return centroids, clusterAssment, M

centroids,clusterAssment,M = kmeans(dataSet, k)


###### Radial Basis Function ######

def Sigma(clusterAssment,M):
    sig=mat(zeros((k, 1)))
    for j in range(k):
        sig[j]=sum(clusterAssment[nonzero(clusterAssment[:,0].A==j)[1]])/M[j]
    return sig

sig=Sigma(clusterAssment,M)

def Gaussion(dataSet,sig,centroids):
    for i in range(dataSet.shape[0]):
        fi[i]=math.exp(-((dataSet[i,0]-centroids[i, 0])**2)/(2*(sig[i]**2)))
    return fi

fi=Gaussion(dataSet,sig,centroids)



####### LMS #########

# single layer

m=k# initial assumption of m
n=dataSet.shape[0]

def iniWeight(n,m):
    # m numbers of units
    # n numbers of data
    for i in range(n):
        for j in range(m):
            Weight[n][m]=random.random()
    return Weight

Weight=iniWeight(n,m)





s=0 # initialize
S=0
S_E=0
rate=0.001
def LMS(Weight,dataSet,Output,n,m):
    
    for i in range(n):
        for j in range(m):
            s=s+Weight*fi
    S=S+(Ouput[i,0]-s)*fi #rate*(sum(y-sum(w*R(x))*R(x)))
    S_E=S_E+(Ouput[i,0]-s)**2
    return rate*S, 0.5*S_E

delta_w, E = LMS(Weight,dataSet,Output,n,m)

while E > 1:
    Weight=Weight+delta_w
    delta_w, E = LMS(Weight,dataSet,Output,n,m)
    
print(Weight)












