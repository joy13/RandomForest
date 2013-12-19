import numpy as np;
import csv;
import math,random,sys,bisect,time
import numpy,scipy.spatial.distance
from random import randint
from numpy.random import shuffle

class RandomForestLearner(object):
    
    def __init__(self, k, isBagging):
        self.data = None
        self.tree = None
        self.count = 0
        self.k = k
        self.isBagging = isBagging
        self.trees = None
    
    def buildTree(self,data):
        parent = self.count
        x = data
        xrow = len(x) - 1
        xcol = len(x[0]) - 1
        if (xrow+1) == 1:
            self.tree[self.count,:] = [-1, x[0,xcol],-1,-1]
        else:
            left = np.zeros((xrow+1,xcol+1))
            right = np.zeros((xrow+1,xcol+1))
            feature = randint(0,xcol-1)
            flag = False
            sum = 0
            for i in range(0,len(x)-1):
                if(x[i,feature] == x[i+1,feature]):
                    flag = True
                else:
                    flag = False
            if flag == True:
                mean = np.mean(x[:,-1])
                self.tree[self.count,:] = [-1,mean,-1,-1]
            else:
                val1 = x[randint(0,xrow),feature]
                val2 = x[randint(0,xrow),feature]
                splitVal = (val1+val2)/2
                self.tree[self.count,:] = [feature,splitVal,self.count+1,-1]
                l = 0
                r = 0
                lcount = 0
                rcount = 0
                for i in range(0,len(x)):
                    if x[i,feature] <= splitVal:
                        left[l,:] = x[i,:]
                        l += 1
                        lcount += 1
                    else:
                        right[r,:] = x[i,:]
                        r += 1
                        rcount += 1
                self.count += 1 
                self.buildTree(left[0:lcount])
                if(rcount >= 1):
                    self.count += 1
                    rtree = self.count
                    self.buildTree(right[0:rcount])
                    self.tree[parent,-1] = rtree

    def addEvidence(self,dataX, dataY):
        if not dataY == None:
            data = np.zeros([dataX.shape[0],dataX.shape[1]+1])
            data[:,0:dataX.shape[1]]=dataX
            data[:,(dataX.shape[1])]=dataY
        else:
            data = dataX
        self.data = data
        self.trees = []
        k = self.k
        for i in range (0, k):
            if self.isBagging:
                self.count = 0
                shuffle(data)
                randata = data[:(len(data)*0.6)]
                self.tree = np.zeros((randata.size,4))
                self.buildTree(randata)
                self.trees.append(self.tree[0:self.count+1])
                self.tree = None
                randata = None
            else:
                self.count = 0
                self.tree = np.zeros((data.size,4))
                self.buildTree(data)
                self.trees.append(self.tree[0:self.count+1])
                self.tree = None

    def query(self,test):
        k = self.k
        Yout = np.zeros(len(test))
        for t in range(0, k):
            tree = self.trees[t]
            for i in range(0,len(test)):
                j = 0;
                while tree[j,0] != -1:
                    if test[i,tree[j,0]] <= tree[j,1]:
                        j = tree[j,2]
                    elif test[i,tree[j,0]] > tree[j,1]:
                        j = tree[j,3]
                Yout[i] = Yout[i] + tree[j,1]
        Yout /= k
        return Yout

        
        
        
