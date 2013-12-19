import numpy as np;
import csv;
import math,random,sys,bisect,time
import numpy,scipy.spatial.distance

class KNNLearner(object):

    def __init__(self,k):
        self.k = k
        self.data = None;
    
    def getDistance(self, testP, dataP):
        diff = testP - dataP
        sq = diff**2
        total_dist = math.sqrt(sq.sum())
        return total_dist
    
    
    def addEvidence(self,dataX, dataY):
        stime = time.time()
        if not dataY == None:
            data = np.zeros([dataX.shape[0],dataX.shape[1]+1])
            data[:,0:dataX.shape[1]]=dataX
            data[:,(dataX.shape[1])]=dataY
        else:
            data = dataX
        etime = time.time()
        self.data = data
        train_time = (etime-stime)/float(len(self.data))
        return train_time

    def query(self,test):
        avg = np.zeros(len(test))
        dist = np.zeros((len(test), len(self.data)))
        stime = time.time()
        for t in range(0, len(test)):
            for d in range(0, len(self.data)):
                dist[t,d] = self.getDistance(test[t],self.data[d,:(len(self.data[0])-1)])
            dist_sort = np.zeros((len(self.data),2))
            dist_sort[:,0] = dist[t,:]
            dist_sort[:,1] = self.data[:,(len(self.data[0])-1)]
            dist_sort = dist_sort[np.argsort(dist_sort[:,0])]
            knn = dist_sort[:self.k,1]
            sum = knn.sum(axis = 0)
            avg[t] = sum/self.k
        etime = time.time()
        test_time = (etime-stime)/float(len(self.data))
        return avg, test_time