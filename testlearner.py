import numpy as np;
import csv;
import math,random,sys,bisect,time
import numpy,scipy.spatial.distance
import matplotlib.pyplot as plt
from KNNLearner import KNNLearner
from RandomForestLearner import RandomForestLearner

def getflatcsv(fname):
    inf = open(fname)
    return numpy.array([map(float,s.strip().split(',')) for s in inf.readlines()])

def getstats(Y, YTest):
    diff = (Y - YTest)
    diff = diff**2
    sum = diff.sum()
    rms = math.sqrt(sum/len(YTest))
    corMatrix = np.corrcoef(Y, YTest)
    corrcoef_value = corMatrix[0,1]
    return rms, corrcoef_value

def main():
    
    isBagging = True
    
    file1 = "data-classification-prob.csv"
    file2 = "data-ripple-prob.csv"
    knn_rms1 = np.zeros((101,1))
    knn_corrcoef1 = np.zeros((101,1))
    
    knn_rms2 = np.zeros((101,1))
    knn_corrcoef2 = np.zeros((101,1))
    
    randomForest_rms1 = np.zeros((101,1))
    randomForest_corrcoef1 = np.zeros((101,1))
    
    randomForest_rms2 = np.zeros((101,1))
    randomForest_corrcoef2 = np.zeros((101,1))
    
    randomForestBagging_corrcoef1 = np.zeros((101,1))
    randomForestBagging_corrcoef2 = np.zeros((101,1))
    
    randomForestBagging_rms1 = np.zeros((101,1))
    randomForestBagging_rms2 = np.zeros((101,1))
    
    k = np.arange(1,101)
    
    for i in range(1,3):
        if i == 1:
            print 'Starting with dataset 1....'
            file = file1
        else:
            print 'Starting with dataset 2....'
            file = file2
    
        data = getflatcsv(file)

        XTrain = data[:(len(data)*0.6),:(len(data[0])-1)]
        XTest = data[(len(data)*0.6):,:(len(data[0])-1)]
    
        YTrain = data[:(len(data)*0.6),-1]
        YTest = data[(len(data)*0.6):,-1]
        if i == 1:
            YTest1 = YTest
        else:
            YTest2 = YTest
            
        for j in range(1,3):
            if j == 1:
                print 'Calling KNNLearner for dataset %d...' % i
                for count in range(1,101):
                    knnLearner = KNNLearner(k=count)
                    train_t = knnLearner.addEvidence(XTrain, YTrain)
                    Y, test_t = knnLearner.query(XTest)
                    if i == 1:
                        knn_rms1[count,0], knn_corrcoef1[count,0] = getstats(Y, YTest)
                    else:
                        knn_rms2[count,0], knn_corrcoef2[count,0] = getstats(Y, YTest)
            elif j == 2:
                print 'Calling RandomForestLearner for dataset %d...' % i
                for count in range(1,101):
                    if isBagging:
                        randomForestLearner = RandomForestLearner(k=count, isBagging = True)
                        randomForestLearner.addEvidence(XTrain, YTrain)
                        Y = randomForestLearner.query(XTest)
                        if i == 1:
                            randomForestBagging_rms1[count,0], randomForestBagging_corrcoef1[count,0] = getstats(Y, YTest)
                            print count, randomForestBagging_corrcoef1[count,0]
                        else:
                            randomForestBagging_rms2[count,0], randomForestBagging_corrcoef2[count,0] = getstats(Y, YTest)
                            print count, randomForestBagging_corrcoef2[count,0]
                
                    randomForestLearner = RandomForestLearner(k=count, isBagging = False)
                    randomForestLearner.addEvidence(XTrain, YTrain)
                    Y = randomForestLearner.query(XTest)
                    if i == 1:
                        randomForest_rms1[count,0], randomForest_corrcoef1[count,0] = getstats(Y, YTest)
                        print count, randomForest_corrcoef1[count,0]
                    else:
                        randomForest_rms2[count,0], randomForest_corrcoef2[count,0] = getstats(Y, YTest)
                        print count, randomForest_corrcoef2[count,0]
                
    if isBagging:
        plt.ylabel('Random Forest:Corelation Coefficient - dataset 1')
        plt.xlabel('K')
        plt.legend(['Without Bagging','With Bagging'])
        plt.plot(k, randomForest_corrcoef1[1:], k, randomForestBagging_corrcoef1[1:]);
        plt.savefig('bagging_corr1.png')
        plt.close()
                                
        plt.ylabel('Random Forest:Corelation Coefficient - dataset 2')
        plt.xlabel('K')
        plt.legend(['Without Bagging','With Bagging'])
        plt.plot(k, randomForest_corrcoef2[1:], k, randomForestBagging_corrcoef2[1:]);
        plt.savefig('bagging_corr2.png')
        plt.close()

    plt.ylabel('Corelation Coefficient - dataset 1')
    plt.xlabel('K')
    plt.legend(['KNN','Random Forest'])
    plt.plot(k, knn_corrcoef1[1:], k, randomForest_corrcoef1[1:]);
    plt.savefig('corr1.png')
    plt.close()

    plt.ylabel('Corelation Coefficient - dataset 2')
    plt.xlabel('K')
    plt.legend(['KNN','Random Forest'])
    plt.plot(k, knn_corrcoef2[1:], k, randomForest_corrcoef2[1:]);
    plt.savefig('corr2.png')
    plt.close()

    plt.ylabel('RMS - dataset 1')
    plt.xlabel('K')
    plt.legend(['KNN','Random Forest'])
    plt.plot(k, knn_rms1[1:], k, randomForest_rms1[1:])
    plt.savefig('Compare_RMS1.png')
    plt.close()

    plt.ylabel('RMS - dataset 2')
    plt.xlabel('K')
    plt.legend(['KNN','Random Forest'])
    plt.plot(k, knn_rms2[1:], k, randomForest_rms2[1:])
    plt.savefig('Compare_RMS2.png')
    plt.close()


if __name__ == '__main__':
    main()

