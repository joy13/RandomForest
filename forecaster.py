import numpy as np;
import csv;
import math,random,sys,bisect,time
import numpy,scipy.spatial.distance
import matplotlib.pyplot as plt
from LinRegLearner import LinRegLearner
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


def getFeatures(train):
    features = np.zeros((train.shape[0],5))
    YTrain = np.zeros((train.shape[0],1))
    length = 0
    for record in range(20,train.shape[0]-5):
        start = record+1 - 21
#        features[start,0] = train[record] - train[record-1]
#        features[start,1] = (train[record] - train[record-1]) - (train[record-1]- train[record-2])
        dv1 = 0
        dv2 = 0
        c1 = 0
        c2 = 0
        for j in range(start+1,record):
            dv1 = dv1 + (train[j] - train[j-1])
            c1 += 1
        for j in range(start+2,record):
            dv2 = dv2 + (train[j] - train[j-1]) - (train[j-1]- train[j-2])
            c2 += 1
        features[start,0] = dv1/c1
        features[start,1] = dv2/c2
        amp = np.amax(train[start:record]) - np.amin(train[start:record])
        std = np.std(train[start:record])
        mean = np.mean(train[start:record])
        count = 0
        for i in range (start,record):
            if train[i] > mean:
                count += 1
        features[start,2] = amp
        features[start,3] = std
        features[start,4] = count

        YTrain[start,0] = train[record] - train[record+5]
        length += 1

    features = features[:length]
    YTrain = YTrain[:length]
            
    return features, YTrain[:,0]

def main():    
    data = []
    datat = []
    datat1 = []
    for i in range(0,9):
        file = 'ML4T-00'+str(i)+'.csv'
        reader = csv.reader(open(file,'rU'),delimiter = ',')
        firstline = True
        for row in reader:
            if firstline:    #skip first line
                firstline = False
                continue
            data.append(row[1])
    for i in range(10,99):
        file = 'ML4T-0'+str(i)+'.csv'
        reader = csv.reader(open(file,'rU'),delimiter = ',')
        firstline = True
        for row in reader:
            if firstline:    #skip first line
                firstline = False
                continue
            data.append(row[1])

    file = 'ML4T-292.csv'
    reader = csv.reader(open(file,'rU'),delimiter = ',')
    firstline = True
    for row in reader:
        if firstline:    #skip first line
            firstline = False
            continue
        datat.append(row[1])

    file = 'ML4T-112.csv'
    reader = csv.reader(open(file,'rU'),delimiter = ',')
    firstline = True
    for row in reader:
        if firstline:    #skip first line
            firstline = False
            continue
        datat1.append(row[1])

    data.reverse()
    datat.reverse()
    datat1.reverse()

    train = np.zeros((len(data),1))

    for i in range(0, len(data)):
        train[i] = float(data[i])
    
    test = np.zeros((len(datat),1))

    for i in range(0, len(datat)):
        test[i] = float(datat[i])

    test1 = np.zeros((len(datat1),1))

    for i in range(0, len(datat1)):
        test1[i] = float(datat1[i])

    XTrain,YTrain = getFeatures(train)

    f11 = XTrain[0:100,0]
    f12 = XTrain[0:100,1]
    f13 = XTrain[0:100,2]
    f14 = XTrain[0:100,3]
    f15 = XTrain[0:100,4]

    XTest, dummy = getFeatures(test)
    YTest = test[25:,0]

    XTest1, dummy1 = getFeatures(test1)
    YTest1 = test1[25:,0]

    f21 = XTest[0:100,0]
    f22 = XTest[0:100,1]
    f23 = XTest[0:100,2]
    f24 = XTest[0:100,3]
    f25 = XTest[0:100,4]
    
    print 'Calling RandomForestLearner for dataset...'
    randomForestLearner = RandomForestLearner(k=70, isBagging = False)
    randomForestLearner.addEvidence(XTrain, YTrain)
    Yr = randomForestLearner.query(XTest)
    temp2 = test[25:,0]
    Yr += temp2
    Ypt = np.zeros(100)
    Ypt[25:100] = Yr[0:75]
    
    Ypl = Yr[(len(Yr)-100):]

    rms3,corr3 = getstats(Yr, YTest)
    print 'Random Forest Corelation Coeficient for GLOBAL dataset with 70 trees', corr3

    Yr1 = randomForestLearner.query(XTest1)
    temp3 = test1[25:,0]
    Yr1 += temp3
    Ypt1 = np.zeros(100)
    Ypt1[25:100] = Yr1[0:75]
    Ypl1 = Yr1[(len(Yr1)-100):]
    rms4,corr4 = getstats(Yr1, YTest1)
    print 'Random Forest Corelation Coeficient for MY dataset with 70 trees', corr4
            
    k = np.arange(1,101)
    plt.title('Global Data: First 100 days')
    plt.ylabel('Y Value')
    plt.xlabel('Days')
    plt.plot(k,YTest[0:100], k, Ypt);
    plt.savefig('first1.png')
    plt.close()
    
    plt.title('MLT4-112.csv Data: First 100 days')
    plt.ylabel('Predicted Y')
    plt.xlabel('Days')
    plt.plot(k, YTest1[0:100], k, Ypt1);
    plt.savefig('first2.png')
    plt.close()
    
    plt.title('Global Data: Last 100 days')
    plt.ylabel('Y Value')
    plt.xlabel('Days')
    print YTest[(len(YTest)-100):len(YTest)].shape
    plt.plot(k,YTest[(len(YTest)-100):len(YTest)], k, Ypl);
    plt.savefig('last1.png')
    plt.close()
                
    plt.title('MLT4-112.csv Data: Last 100 days')
    plt.ylabel('Predicted Y')
    plt.xlabel('Days')
    plt.plot(k, YTest1[(len(YTest1)-100):], k, Ypl1);
    plt.savefig('last2.png')
    plt.close()
    
    plt.xlabel('Predicted Y')
    plt.ylabel('Actual Y')
    plt.scatter(Yr, YTest)
    plt.savefig('Ycompare1.png')        
    plt.close()
                
    plt.xlabel('Predicted Y')
    plt.ylabel('Actual Y')
    plt.scatter(Yr1, YTest1)
    plt.savefig('Ycompare2.png')
    plt.close()
                
    plt.title('Features: GLOBAL Dataset')
    plt.ylabel('Feature Value')
    plt.xlabel('Days')
    plt.plot(k,f11,k,f12,k,f13,k,f14,k,f15);
    plt.savefig('feature1.png')
    plt.close()
                
    plt.title('Features: MLT4-112.csv Data')
    plt.ylabel('Feature Value')
    plt.xlabel('Days')
    plt.plot(k,f21,k,f22,k,f23,k,f24,k,f25);
    plt.savefig('feature2.png')
    plt.close()


if __name__ == '__main__':
    main()

