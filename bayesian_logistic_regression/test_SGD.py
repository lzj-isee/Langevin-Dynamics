import numpy as np
import matplotlib.pyplot as plt
import csv
from matplotlib.gridspec import GridSpec
from class_LD import LD
from class_SGD import SGD
from class_SGLD import SGLD

if __name__ == "__main__":
    trainSource='./dataSet/a9a-train.txt'
    testSource='./dataSet/a9a-test.txt'
    featureDim=123
    maxIteration=10000
    batchSize=10
    startPoint=np.zeros(featureDim+1)
    stepSetting=[0.55,0.001,20]
    paraLambda=1
    dataAccessing='CA'
    test=SGD(
        trainSource=trainSource,
        testSource=testSource,
        featureDim=featureDim,
        maxIteration=maxIteration,
        batchSize=batchSize,
        startPoint=startPoint,
        stepSetting=stepSetting,
        paraLambda=paraLambda,
        dataAccessing=dataAccessing,
        interval=100
    )
    test.sgd()
    f=open('./targets/SGD:%d.csv'%(maxIteration),'w',encoding='utf-8')
    csv_writer=csv.writer(f)
    csv_writer.writerow(['dim','value'])
    for i in range(featureDim+1):
        csv_writer.writerow([i,test.iterations[maxIteration][i]])
    f.close()
    a=1