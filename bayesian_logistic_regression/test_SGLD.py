import numpy as np
import matplotlib.pyplot as plt
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
    scale=1000
    stepSetting=[0.55,0.001,20]
    paraLambda=1
    dataAccessing='CA'
    test=SGLD(
        trainSource=trainSource,
        testSource=testSource,
        featureDim=featureDim,
        maxIteration=maxIteration,
        batchSize=batchSize,
        startPoint=startPoint,
        stepSetting=stepSetting,
        paraLambda=paraLambda,
        dataAccessing=dataAccessing,
        lens=1000,
        interval=100,
    )
    test.sgld(noiseCalc=False)

    a=1