from class_LD import LD
import numpy as np

class SGD(LD):
    def __init__(self,trainSource,testSource,featureDim,
    maxIteration,batchSize,startPoint,stepSetting,
    paraLambda,dataAccessing='RA',interval=100):
        super(SGD,self).__init__(
            trainSource=trainSource,
            testSource=testSource,
            featureDim=featureDim,
            maxIteration=maxIteration,
            batchSize=batchSize,
            startPoint=startPoint,
            stepSetting=stepSetting,
            paraLambda=paraLambda,
            dataAccessing=dataAccessing,
            interval=interval
        )
    
    def sgd(self):
        for i in range(self._maxIteration):
            '''
            SGD 迭代公式
            '''
            gradient=self._priorGradientCalc(i)+self._batchGradientCalc(i)
            iteration=self.iterations[i]+self.steps[i]*gradient
            self.iterations.append(np.array(iteration))
            self.gradients.append(np.array(gradient))
            '''
            test
            '''
            self._testCalc(i,np.array(iteration))
            self._informationPrint(i)
        self._postProcessing()

