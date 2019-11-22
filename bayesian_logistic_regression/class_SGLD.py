from class_LD import LD
import numpy as np
import math

class SGLD(LD):
    def __init__(self,trainSource,testSource,featureDim,
    maxIteration,batchSize,startPoint,stepSetting,
    paraLambda,dataAccessing='RA',lens='all',interval=100):
        super(SGLD,self).__init__(
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
        self.__lens=lens
        self.injectedNoise=np.zeros(self._maxIteration)
        self.gradientNoise=np.zeros(self._maxIteration)
        self.distribution=np.zeros((self._dim,self._maxIteration))

    def _testCalc(self,i,paras,weights):
        '''
        SGLD迭代测试
        '''
        if((self._testNum!=0 and  i%self._interval==0 and self._interval!=0)or i==self._maxIteration-1):
            results=np.array([LD._testCalc(self,i,para) for para in paras])
            Y=np.dot(np.transpose(results),np.array(weights))/sum(np.array(weights))
            T=Y*2
            accuracy=sum(T.astype(int))/self._testNum
            avgloglikelihood=-sum(np.log(Y))/self._testNum
            self.accuracy.append(accuracy)
            self.avgloglikelihood.append(avgloglikelihood)
        
    def _noiseCalc(self,i,snaps):
        '''
        计算梯度噪声和注入噪声,可能派生类用不到这个函数
        '''
        lens=len(snaps)
        self.injectedNoise[i]=2*self.steps[i]
        average=sum(snaps)/lens
        covariance=np.zeros((self._dim,self._dim))
        for k in range(lens):
            covariance+=np.outer((snaps[k]-average),(snaps[k]-average))
        covariance=covariance*self._trainNum**2/lens**2
        a=np.linalg.eig(covariance)[0]
        self.gradientNoise[i]=max(a)*self.steps[i]**2

    def _postProcessing(self):
        LD._postProcessing(self)
        self.distribution=self.iterations.T


    def sgld(self,noiseCalc=False):
        for i in range(self._maxIteration):
            '''
            SGLD 迭代公式
            '''
            gradient=self._priorGradientCalc(i)+self._batchGradientCalc(i)
            iteration=self.iterations[i]+self.steps[i]*gradient+math.sqrt(2*self.steps[i])*\
                np.random.normal(0,1,self._dim)
            '''
            计算噪声
            '''
            if(noiseCalc==True):
                self._noiseCalc(i,self._gradientSnaps)
            '''
            append 新值
            '''
            self.iterations.append(np.array(iteration))
            self.gradients.append(np.array(gradient))
            '''
            test
            '''
            if(self.__lens=='all' or i+1<=self.__lens):
                self._testCalc(i,self.iterations[1:i+2],self.steps[0:i+1])
            elif(self.__lens==1):
                self._testCalc(i,iteration,self.steps[i])
            else:
                self._testCalc(i,self.iterations[i+2-self.__lens:i+2],self.steps[i+1-self.__lens:i+1])
            self._informationPrint(i)
        self._postProcessing()

