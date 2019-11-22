import numpy as np
import random
import math
from scipy.special import expit as sigmoid

class LD(object):
    def __init__(self,trainSource,testSource,featureDim,
    maxIteration,batchSize,startPoint,stepSetting,
    paraLambda,dataAccessing='RA',interval=100):
        self._maxIteration=maxIteration#最大迭代次数
        self._batchSize=batchSize#batch大小
        self._startPoint=startPoint#迭代起始点
        self._paraLambda=paraLambda#正则化参数
        self._dataAccessing=dataAccessing#batch选取方式
        self._trainLabels,self._trainFeatures=self.__loading(trainSource,featureDim)
        self._testLabels,self._testFeatures=self.__loading(testSource,featureDim)
        self._trainNum=len(self._trainLabels)
        self._testNum=len(self._testLabels)
        self._dim=len(self._trainFeatures[0])#训练维数
        self._interval=interval#测试间隔
        self.__batchCount=0#batch在CA获取方式下的标记点
        self.epochs=list(map(lambda x: batchSize*x/self._trainNum,list(range(maxIteration))))#iteration转epoch
        self.iterations=[]#迭代过程中的坐标点
        self.iterations.append(startPoint)
        self.gradients=[]#迭代过程中的梯度
        self._gradientSnaps=np.zeros((self._batchSize,self._dim))#batch中每项的梯度snapshot
        self.choosenTimes=np.zeros(self._trainNum)#记录每个样本的选中次数
        self.steps=self.__stepsGen(stepSetting)#迭代步长
        self.accuracy=[]#测试项目，准确度
        self.avgloglikelihood=[]#测试项目，loss

    def __loading(self,source,dim):
        labels=[]
        features=[]
        dataNum=0
        '''
        读取数据
        '''
        with open(source,'r') as f:
            datas=f.readlines()
            dataNum=len(datas)
        '''
        转换格式
        '''
        for i in range(dataNum):
            line=datas[i]
            position=line.find(' ')
            label=eval(line[:position])
            feature=eval('{'+line[position+1:len(line)-2].replace(' ',',')+'}')
            temp=np.zeros(dim+1)
            temp[0]=1
            for key in feature.keys():
                temp[key]=1
            labels.append(label)
            features.append(temp)
        labels=np.array(labels)
        features=np.array(features)
        return labels,features

    def __stepsGen(self,stepSetting):
        '''
        计算迭代步长
        '''
        gamma,a,b=stepSetting
        #steps=list(map(lambda x: a*(b+x+1)**(-gamma),list(range(self._maxIteration))))
        steps=[a*(b+i+1)**(-gamma) for i in range(self._maxIteration)]
        return np.array(steps)

    def _getBatch(self):
        '''
        获取batch索引
        '''
        if(self._dataAccessing=='RA'):
            batches=random.sample(range(0,self._trainNum),self._batchSize)
        elif (self._dataAccessing=='CA'):
            start=self.__batchCount
            end=start+self._batchSize
            if(end<=len(self._trainLabels)):
                batches=list(range(start,end))
                self.__batchCount=end
            else:
                batches=list(range(start,len(self._trainLabels)))
                self.__batchCount=end-len(self._trainLabels)
                batches+=list(range(self.__batchCount))
        else:
            assert 0
        for index in batches:
            self.choosenTimes[index]+=1
        return batches

    def _priorGradientCalc(self,i):
        '''
        此处是先验的梯度算式
        '''
        return -self.iterations[i]/self._paraLambda**2

    def _likelihoodGradientCalc(self,i):
        '''
        计算batch中每个样本的梯度，保存在self._gradientSnaps中
        '''
        miniBatchs=self._getBatch()
        '''
        此处是似然的梯度算式
        '''
        Z=[-self._trainLabels[index]*np.dot(self._trainFeatures[index],self.iterations[i])\
            for index in miniBatchs]
        T=np.transpose(np.array([self._trainLabels[index]*self._trainFeatures[index]\
            for index in miniBatchs]))
        self._gradientSnaps=np.transpose(np.dot(T,np.diag(sigmoid(Z))))

    def _batchGradientCalc(self,i):
        '''
        计算整个batch的梯度
        '''
        self._likelihoodGradientCalc(i)
        return sum(self._gradientSnaps)*self._trainNum/self._batchSize

    def _testCalc(self,i,para):
        '''
        测试
        '''
        if((self._testNum!=0 and  i%self._interval==0 and self._interval!=0)or i==self._maxIteration-1):
            accuracy=0
            avgloglikelihood=0
            Z=[self._testLabels[j]*np.dot(self._testFeatures[j],para) for j in range(self._testNum)]
            Y=sigmoid(Z)
            T=Y*2
            accuracy=sum(T.astype(int))/self._testNum
            avgloglikelihood=-sum(np.log(Y))/self._testNum
            self.accuracy.append(accuracy)
            self.avgloglikelihood.append(avgloglikelihood)
            return Y

    def _informationPrint(self,i):
        if((self._testNum!=0 and  i%self._interval==0 and self._interval!=0)or i==self._maxIteration-1):
            print('第%d次迭代: '%(i))
            print('setpSize: ',self.steps[i])
            print('epochs: ',self.epochs[i])
            print('test_accuracy: ',self.accuracy[len(self.accuracy)-1])
            print('avgloglikelihood: ',self.avgloglikelihood[len(self.avgloglikelihood)-1])
            print('----------------------------------------------')
    
    def _postProcessing(self):
        '''
        list转array
        '''
        self.iterations=np.array(self.iterations)
        self.gradients=np.array(self.gradients)
        self.accuracy=np.array(self.accuracy)
        self.avgloglikelihood=np.array(self.avgloglikelihood)
        self.epochs=np.array(self.epochs)