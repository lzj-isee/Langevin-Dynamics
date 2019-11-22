import numpy as np
import random
import math
import time

class LangevinDynamics(object):
    def __init__(self,dim,data_num,data_vec,max_iteration,batch_size,start_setting):
        self._dim=dim
        self._data_num=data_num
        self._data_vec=data_vec
        self._max_iteration=max_iteration
        self._batch_size=batch_size
        self._start_setting=start_setting
        self._getData()
        self._startInit()

    def _getData(self):     #获取样本点
        self.datas=np.random.multivariate_normal(self._data_vec,np.eye(self._dim),(self._data_num))

    '''
    def _getSamples(self):      #选取batch
        self.index=random.sample(range(0,self._data_num),self._batch_size)
    '''
    def _startInit(self):   #初始化
        self.x=[]
        self.gradients=[]
        self.steps=np.zeros(self._max_iteration)
        self.injectedNoise=np.zeros(self._max_iteration)
        self.gradientNoise=np.zeros(self._max_iteration)
        self.choosenTimes=np.zeros(self._data_num)
        self._score=np.zeros((self._batch_size,self._dim))
        if(self._start_setting[2]==False):      #若不使用高斯随机，初始值固定
            self.x.append(np.ones(self._dim)*self._start_setting[0])
        else:       #使用高斯随机
            self.x.append(np.random.normal(self._start_setting[0],self._start_setting[1],self._dim))

    def _stepSizeparaCal(self,step_setting):    #步长计算所需的参数
        if (step_setting[3]==False):
            return step_setting[0:3]
        else:
            max_step,mid_step,min_step=step_setting[0:3]
            gamma=math.log2(min_step/mid_step)
            a=mid_step/(self._max_iteration/2)**gamma
            b=((max_step/a)**(1/gamma))-1
            return gamma,a,b
            
    def _getBatch(self):
        return random.sample(range(0,self._data_num),self._batch_size)

    def _stepCalculate(self,i,step_setting):
        gamma,a,b=self._stepSizeparaCal(step_setting)
        self.maxStepSize=a*b**gamma    #计算最大步长，用于调整
        self.minStepSize=a*(b+self._max_iteration)**gamma#计算最小步长
        self.stepPara=[gamma,a,b]
        step=a*(b+i+1)**gamma
        self.steps[i]=step#记录第i次迭代的步长
        return step

    def _gradientCalculate(self,step,i):
        minibatchs=self._getBatch()
        gradient=np.zeros(self._dim)
        count=0
        for j in minibatchs:
            self.choosenTimes[j]+=1#记录被选中样本的频次
            '''
            gradient calculate
            '''
            self._score[count]=self.x[i]-self.datas[j]+2*self.datas[j]/(1+math.exp(2*np.sum(self.x[i]*self.datas[j])))
            gradient+=self._score[count]#score用来算梯度的噪声
            count+=1
        return gradient/self._batch_size

    def _noiseCalculate(self,i):
        self.injectedNoise[i]=2*self.steps[i]   #注入噪声的方差为两倍步长
        average=np.sum(self._score,0)/self._batch_size
        covariance=np.zeros((self._dim,self._dim))
        for k in range(self._batch_size):
            covariance+=np.outer((self._score[k]-average),(self._score[k]-average))
        covariance=covariance/self._batch_size**2
        a,b=np.linalg.eig(covariance)
        self.gradientNoise[i]=max(a)*self.steps[i]**2


    def sgld(self,step_setting,mode=False,latency=0.01):     #SGLD算法
        for i in range(self._max_iteration):
            if(mode==True): time.sleep(latency)  #延时
            '''
            步长计算
            '''
            step=self._stepCalculate(i,step_setting)
            '''
            计算梯度
            '''
            gradient=self._gradientCalculate(step,i)
            '''
            记录噪声
            '''
            self._noiseCalculate(i)
            '''
            迭代更新
            '''
            temp=self.x[i]-step*gradient+math.sqrt(2*step)*np.random.normal(0,1,self._dim)
            self.x.append(temp)
            self.gradients.append(np.array(gradient))
