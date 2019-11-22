from class_LangevinDynamics import LangevinDynamics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math

def pdf(x,center=[2,2]):
    return (math.exp(-(x-center[0])**2/2)+math.exp(-(x+center[1])**2/2))/(2*math.sqrt(2*np.pi))

if __name__ == "__main__":
    dimension=3  #维数
    mean_vec=[2,2,2] #高斯分布中心坐标
    sample_size=500  #样本量
    max_iteration=20000 #最大迭代次数
    #step_setting=[-0.75,150,500,False] #步长设置参数，三参数分别为gamma, a, b :  step_size=a*(b+t)**gamma
    step_setting=[1.3,0.15,0.1,True]
    batch_size=10   #设置batch的大小
    start_setting=[0,1,False]  #起始坐标设置, 三参数分别为坐标均值，坐标方差，是否使用高斯随机
    SGLD=LangevinDynamics(dimension,sample_size,mean_vec,\
        max_iteration,batch_size,start_setting)
    
    SGLD.sgld(step_setting,mode=False,latency=0.005) #调用函数进行计算
    points=np.array(SGLD.x).T
    pointsA=(np.array(SGLD.x)[0:(int)(max_iteration/2)]).T    #获取前10000个迭代点
    pointsB=(np.array(SGLD.x)[(int)(max_iteration/2):max_iteration-1]).T #获取后10000个迭代点
    '''
    ref calculate
    '''
    xs1=np.linspace(-6,6,120)   #准备参考的曲线数据
    ys1=[]
    for x in xs1:
        ys1.append(pdf(x))
    ys1=np.array(ys1)
    '''
    数据plot
    '''
    print("maxStrpSize:%.4f"%(SGLD.maxStepSize))
    print("minStepSiza:%.4f"%SGLD.minStepSize)
    print('setpPara:',SGLD.stepPara)
    '''
    第一个figure：
    画出散点图，二维直方图，两个维度分别的直方图和标准曲线
    '''
    plt.figure(figsize=(12.8,7.2))
    ax1=plt.subplot2grid((8,13),(0,0),rowspan=6,colspan=6)      
    plt.xlabel("dim0",size=15)
    plt.ylabel("dim1",size=15)
    ax2=plt.subplot2grid((8,13),(0,7),rowspan=6,colspan=6)
    plt.xlabel("dim0",size=15)
    plt.ylabel("dim1",size=15)
    ax3=plt.subplot2grid((8,13),(7,0),colspan=6)
    plt.xlabel("dim1",size=15)
    ax4=plt.subplot2grid((8,13),(7,7),colspan=6)
    plt.xlabel("dim0",size=15)
    ax1.scatter(pointsA[0],pointsA[1],s=20,c='r',alpha=0.05)        #红色表示前10000次迭代点
    ax1.scatter(pointsB[0],pointsB[1],s=20,c='b',alpha=0.05)        #蓝色表示后10000次迭代点
    ax1.set_xlim(-6,6)
    ax1.set_ylim(-6,6)
    ax2.hist2d(points[0],points[1],bins=40,range=[[-6,6],[-6,6]])   
    ax3.hist(points[1],bins=40,range=[-6,6],density=True)
    ax3.plot(xs1,ys1)
    ax4.hist(points[0],bins=40,range=[-6,6],density=True)
    ax4.plot(xs1,ys1)
    '''
    第二个figure：
    梯度噪声与注入噪声
    '''
    plt.figure()
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(range(max_iteration),SGLD.injectedNoise)
    plt.plot(range(max_iteration),SGLD.gradientNoise)
    plt.show()


    a=1