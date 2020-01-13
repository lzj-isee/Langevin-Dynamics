import numpy as np
import torch
#from numba import jit

'''
def _f(x,a):
    result=2*torch.exp(-torch.pow(torch.norm(x-a),2)/2)
    result+=torch.exp(-torch.pow(torch.norm(x+a),2)/2)
    return torch.log(-result)
def grad_f(x,a):
    grad=torch.autograd.grad(_f(x,a),x)
    return grad[0]
'''
'''
@torch.no_grad()
def ng_grad_f(x,a):
    result=x-a
    result+=2*a/(1+2*torch.exp(2*torch.matmul(x,a)))
    #result+=2*a/(1+torch.exp(2*torch.matmul(x,a)))
    return result
'''
'''
@torch.no_grad()
def ng_en_f(x,a):
    result=2*torch.exp(-torch.pow(torch.norm(x-a),2)/2)
    result+=torch.exp(-torch.pow(torch.norm(x+a),2)/2)
    return result
'''
def grad_f(x,a):
    batchSize=len(a)
    dim=len(x)
    X=np.repeat(np.reshape(x,(-1,dim)),repeats=batchSize,axis=0)
    z=np.reshape((1+2*np.exp(np.dot(x,a.T))),(-1,1))
    result=X-a
    result+=2*a/np.repeat(z,repeats=dim,axis=1)
    return result


def noise_Gen1(u,gamma,eta,dim):
    cov=np.ones((dim*2,dim*2))*\
        (u*gamma**(-1)*(1-2*np.exp(-gamma*eta)+np.exp(-2*gamma*eta)))
    stdx=np.ones(dim)*\
        (u*gamma**(-2)*(2*gamma*eta+4*np.exp(-gamma*eta)-np.exp(-2*gamma*eta)-3))
    stdv=np.ones(dim)*\
        (u*(1-np.exp(-2*gamma*eta)))
    std=np.concatenate((stdx,stdv))
    m=cov-np.diag(np.diag(cov))+np.diag(std)
    m=np.diag(std)
    noise=np.random.multivariate_normal(np.zeros(dim*2),m)
    return noise[:dim], noise[dim:]

'''
def  noise_Gen1(u,gamma,eta,dim):
    cov=np.ones((dim*2,dim*2))*\
        (u*gamma**(-1)*(1-2*np.exp(-gamma*eta)+np.exp(-2*gamma*eta)))
    stdx=np.ones(dim)*\
        (u*gamma**(-2)*(2*gamma*eta+4*np.exp(-gamma*eta)-np.exp(-2*gamma*eta)-3))
    stdv=np.ones(dim)*\
        (u*(1-np.exp(-2*gamma*eta)))
    std=np.concatenate((stdx,stdv))
    m=torch.Tensor(cov-np.diag(np.diag(cov))+np.diag(std))
    Normal=torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(dim*2),m)
    noise=Normal.sample()
    return noise[:dim].numpy(), noise[dim:].numpy()
    '''



