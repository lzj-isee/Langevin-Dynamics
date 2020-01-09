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