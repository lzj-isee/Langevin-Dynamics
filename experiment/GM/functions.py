import numpy as np
import torch

def _f(x,a):
    result=2*torch.exp(-torch.pow(torch.norm(x-a),2)/2)
    result+=torch.exp(-torch.pow(torch.norm(x+a),2)/2)
    return torch.log(-result)
def grad_f(x,a):
    grad=torch.autograd.grad(_f(x,a),x)
    return grad[0]

@torch.no_grad()
def ng_grad_f(x,a):
    result=x-a
    result+=2*a/(1+2*torch.exp(2*torch.matmul(x,a)))
    #result+=2*a/(1+torch.exp(2*torch.matmul(x,a)))
    return result

@torch.no_grad()
def ng_en_f(x,a):
    result=2*torch.exp(-torch.pow(torch.norm(x-a),2)/2)
    result+=torch.exp(-torch.pow(torch.norm(x+a),2)/2)
    return result
