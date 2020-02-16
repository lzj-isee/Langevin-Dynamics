import numpy as np

def load_dataset(path='./dataset/points.npy'):
    return np.load(path)

def load_ref(path='./ref/ref.npy'):
    return np.load(path)