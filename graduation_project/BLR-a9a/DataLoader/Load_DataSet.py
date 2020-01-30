import os
import numpy as np
from libsvm.python.svmutil import *
from scipy.sparse import csr_matrix



def Load_a9a(train_path,test_path):
    trainLabels, trainFeatures = svm_read_problem(train_path, return_scipy=True)
    testLabels, testFeatures = svm_read_problem(test_path, return_scipy=True)
    train_Datas={'labels':trainLabels,'features':trainFeatures}
    test_Datas={'labels':testLabels,'features':testFeatures}
    return train_Datas,test_Datas