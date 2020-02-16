from libsvm.python.svmutil import *
from scipy.sparse import csr_matrix



def load_dataset_sp(
    train_path='./DataSet/a9a-train.txt',
    test_path='./DataSet/a9a-test.txt'):
    trainLabels, trainFeatures = svm_read_problem(train_path, return_scipy=True)
    testLabels, testFeatures = svm_read_problem(test_path, return_scipy=True)
    train_Datas={'labels':trainLabels,'features':trainFeatures}
    test_Datas={'labels':testLabels,'features':testFeatures}
    train_dim=trainFeatures.shape[1]
    test_dim=testFeatures.shape[1]
    assert(train_dim>=test_dim,"ERROR: train_dim < test_dim")
    return train_Datas,test_Datas