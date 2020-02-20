import pandas as pd
from scipy.sparse import csr_matrix



def load_dataset(
    train_path='./DataSet/superconduct_train.csv'):
    d=pd.read_csv(train_path)._values
    trainFeatures=d[:,0:81]
    trainLabels=d[:,81]
    #testLabels, testFeatures = svm_read_problem(test_path, return_scipy=True)
    train_Datas={'labels':trainLabels,'features':trainFeatures}
    #test_Datas={'labels':testLabels,'features':testFeatures}
    #train_dim=trainFeatures.shape[1]
    #test_dim=testFeatures.shape[1]
    #assert(train_dim>=test_dim,"ERROR: train_dim < test_dim")
    return train_Datas#,test_Datas