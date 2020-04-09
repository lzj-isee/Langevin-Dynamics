import pandas
import numpy as np
import scipy.sparse as sp
from libsvm.python.svmutil import *

train_path='./DataSet/YearPredictionMSD-train.txt'
test_path='./DataSet/YearPredictionMSD-test.txt'
np.random.seed(1)

trainLabels, trainFeatures = svm_read_problem(train_path, return_scipy=True)
testLabels, testFeatures = svm_read_problem(test_path, return_scipy=True)

# DataSet is dense
trainFeatures=trainFeatures.toarray().astype('float32')
testFeatures=testFeatures.toarray().astype('float32')

# Save
np.save('./DataSet/train_labels.npy',trainLabels)
np.save('./DataSet/test_labels.npy',testLabels)
np.save('./DataSet/train_features.npy',trainFeatures)
np.save('./DataSet/test_features.npy',testFeatures)



a=1