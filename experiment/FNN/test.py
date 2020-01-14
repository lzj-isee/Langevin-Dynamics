import numpy as np
import matplotlib.pyplot as plt
sgld=np.load('./result/SGLD/lr_a[0.0003]lr_gamma[0.01].npy')
train_loss=sgld[0]
train_corr=sgld[1]
test_loss=sgld[2]
test_corr=sgld[3]

plt.plot(train_corr,label='train_corr')
plt.plot(test_corr,label='test_corr')
plt.legend()
plt.show()