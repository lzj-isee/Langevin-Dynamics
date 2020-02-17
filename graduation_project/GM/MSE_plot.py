import numpy as np
import matplotlib.pyplot as plt



sgld_mse=np.load('./result_MSE/SGLD.npy')
svrgld_mse=np.load('./result_MSE/SVRGLD.npy')
raisld_mse=np.load('./result_MSE/RAISLD.npy')
raislde_mse=np.load('./result_MSE/RAISLDe.npy')

plt.plot(sgld_mse,label='SGLD',color='blue')
plt.plot(svrgld_mse,label='SVRGLD',color='cyan')
plt.plot(raisld_mse,label='RAISLD',color='gold')
plt.plot(raislde_mse,label='RAISLDe',color='deeppink')
plt.legend()
plt.ylim([-0.01,3])
plt.xlabel('Number of iterations')
plt.ylabel('Mean square error')
plt.show()