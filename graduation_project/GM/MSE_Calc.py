import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
mean=np.load('./ref/ref_MSE.npy')

sgld=np.load('./result/SGLD/SGLD lr[2.50e-02,0.00,0.00].npy')
svrgld=np.load('./result/SVRGLD/SVRGLD lr[2.50e-02,0.00,0.00].npy')
raisld=np.load('./result/RAISLD/RAISLD lr[2.50e-02,0.00,0.00] alpha[0.10] d[0.0].npy')
raislde=np.load('./result/RAISLDe/RAISLDe lr[2.50e-02,0.00,0.00] alpha[0.10] d[0.0].npy')

lens=len(sgld)
sgld_mse=np.zeros(lens)
svrgld_mse=np.zeros(lens)
raisld_mse=np.zeros(lens)
raislde_mse=np.zeros(lens)

sgld_temp=0
svrgld_temp=0
raisld_temp=0
raislde_temp=0

for i in tqdm(range(lens)):
    sgld_temp=(sgld_temp*i+sgld[i])/(i+1)
    sgld_mse[i]=np.linalg.norm(sgld_temp-mean)**2

    svrgld_temp=(svrgld_temp*i+svrgld[i])/(i+1)
    svrgld_mse[i]=np.linalg.norm(svrgld_temp-mean)**2

    raisld_temp=(raisld_temp*i+raisld[i])/(i+1)
    raisld_mse[i]=np.linalg.norm(raisld_temp-mean)**2

    raislde_temp=(raislde_temp*i+raislde[i])/(i+1)
    raislde_mse[i]=np.linalg.norm(raislde_temp-mean)**2

save_folder='./result_MSE/'

np.save(save_folder+'SGLD.npy',sgld_mse)
np.save(save_folder+'SVRGLD.npy',svrgld_mse)
np.save(save_folder+'RAISLD.npy',raisld_mse)
np.save(save_folder+'RAISLDe.npy',raislde_mse)