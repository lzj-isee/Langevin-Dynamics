import numpy as np
import matplotlib.pyplot as plt
import os

path='./result/SRVR_HMC/'
file_names=list(os.listdir(path))
for i,file_name in enumerate(file_names):
    if file_name[len(file_name)-3:len(file_name)] != 'npy':
        file_names.pop(i)

for file_name in file_names:
    data=np.load(path+file_name)[1]
    plt.plot(data,label=file_name)
plt.legend()
plt.show()
