import numpy as np
import matplotlib.pyplot as plt
import os



path='./result/SGLD/'
file_names=list(os.listdir(path))
for i,file_name in enumerate(file_names):
    if file_name[len(file_name)-3:len(file_name)] != 'npy':
        file_names.pop(i)

for file_name in file_names:
    data=np.load(path+file_name)[1]
    plt.plot(data,label=file_name)
plt.legend()
plt.ylim([0.9,0.98])
plt.grid()
plt.show()
