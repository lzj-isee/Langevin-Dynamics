from SGLD import SGLD_train
import numpy as np

a=np.linspace(0.001,0.01,10)

for i in range(10):
    random_seed={'pytorch':2020}
    train_setting={
        'num_epoch':10,
        'batchSize':64,
        'dim':123+1,
        'factor_a':a[i],
        'factor_b':0,
        'factor_gamma':0.7
        }
    test_interval=50
    save_folder='./SGLD_result'
    SGLD_train(random_seed,train_setting,test_interval,save_folder)
