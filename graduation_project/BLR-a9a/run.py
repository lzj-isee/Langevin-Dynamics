from Algorithms.SGLD import SGLD_trian
from Algorithms.SVRG import SVRG_trian
from Algorithms.RAIS import RAIS_trian
import pretty_errors

lr_a=8e-5
lr_b=0
lr_gamma=0.5
num_epochs=100
batchSize=500
eval_interval=12
random_seed=1
save_folder='./result/'
train_path='./DataSet/a9a-train.txt'
test_path='./DataSet/a9a-test.txt'
RAIS_trian(
    lr_a=lr_a,
    lr_b=lr_b,
    lr_gamma=lr_gamma,
    num_epochs=num_epochs,
    batchSize=batchSize,
    alpha=0.1,
    d=1,
    eval_interval=eval_interval,
    random_seed=random_seed,
    save_folder=save_folder,
    use_gpu=False,
    train_path=train_path,
    test_path=test_path)
'''
SGLD_trian(
    lr_a=lr_a,
    lr_b=lr_b,
    lr_gamma=lr_gamma,
    num_epochs=num_epochs,
    batchSize=batchSize,
    eval_interval=eval_interval,
    random_seed=random_seed,
    save_folder=save_folder,
    use_gpu=False,
    train_path=train_path,
    test_path=test_path)

SVRG_trian(
    lr_a=lr_a,
    lr_b=lr_b,
    lr_gamma=lr_gamma,
    num_epochs=num_epochs,
    batchSize=batchSize,
    eval_interval=eval_interval,
    random_seed=random_seed,
    save_folder=save_folder,
    use_gpu=True,
    train_path=train_path,
    test_path=test_path)
'''