from SGLD import SGLD_sample
from SVRG_LD import SVRG_LD_sample
from SGHMC import SGHMC_sample
from SG_UL_MCMC import SG_UL_MCMC_sample
from SRVR_HMC import SRVR_HMC_sample
from SRM_HMC import SRM_HMC_sample
from NSRM_HMC import NSRM_HMC_sample
import numpy as np

#SGLD

factor_a_s=np.array([0.6,0.8,1.0,1.2,1.4])
factor_gamma_s=np.array([0,0.05,0.08,0.1])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={'pytorch':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma
            }
        save_folder='./SGLD_result_it'
        SGLD_sample(random_seed,train_setting,save_folder)

#SVRG_LD

factor_a_s=np.array([0.7,0.5,0.3,0.2,0.125,0.09,0.07])
factor_gamma_s=np.array([0,0.01,0.05])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={'pytorch':2020,'numpy':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma
            }
        save_folder='./SVRG_LD_result_it3'
        SVRG_LD_sample(random_seed,train_setting,save_folder)

#SGHMC

factor_a_s=np.array([0.6,0.8,1.0,1.2,1.4])
factor_gamma_s=np.array([0,0.05,0.08,0.1])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={'pytorch':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma,
            'beta':1,
            'gamma':1
            }
        save_folder='./SGHMC_result_it'
        SGHMC_sample(random_seed,train_setting,save_folder)

#SG_UL_MCMC

factor_a_s=np.array([0.06,0.08,0.1,0.15,0.2])
factor_gamma_s=np.array([0,0.05,0.08,0.10])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={
            'pytorch':2020,
            'numpy':2020
            }
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma,
            'u':1,
            'gamma':1
            }
        save_folder='./SG_UL_MCMC_result_it'
        SG_UL_MCMC_sample(random_seed,train_setting,save_folder)

#SRVR_HMC

factor_a_s=np.array([0.03,0.04,0.05,0.06,0.07])
factor_gamma_s=np.array([0,0.05,0.08,0.10])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={
            'pytorch':2020,
            'numpy':2020}
        train_setting={
            'num_epoch':20000,
            'batchSizeB0':500,
            'batchSizeB':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma,
            'u':1,
            'gamma':1
            }
        save_folder='./SRVR_HMC_result_it'
        SRVR_HMC_sample(random_seed,train_setting,save_folder)

#SRM_HMC

factor_a_s=np.array([1,0.5,0.25,0.125,0.0625])
factor_t_s=np.array([0.95,0.9,0.85,0.8])
for factor_a in factor_a_s:
    for factor_t in factor_t_s:
        random_seed={
            'pytorch':2020,
            'numpy':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':0,
            'u':1,
            'gamma':1,
            'factor_t':factor_t
            }
        save_folder='./SRM_HMC_result_it2'
        SRM_HMC_sample(random_seed,train_setting,save_folder)
    
#NSRM_HMC

factor_a_s=np.array([0.02,0.03,0.04,0.05,0.06,0.07,0.08])
factor_gamma_s=np.array([0,0.01,0.05])
for factor_a in factor_a_s:
    for factor_gamma in factor_gamma_s:
        random_seed={
            'pytorch':2020,
            'numpy':2020}
        train_setting={
            'num_epoch':20000,
            'batchSize':10,
            'dim':2,
            'factor_a':factor_a,
            'factor_b':0,
            'factor_gamma':factor_gamma,
            'u':1,
            'gamma':1,
            'factor_i':100
            }
        save_folder='./NSRM_HMC_result_it3'
        NSRM_HMC_sample(random_seed,train_setting,save_folder)
