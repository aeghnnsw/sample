import os
import numpy as np
from sklearn.metrics import (mean_absolute_error,\
                             mean_squared_error, \
                             r2_score)
from scipy.stats import pearsonr
import torch
from qsar.dataset import split_dataset
from qsar.dataset import QSARBaseDataset
import pickle
import time
from glob import glob


class BaseQSARModel():
    def __init__(self,model_dir:str,dataset:QSARBaseDataset,mode='Regression'):
        self.model_dir = model_dir
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        os.chdir(self.model_dir)
        self.random_seeds = np.arange(20)
        self.dataset = dataset
        self.logger = None
        # self.logger_columns = ['MAE','RMSE','MaxE','MAPE','R2','Pearson_r2','Descpts_importance',\
                            #    'XY_plot','Williams_plot','KDE_plot']
        # assert mode in ['Regression','Classification'],'mode can be [Regression,Classification]'
        self.mode = mode

    @classmethod
    def load_kwargs(cls,kwargs_file):
        # load kwargs from file
        with open(kwargs_file,'rb') as f:
            data_kwargs,model_kwargs = pickle.load(f)
        return data_kwargs,model_kwargs

    
    @classmethod
    def calc_scores(cls,y_true,y_pred,modes='Regression'):
        # Take y_true and y_pred as input and return the scores
        # modes can be 'Regression', 'Classification'
        # scores for regression: MAE, R2
        # scores for classification: Accuracy, F1, AUC
        assert modes in ['Regression','Classification'], f'Choose a mode from {modes}'
        if modes=='Regression':
            scores = {}
            scores['MAE'] = mean_absolute_error(y_true,y_pred)
            scores['MSE'] = mean_squared_error(y_true,y_pred)
            scores['RMSE'] = np.sqrt(scores['MSE'])
            scores['R2'] = r2_score(y_true,y_pred)
            scores['Pearson_r2'] = pearsonr(y_true,y_pred)[0]**2
            return scores
        print('Not implemented yet')
        return None

    def log_scaffold_cluster(self,use_dc=False):
        # Log the scaffold cluster to logger
        data_key = 'scaffold_cluster'
        self.dataset.scaffold_cluster(use_dc=use_dc)
        self.logger.log_data(data_key,self.dataset.cluster_dict)

    def kfold_split(self,data:torch.utils.data.Dataset,k:int,mode='Random'):
        # Take dataset as input and split into k folds
        # mode can be 'Random', 'Y-value', 'X-cluster'
        modes = ['Random','Y-value','X-cluster']
        assert mode in modes, f'Choose a mode from {modes}'
        if mode=='Random':
            data_length = len(data)
            # random split the index into k fold
            idx_rand = np.random.choice(data_length,data_length,replace=False)
            idx_folds = np.array_split(idx_rand,k)
            return idx_folds
        print('Not implemented yet')
        return None
    
    def time_split(self,data:QSARBaseDataset,seed,ratio=[0.8,0.0,0.2]):
        # Take dataset as input and split into train, validation and test
        # ratio is a list of [train,validation,test]
        assert data.sort_by_date, 'Data should be sorted by date'
        train_data,val_data,test_data = split_dataset(data,seed=seed,ratio=ratio,mode='date')
        return train_data,val_data,test_data

    # def kfold_validation(self):
    #     pass

    # def k_holdout_validation(self):
    #     pass


