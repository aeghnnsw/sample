import os
from copy import deepcopy
import numpy as np
import pickle
import torch
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.ensemble import (RandomForestRegressor,
                              GradientBoostingRegressor,
                              ExtraTreesRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from qsar.dataset import split_dataset,split_dataset_by_index
from qsar.models.basemodel import BaseQSARModel
from qsar.logger import ResultsLogger
import multiprocessing

class SKlearnModel(BaseQSARModel):
    # Base class for all sklearn models, for example, RandomForest, XGBoost, etc.

    def __init__(self,model_dir:str,dataset,k:int=5,mode='Regression'):
        super().__init__(model_dir,dataset,mode)
        self.logger : Optional[ResultsLogger] = None # Initialize in child class
        self.model = None # Initialize in child class
        self.k = k
        self.n_fold = 0
        

    def projection(self,*data):
        if self.dataset.data_type=='GSet':
            return [np.array(d) for d in data]
        elif self.dataset.data_type=='Descriptor_FP':
            whole_data = np.concatenate(data,axis=0)
            scaler = StandardScaler()
            n_samples = len(self.dataset)
            n_components = min(500,n_samples//2)
            pca = PCA(n_components=n_components)
            pipeline = make_pipeline(scaler,pca)
            pipeline.fit(whole_data)
            return [np.array(pipeline.transform(d)) for d in data]
        else:
            return [np.array(d) for d in data]
    
    def transform_sklearn_data(self,dataset:torch.utils.data.Dataset):
        # Transform dataset to sklearn format
        x = [sample[0] for sample in dataset]
        y = [sample[1] for sample in dataset]
        labels = [sample[2] for sample in dataset]
        return x,y,labels
    
    def get_permutation_importance(self,pipeline,x,y):
        # Get permutation importance of a pipeline
        result = permutation_importance(pipeline,x,y,n_repeats=10,n_jobs=-1,\
                                        scoring='neg_mean_squared_error')
        importance_mean = result['importances_mean'] #shape: (n_features,)
        return importance_mean
    
    def analyze_importance(self,importance_mean,topn=10):
        feature_names = self.dataset.get_descriptro_keys()
        topn_feature_index = np.argsort(importance_mean)[-topn:][::-1]
        topn_feature_names = [feature_names[i] for i in topn_feature_index]
        topn_feature_importances = importance_mean[topn_feature_index]
        return topn_feature_names,topn_feature_importances

    def eval(self,train_data,test_data,model,model_name,importance=False):
        # model is a pipeline consists of scaler, and estimator
        train_x,train_y,train_labels = self.transform_sklearn_data(train_data)
        test_x,test_y,test_labels = self.transform_sklearn_data(test_data)
        model.fit(train_x,train_y)
        test_pred = model.predict(test_x)
        train_pred = model.predict(train_x)
        data_dict = self.calc_scores(test_y,test_pred)
        train_proj,test_proj = self.projection(train_x,test_x)
        if importance:
            importance_mean = self.get_permutation_importance(model,test_x,test_y)
            data_dict['feature_importances'] = importance_mean
        train_samples = {}
        train_samples['labels'] = train_labels
        train_samples['y'] = train_y
        train_samples['pred'] = train_pred
        train_samples['proj'] = train_proj
        test_samples = {}
        test_samples['labels'] = test_labels
        test_samples['y'] = test_y
        test_samples['pred'] = test_pred
        test_samples['proj'] = test_proj
        # Get the variance for ensemble models
        if hasattr(self,'n_estimators'):
            test_x_transform = model[0].transform(test_x)
            estimators = model[1].estimators_
            test_preds = []
            for estimator in estimators:
                test_pred_temp = estimator.predict(test_x_transform)
                test_preds.append(test_pred_temp)
            test_preds = np.array(test_preds)
            test_err = np.std(test_preds,axis=0)
            test_samples['pred_err'] = test_err
        else:
            test_samples['pred_err'] = None
        data_dict['train_samples'] = train_samples
        data_dict['test_samples'] = test_samples   
        self.logger.log_data(model_name,data_dict)
    

    def k_split_eval(self,mode:str,importance:bool=False,parallel=False,use_dc=False):
        # Evaluation run for k holdout, scaffold split and time split
        # mode choices: 'holdout','scaffold','time' 
        assert mode in ['holdout','scaffold','time'],f'Choose mode from holdout, scaffold or time'
        models,train_temps,test_temps,model_names = [],[],[],[]
        for i in range(self.k):
            if mode=='holdout':
                train_temp,_,test_temp = split_dataset(self.dataset,[0.8,0,0.2],seed=self.random_seeds[i])
                model_name = f'holdout_{i}'
            elif mode=='scaffold':
                train_idx,_,test_idx = self.dataset.get_scaffold_splits(ratio=[0.8,0,0.2],\
                                                                        seed=self.random_seeds[i],use_dc=use_dc)
                train_temp,test_temp = split_dataset_by_index(self.dataset,train_idx,test_idx)
                model_name = f'scaffold_{i}'
            elif mode=='time':
                train_temp,_,test_temp = split_dataset(self.dataset,[0.8,0,0.2],mode='date',seed=self.random_seeds[i])
                model_name = f'timesplit_{i}'
            model = make_pipeline(StandardScaler(),deepcopy(self.model))
            if parallel:
                models.append(model)
                train_temps.append(train_temp)
                test_temps.append(test_temp)
                model_names.append(model_name)
            else:
                self.eval(train_temp,test_temp,model,model_name,importance=importance)
        if parallel:
            if hasattr(os,'sched_getaffinity'):
                n_cpu = len(os.sched_getaffinity(0))
            else:
                n_cpu = os.cpu_count()
            n_jobs = n_cpu//2
            with multiprocessing.get_context('fork').Pool(n_jobs) as pool:
                pool.starmap(self.eval,zip(train_temps,test_temps,models,model_names,[importance]*self.k))



    def train_final(self):
        model = make_pipeline(StandardScaler(),self.model)
        model_file = 'final_model.pkl'
        model_path = os.path.join(self.model_dir,model_file)
        train_x,train_y,_ = self.transform_sklearn_data(self.dataset)
        model.fit(train_x,train_y)
        with open(model_path,'wb') as f:
            pickle.dump(model,f)

    def prediction(self):
        pass
        # # Load the final model:
        # model_path = os.path.join(self.model_dir,'final_model.pkl')
        # assert os.path.exists(model_path),'No final model has been trained'
        # with open(model_path,'rb') as f:
        #     pipeline = pickle.load(f)
        # pred_x,_,pred_labels = self.transform_sklearn_data(self.dataset)
        # pred = pipeline.predict(pred_x)
        # pred_list = [pred]
        # self.logger.log_pred(pred_labels,pred_list)

class EnsembleModel(SKlearnModel):
    # Ensemble model for regression, include RF, GB and XRT
    def __init__(self,model_dir,dataset,k:int=5,n_estimators:int=128):
        super().__init__(model_dir,dataset,k)
        self.n_estimators = n_estimators

    def prediction(self):
        # Load the final model:
        model_path = os.path.join(self.model_dir,'final_model.pkl')
        assert os.path.exists(model_path),'No final model has been trained'
        with open(model_path,'rb') as f:
            pipeline = pickle.load(f)
        pred_list = []
        pred_x,_,pred_labels = self.transform_sklearn_data(self.dataset)
        scaler = pipeline[0]
        model = pipeline[1]
        for estimator in model.estimators_:
            transform_x = scaler.transform(pred_x)
            pred_temp = estimator.predict(transform_x)
            pred_list.append(pred_temp)
        pred_array = np.array(pred_list)
        pred_mean = np.mean(pred_array,axis=0)
        pred_err = np.std(pred_array,axis=0)
        self.logger.log_pred(pred_labels,pred_mean,pred_err)


class KNNR(SKlearnModel):
    # K Nearest Neighbors Regression
    def __init__(self,model_dir,dataset,k:int=5,n_neighbor:int=5):
        super().__init__(model_dir,dataset,k)
        self.model_name = 'KNNR'
        self.model = KNeighborsRegressor(n_neighbors=n_neighbor)
        logdir = os.path.join(self.model_dir,'log')
        self.logger = ResultsLogger(logdir,self.k,self.n_fold)


class RFR(EnsembleModel):
    # Random Forest model for regression
    def __init__(self,model_dir:str,dataset,k:int=5,n_estimators=128,**kwargs):
        super().__init__(model_dir,dataset,k=k,n_estimators=n_estimators)
        self.model_name = 'RFR'
        self.model = RandomForestRegressor(n_estimators=n_estimators,**kwargs)
        logdir = os.path.join(self.model_dir,'log')
        self.logger = ResultsLogger(logdir,self.k,self.n_fold)

    def hp_tuning(self,dataset:torch.utils.data.Dataset):
        n_estimators = [100,200,500,1000]
        max_depth = [10,20,30,None]
        min_samples_split = [2,3]
        min_samples_leaf = [1,2,3]
        random_grid = {'n_estimators':n_estimators,\
                       'max_depth':max_depth,\
                       'min_samples_split':min_samples_split,\
                       'min_samples_leaf':min_samples_leaf}
        rf = RandomForestRegressor()
        scaler = StandardScaler()
        rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,\
                                       n_iter=50,cv=5,verbose=0,n_jobs=-1,\
                                       scoring='neg_mean_squared_error')
        pipeline = make_pipeline(scaler,rf_random)
        x,y,_ = self.transform_sklearn_data(dataset)
        pipeline.fit(x,y)
        params = rf_random.best_params_
        self.model = RandomForestRegressor(**params)
        return params

class GBR(EnsembleModel):
    # XGBoost model for regression
    def __init__(self,model_dir:str,dataset,k:int=5,n_estimators:int=128,**kwargs):
        super().__init__(model_dir,dataset,k=k,n_estimators=n_estimators)
        self.model_name = 'GBR'
        self.model = GradientBoostingRegressor(n_estimators=n_estimators,**kwargs)
        logdir = os.path.join(self.model_dir,'log')
        self.logger = ResultsLogger(logdir,self.k,self.n_fold)

class XRTR(EnsembleModel):
    # Extremely Randomized Trees model for regression
    def __init__(self,model_dir:str,dataset,k:int=5,n_estimators:int=128,**kwargs):
        super().__init__(model_dir,dataset,k=k,n_estimators=n_estimators)
        self.model_name = 'XRTR'
        self.model = ExtraTreesRegressor(**kwargs)
        logdir = os.path.join(self.model_dir,'log')
        self.logger = ResultsLogger(logdir,self.k,self.n_fold)
