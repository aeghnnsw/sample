import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import kdeplot,boxplot,stripplot
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from qsar.dataset import QSARBaseDataset
import pickle
import time
from glob import glob
import umap
import csv
from typing import Optional
from bokeh.io import output_file
from bokeh.plotting import figure, ColumnDataSource, save
from bokeh.models import HoverTool,LinearColorMapper
from bokeh.palettes import Turbo256,Category20c
from qsar.models.basemodel import BaseQSARModel
from rdkit import Chem
from rdkit.Chem import Draw
import multiprocessing
from qsar.uncertainty import plot_rmse_binned_err,calc_miscalibration_area
from copy import deepcopy
# import sys

class ResultsLogger():
    # Save runing details for each individual experiments
    # Store all raw data in a diction
    #       Key:    experiment's name
    #       value:  Diction with keys property names, 
    #               for sklearn model also has 'top_features' and 'top_importances'            
    #               'train_samples', 'test_samples'
    #       train_samples and test_samples are dict with keys: 'labels','y','pred','proj'

    def __init__(self,log_dir:str,k,n,load=False):
        # init ResultsLogger
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.raw_data_dir = os.path.join(self.log_dir,'raw_data')
        os.makedirs(self.raw_data_dir,exist_ok=True)
        self.pred_dir = os.path.join(self.log_dir,'pred')
        os.makedirs(self.pred_dir,exist_ok=True)
        os.chdir(log_dir)
        self.log_file = os.path.join(log_dir,'results_log.html')
        # self.log_data_file = os.path.join(log_dir,'results_data.pkl')
        self.k = k
        self.n_fold = n
        self.tables = {}
        # self.hidden_tables = {}
        self.data = {}
        if load:
            self.load()
        if 'scaffold_cluster' in self.data.keys():
            self.scaffold_cluster = self.data['scaffold_cluster']
        else:
            self.scaffold_cluster = None

    def aggregate_tables(self,exp:str):
        # aggregate results from n-fold tables, only used for n_fold>0
        # create figures for box_plot for each run (n_fold points)
        assert self.n_fold > 0, 'n_fold should be larger than 0'
        def box_plot(key:str,box_plot=True):
            # get k * n_fold data for each key
            data_list = []
            for i in range(self.k):
                table_name = f'{exp}_{i}'
                data_list.append(self.tables[table_name][key].values)
            # data_list = np.array(data_list)
            # print(data_list)
            # data_column = np.mean(data_list,axis=1)
            # print(data_column)
            # create box plot if box_plot is True
            if box_plot:
                fig_name = f'{exp}_{key}_box_plot.png'
                plt.figure(figsize=(10,8))
                ax = boxplot(data=data_list,meanline=True,showmeans=True)
                stripplot(data_list)
                for patch in ax.patches:
                    fc = patch.get_facecolor()
                    patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.3))
                fig_log = self.log_image(fig_name)
                ax.set_xlabel('Fold')
                ax.set_ylabel(key)
                plt.savefig(fig_name)
                plt.close()
            else:
                fig_log=''
            return fig_log
        def aggregate_test_xys():
            for i in range(self.k):
                test_preds = []
                test_preds_var = []
                for j in range(self.n_fold):
                    key = f'{exp}_{i}_{j}'
                    test_preds.append(self.data[key]['test_samples']['pred'])
                    if 'pred_var' not in self.data[key]['test_samples'].keys():
                        continue
                    elif self.data[key]['test_samples']['pred_var'] is not None:
                        test_preds_var.append(self.data[key]['test_samples']['pred_var'])
                test_preds = np.array(test_preds)
                test_y = self.data[key]['test_samples']['y']
                test_labels = self.data[key]['test_samples']['labels']
                test_pred_mean = np.mean(test_preds,axis=0)
                if len(test_preds_var) == 0:
                    test_pred_err = np.std(test_preds,axis=0)
                else:
                    test_preds_var = np.array(test_preds_var)
                    var_temp = np.mean(test_preds_var,axis=0)+np.mean(np.power(test_preds,2),axis=0)-test_pred_mean**2
                    test_pred_err = np.sqrt(var_temp)
                index = f'{exp}_{i}'
                # save aggregated test pred to the data dict
                self.data[index] = {}
                self.data[index]['test_samples'] = {}
                self.data[index]['test_samples']['pred'] = test_pred_mean
                self.data[index]['test_samples']['pred_err'] = test_pred_err
                self.data[index]['test_samples']['y'] = test_y
                self.data[index]['test_samples']['labels'] = test_labels
                # calc average miscalibration area and 90% confidence miscaliabration score
                ama,m_90 = calc_miscalibration_area(test_y,test_pred_mean,test_pred_err)
                self.data[index]['test_samples']['AMA'] = ama
                self.data[index]['test_samples']['m_90'] = m_90
                pred_table_temp = self.aggregate_test_xy(test_y,test_pred_mean,test_pred_err,index,test_labels)
                if i==0:
                    pred_table = pred_table_temp
                else:
                    pred_table = pd.concat((pred_table,pred_table_temp))
            return pred_table

        mae_log = box_plot('MAE')
        rmse_log = box_plot('RMSE')
        r2_log = box_plot('R2')
        r2_pearon_log = box_plot('Pearson_r2')
        box_plot_row = [mae_log,rmse_log,r2_log,r2_pearon_log]
        box_plot_columns = ['MAE','RMSE','R2','Pearson_r2']
        box_plot_index = [exp]
        box_plot_df = pd.DataFrame([box_plot_row],index=box_plot_index,columns=box_plot_columns)
        if 'box_plot' not in self.tables.keys():
            self.tables['box_plot'] = box_plot_df
        else:
            self.tables['box_plot'] = pd.concat((self.tables['box_plot'],box_plot_df))
        pred_table = aggregate_test_xys()
        # calculate mean and std for first 4 columns
        pred_table = self.get_mean_std(pred_table)
        return pred_table

    def aggregate_test_xy(self,test_y,test_pred_mean,test_pred_err,index,test_labels):
        # Aggregate test data and plot xy figure
        columns = ['MAE','RMSE','R2','Pearson_r2','XY_Plot','RMSE_uncertatinty']
        scores = BaseQSARModel.calc_scores(test_y,test_pred_mean)
        xy_fig_path = f'{index}_xy.html'
        # self.plot_xy(test_pred_mean,test_y,xy_fig_path,test_error=test_pred_error)
        self.plot_xy_bokeh(test_pred_mean,test_y,xy_fig_path,cluster_dict=self.scaffold_cluster,test_labels=test_labels,\
                           test_error=test_pred_err)
        xy_fig_log = self.log_bokeh_image(xy_fig_path)
        # plot RMSE_binned_err plot
        rmse_bin_err_fig_path = f'{index}_rmse_bin_err.png'
        rmse_bin_err_fig_log = self.log_image(rmse_bin_err_fig_path)
        plot_rmse_binned_err(test_y,test_pred_mean,test_pred_err,rmse_bin_err_fig_path)
        row = [scores['MAE'],scores['RMSE'],scores['R2'],scores['Pearson_r2'],xy_fig_log,rmse_bin_err_fig_log]
        df_temp = pd.DataFrame([row],index=[index],columns=columns)
        return df_temp

    def create_tables(self,exp:str,plot_xy=True,plot_williams=True,\
                      plot_umap=True,plot_uncertainty=True,color_by_cluster=True,simple=True):
        # find all experiments and organzie into different lists
        # see kwargs in process_data
        # 3 different experiments:
        #   1. random holdout splits:   'holdout_i' for sklearn_model
        #                               'houldout_i_j' for torch model
        #   2. random scaffold splits:  'scaffold_i' for sklearn_model
        #                               'scaffold_i_j' for torch model
        #   3. time splits:             'timesplit_i' for sklearn_model
        #                               'timesplit_i_j' for torch model
        # Table is used to write log file
        # create table for each experiment
        # If set simple, only plot one figure for n_folds experiments
        os.chdir(self.log_dir)
        self.init_plot_params()
        assert exp in ['holdout','scaffold','timesplit']
        if hasattr(os,'sched_getaffinity'):
            n_cpu = len(os.sched_getaffinity(0))
        else:
            n_cpu = os.cpu_count()
        n_jobs = n_cpu//2
        if exp == 'holdout' or  exp == 'scaffold' or exp == 'timesplit':
            if self.n_fold>0:
                for i in range(self.k):
                        # create table for n_fold experiments
                        table_name = f'{exp}_{i}'
                        task_list = []
                        for j in range(self.n_fold):
                            if simple and j>0:
                                task_list.append([f'{exp}_{i}_{j}',False,False,False,False,color_by_cluster,simple])
                            else:
                                task_list.append([f'{exp}_{i}_{j}',plot_xy,False,plot_umap,plot_uncertainty,color_by_cluster,simple])
                        with multiprocessing.get_context('fork').Pool(n_jobs) as pool:
                            logger_rows = pool.starmap(self.process_data,task_list)
                        # print(logger_rows)
                        self.tables[table_name] = pd.concat(logger_rows)
                table_name = f'{exp}'
                self.tables[table_name] = self.aggregate_tables(exp)
                # Add extra plots columns to the table
                # if plot_williams:
                #     williams_cols = []
                #     for i in range(self.k):
                #         fig_name = f'{exp}_{i}_0_williams.html'
                #         fig_info = self.log_bokeh_image(fig_name)
                #         williams_cols.append(fig_info)
                #     williams_cols.append('')
                #     williams_cols.append('')
                #     self.tables[table_name]['Williams_Plot'] = williams_cols
                if plot_umap:
                    umap_cols = []
                    fig_name_temp = f'{exp}_0_0_umap.html'
                    if os.path.exists(fig_name_temp):
                        for i in range(self.k):
                            fig_name = f'{exp}_{i}_0_umap.html'
                            fig_info = self.log_bokeh_image(fig_name)
                            umap_cols.append(fig_info)
                        umap_cols.append('')
                        umap_cols.append('')
                        self.tables[table_name]['UMAP_Plot'] = umap_cols
            else:
                table_name = f'{exp}'
                task_list = []
                for i in range(self.k):
                    self.calc_uncertainty(f'{exp}_{i}')
                    task_list.append([f'{exp}_{i}',plot_xy,plot_williams,plot_umap,plot_uncertainty,color_by_cluster])
                with multiprocessing.get_context('fork').Pool(n_jobs) as pool:
                    logger_rows = pool.starmap(self.process_data,task_list)
                self.tables[table_name] = pd.concat(logger_rows)
        # elif exp == 'timesplit':
        #     table_name = f'{exp}'
        #     if self.n_fold>0:
        #         test_preds = []
        #         task_list = []
        #         for j in range(self.n_fold):
        #             task_list.append(f'{exp}_{j}')
        #             test_preds.append(self.data[f'{exp}_{j}']['test_samples']['pred'])
        #         with multiprocessing.get_context('spawn').Pool(n_jobs) as pool:
        #             logger_rows = pool.map(self.process_data,task_list)
        #         # table_temp = self.get_mean_std(table_temp)
        #         self.tables['timesplit_details'] = pd.concat(logger_rows)
        #         test_y = self.data[f'{exp}_{j}']['test_samples']['y']
        #         index = f'{exp}'
        #         test_labels = self.data[f'{exp}_{j}']['test_samples']['labels']
        #         pred_table = self.aggregate_test_xy(test_y,test_preds,index,test_labels)
        #         # print(pred_table)
        #         # print(pred_table['R2'])
        #         self.tables[table_name] = pred_table
        #     else:
        #         logger_row = self.process_data(f'{exp}')
        #         self.tables[table_name] = logger_row
        return None
    
    @classmethod
    def get_mean_std(cls,table:pd.DataFrame):
        # get mean and std for each column in table
        describe = table.describe()
        table = pd.concat((table,describe[1:3]))
        return table
    
    @classmethod
    def init_plot_params(cls):
        # Initialize the plot parameters
        size1 = 16
        size2 = 20
        size3 = 16
        plt.rc('font',family='serif')
        plt.rc('axes',titlesize=size1)
        plt.rc('axes',titleweight='bold')
        plt.rc('axes',labelsize=size2)

        plt.rc('xtick',labelsize=size2)
        plt.rc('ytick',labelsize=size2)

        plt.rc('legend',fontsize=size3)
        #plt.rc('legend',markerscale=1)
        plt.rc('legend',borderpad=0.1)
        matplotlib.rcParams['figure.dpi']=100
        matplotlib.rcParams['savefig.dpi']=100    

    def load(self):
        # load data
        # read all raw data files in 
        self.data = {}
        file_list = glob(self.raw_data_dir+'/*.pkl')
        # print(file_list)
        for file in file_list:
            with open(file,'rb') as f:
                data_temp = pickle.load(f)
                self.data.update(data_temp)
        return None
    
    @classmethod
    def log_bokeh_image(cls,image:str,width=380,height=380):
        return f'<embed type="text/html" src="{image}" width="{width}px" height="{height}px">\n</embed>' 
      
    def log_data(self,model_name,data):
        # Add data to data dict
        log_data_file = os.path.join(self.raw_data_dir,f'{model_name}.pkl')
        f = open(log_data_file,'wb')
        data = {model_name:data}
        pickle.dump(data,f)
        f.close()  
    
    @classmethod
    def log_feature_importance(cls,features,importances):
        # Convert descriptors and importance to html style
        log_info = ''
        for feature,importance in zip(features,importances):
            log_info += f'{feature}: {importance:.3f}<br>'
        return log_info
    
    @classmethod
    def log_image(cls,image:str):
        # Convert image name to html image block
        return f'<img src=\"{image}\" style=\"width:auto;height:350px\">'


    def log_pred(self,labels,pred_values,pred_err,use_eval=False):
        if use_eval:
            pred_file = os.path.join(self.pred_dir,'pred_eval_model.pkl')
        else:
            pred_file = os.path.join(self.pred_dir,'pred.pkl')
        f = open(pred_file,'wb')
        data = {'labels':labels,'pred_values':pred_values,'pred_err':pred_err}
        pickle.dump(data,f)
        f.close()

    def plot_mol_structures(self,dataset:QSARBaseDataset):
        self.mol_plots = os.path.join(self.log_dir,'mol_plots')
        os.makedirs(self.mol_plots,exist_ok=True)
        # check if the mol_plots has been created
        raw_samples = dataset.raw_samples
        n_samples = len(raw_samples)
        plots_files = os.path.join(self.mol_plots,'*.png')
        n_plots = glob(plots_files)
        if len(n_plots)==n_samples:
            return None
        for sample in raw_samples:
            fig_file = os.path.join(self.mol_plots,f'{sample["label"]}.png')
            mol = Chem.MolFromSmiles(sample['SMILES'])
            Draw.MolToFile(mol,fig_file)
        return None

    def plot_kde(self,train_x,test_x,save_path):
        # Plot the kde plot of train and test data on first 2 PC space
        pca = PCA(n_components=2)
        whole_data = np.vstack((train_x,test_x))
        pca.fit(whole_data)
        train_data = pca.transform(train_x)
        test_data = pca.transform(test_x)
        train_data_list = [[i[0],i[1],'train'] for i in train_data]
        test_data_list = [[i[0],i[1],'test'] for i in test_data]
        data = train_data_list+test_data_list
        data = pd.DataFrame(data,columns=['PC1','PC2','type'])

        fig = plt.figure(figsize=(10,8))
        kdeplot(data,x='PC1',y='PC2',hue='type',fill=True,palette=['b','r'],alpha=0.3)
        plt.scatter(train_data[:,0],train_data[:,1],c='b',label='Train',s=6)
        plt.scatter(test_data[:,0],test_data[:,1],c='r',label='Test',s=6)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(save_path)
        plt.close(fig)

    def plot_umap(self,train_proj,test_proj,save_path):
        # project train and test onto umap space
        reducer = umap.UMAP(n_neighbors=15,min_dist=0.4,n_components=2)
        whole_data = np.vstack((train_proj,test_proj))
        reducer.fit(whole_data)
        train_reduced = reducer.transform(train_proj)
        test_reduced = reducer.transform(test_proj)
        fig = plt.figure(figsize=(10,8))
        plt.scatter(train_reduced[:,0],train_reduced[:,1],c='b',label='Train',s=8)
        plt.scatter(test_reduced[:,0],test_reduced[:,1],c='r',label='Test',s=8)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.legend()
        plt.savefig(save_path)
        plt.close(fig)

    def plot_umap_bokeh(self,train_proj,test_proj,save_path,cluster_dict:Optional[dict]=None,\
                        train_labels:Optional[dict]=None,test_labels:Optional[dict]=None,color_by_cluster:bool=True):
        # Umap plot from bokeh
        colors = Category20c[20]
        reducer = umap.UMAP(n_neighbors=15,min_dist=0.4,n_components=2)
        whole_data = np.vstack((train_proj,test_proj))
        reducer.fit(whole_data)
        train_reduced = reducer.transform(train_proj)
        test_reduced = reducer.transform(test_proj)
        train_x = train_reduced[:,0]
        train_y = train_reduced[:,1]
        train_imgs =[f'./mol_plots/{label}.png' for label in train_labels]
        train_cluster = [cluster_dict[label] for label in train_labels]
        train_color = [colors[i] if i<19 else colors[19] for i in train_cluster]
        train = ['train' for i in range(len(train_x))]
        # train_color = [cluster_dict[i] for i in train_labels]
        test_x = test_reduced[:,0]
        test_y = test_reduced[:,1]
        test_imgs =[f'./mol_plots/{label}.png' for label in test_labels]
        test_cluster = [cluster_dict[label] for label in test_labels]
        test_color = [colors[i] if i<19 else colors[19] for i in test_cluster]
        test = ['test' for i in range(len(test_x))]
        # test_color = [cluster_dict[i] for i in test_labels]
        # min_color = min(train_color+test_color)
        # max_color = max(train_color+test_color)
        # mapper = LinearColorMapper(palette=Turbo256,low=min_color,high=max_color)
        source_train = ColumnDataSource(data=dict(x=train_x,y=train_y,imgs=train_imgs,label=train_labels,\
                                                  cluster=train_cluster,color=train_color,type=train))
        source_test = ColumnDataSource(data=dict(x=test_x,y=test_y,imgs=test_imgs,label=test_labels,\
                                                 cluster=test_cluster,color=test_color,type=test))
        output_file(save_path)
        hover = HoverTool(tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="80" alt="@imgs" width="80"
                ></img>
            </div>
            <div>
                <span style="font-size: 12px;font-weight: bold;">@label  Cluster:@cluster  @type</span>
            </div>
        </div>
        """)    
        if color_by_cluster:
            p1 = figure(width=350,height=350,x_axis_label='UMAP1',y_axis_label='UMAP2')
            source_train_cp = deepcopy(source_train)
            source_test_cp = deepcopy(source_test)
            p1.circle('x','y',source=source_train_cp,color='color',\
                        size=3,line_color=None)
            p1.square('x','y',source=source_test_cp,color='color',\
                        size=2.5,alpha=0.9,line_color='black',line_width=0.1)
            hover1 = deepcopy(hover)
            p1.add_tools(hover1)
            save(p1)
            save_path = save_path.replace('.html','_nc.html')
            output_file(save_path)
        p = figure(width=350,height=350,x_axis_label='UMAP1',y_axis_label='UMAP2')
        p.circle('x','y',source=source_train,fill_color='steelblue',\
                size=3,line_color=None,alpha=0.9)
        p.square('x','y',source=source_test,fill_color='firebrick',\
                size=2.5,alpha=0.9,line_color='black',line_width=0.1)
        p.add_tools(hover)
        save(p)
        return None
    
    def plot_williams(self,train_proj,train_y,train_pred,\
                      test_proj,test_y,test_pred,save_path):
        # Calculate leverage for train and test data from train_proj and test_proj
        train_X = train_proj
        test_X = test_proj
        # prevent singular matrix
        train_X += np.random.normal(0,1e-3,train_X.shape)
        # Shape (n_samples,n_samples)
        train_X_leverage = train_X @ np.linalg.inv(train_X.T @ train_X) @ train_X.T 

        test_X_leverage = test_X @ np.linalg.inv(train_X.T @ train_X) @ test_X.T

        train_X_leverage = np.diag(train_X_leverage) # Shape (n_samples,)
        test_X_leverage = np.diag(test_X_leverage)

        h_threshold = 3*(train_X.shape[1]+1)/train_X.shape[0] # Shape (1,)
        # h_threshold = 3*np.sum(train_X_leverage)/train_X.shape[0] # Shape (1,)

        # Calculate the standardized residual for train and test data
        train_residual = train_y - train_pred
        test_residual = test_y - test_pred
        rmse = np.sqrt(mean_squared_error(train_y,train_pred))
        train_standardized_residual = train_residual/rmse
        test_standardized_residual = test_residual/rmse
        fig = plt.figure(figsize=(8,8))
        plt.scatter(train_X_leverage,train_standardized_residual,c='steelblue',label='Train',s=5)
        plt.scatter(test_X_leverage,test_standardized_residual,c='firebrick',label='Test',s=5)
        plt.axhline(y=3,c='black',ls='--')
        plt.axhline(y=-3,c='black',ls='--')
        plt.axvline(x=h_threshold,c='black',ls='--')
        plt.xlabel('Leverage')
        plt.ylabel('Standardized Residual')
        plt.legend()
        plt.xlim(0,3)
        plt.savefig(save_path)
        plt.close(fig)

    def plot_williams_bokeh(self,train_proj,train_y,train_pred,train_labels,\
                            test_proj,test_y,test_pred,test_labels,save_path):
        train_X = train_proj
        test_X = test_proj
        # prevent singular matrix
        # train_X += np.random.normal(0,1e-3,train_X.shape)
        # Shape (n_samples,n_samples)
        # whole_X = np.concatenate((train_X,test_X),axis=0)
        try:
            train_X_leverage = train_X @ np.linalg.inv(train_X.T @ train_X) @ train_X.T 
            test_X_leverage = test_X @ np.linalg.inv(train_X.T @ train_X) @ test_X.T

        except:
            train_X_new = train_X + 0.001*np.random.rand(*train_X.shape)
            train_X_leverage = train_X_new @ np.linalg.inv(train_X_new.T @ train_X_new) @ train_X_new.T 
            test_X_leverage = test_X @ np.linalg.inv(train_X_new.T @ train_X_new) @ test_X.T

        train_X_leverage = np.diag(train_X_leverage) # Shape (n_samples,)
        test_X_leverage = np.diag(test_X_leverage)

        h_threshold = 3*(train_X.shape[1]+1)/train_X.shape[0] # Shape (1,)
        # h_threshold = 3*np.sum(train_X_leverage)/train_X.shape[0] # Shape (1,)

        # Calculate the standardized residual for train and test data
        train_residual = train_y - train_pred
        test_residual = test_y - test_pred
        rmse = np.sqrt(mean_squared_error(train_y,train_pred))
        train_standardized_residual = train_residual/rmse
        test_standardized_residual = test_residual/rmse
        output_file(save_path)
        p=figure(width=350,height=350,x_axis_label='Leverage',y_axis_label='Standardized Residual')
        train_clusters = [self.scaffold_cluster[label] for label in train_labels]
        test_clusters = [self.scaffold_cluster[label] for label in test_labels]
        train_imgs = [f'./mol_plots/{label}.png' for label in train_labels]
        test_imgs = [f'./mol_plots/{label}.png' for label in test_labels]
        source_train = ColumnDataSource(data=dict(x=train_X_leverage,y=train_standardized_residual,\
                                                labels=train_labels,clusters=train_clusters,imgs=train_imgs))
        source_test = ColumnDataSource(data=dict(x=test_X_leverage,y=test_standardized_residual,\
                                                labels=test_labels,clusters=test_clusters,imgs=test_imgs))

        s1 = p.circle('x','y',source=source_train,size=2.5,color='steelblue',alpha=0.8)
        s2 = p.circle('x','y',source=source_test,size=2.5,color='firebrick',alpha=0.8)
        hover1 = HoverTool(tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="80" alt="@imgs" width="80"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 12px;font-weight: bold;">@labels Cluster: @clusters</span>
                </div>
            </div>
            """,renderers=[s1,s2])
        p.line([0,10],[-3,-3],line_dash='dashed',line_color='black')
        p.line([0,10],[3,3],line_dash='dashed',line_color='black')
        p.line([h_threshold,h_threshold],[-10,10],line_dash='dashed',line_color='black')
        p.add_tools(hover1)
        save(p)
        return None

    def plot_xy(self,test_pred,test_y,save_path,test_error=None,train_pred=None,train_y=None):
        fig = plt.figure(figsize=(8,8))
        if test_error is not None:
            plt.errorbar(test_y,test_pred,yerr=test_error,fmt='none',c='black',elinewidth=0.5)
        plt.scatter(test_y,test_pred,c='r',label='Test',s=8)
        # linear regression of train data      
        test_pred,test_y = np.array(test_pred),np.array(test_y)
        # train_pred = train_pred.reshape(-1,1)
        # train_y = train_y.reshape(-1,1)
        if train_pred is not None and train_y is not None:
            plt.scatter(train_y,train_pred,c='b',label='Train',s=8)
            train_pred,train_y = np.array(train_pred),np.array(train_y)
        x_min = min(min(test_pred),min(test_y))-1
        x_max = max(max(test_pred),max(test_y))+1
        plt.xlim(x_min,x_max)
        plt.ylim(x_min,x_max)
        m, b = np.polyfit(test_y, test_pred, deg=1)
        plt.axline((0,b),slope=m,c='black',ls='--')
        plt.plot([x_min,x_max],[x_min,x_max],c='black',ls='--')
        plt.xlabel('Experimental Value')
        plt.ylabel('Predicted Value')
        plt.legend()
        plt.savefig(save_path)
        plt.close(fig)

    def plot_xy_bokeh(self,test_pred,test_y,save_path,test_error=None,\
                      cluster_dict:Optional[dict]=None,test_labels:Optional[dict]=None):
        output_file(save_path)
        def error_lines(xs,ys,yerrs):
            x_lines = []
            y_lines = []
            for x,y,yerr in zip(xs,ys,yerrs):
                x_lines.append((x,x))
                y_lines.append((y-yerr,y+yerr))
            return x_lines,y_lines
        p = figure(width=350,height=350,x_axis_label='Experimental Value',\
                   y_axis_label='Predicted Value')
        if cluster_dict is not None and test_labels is not None:
            # plot each cluster with different color
            xs = [x for x in test_y]
            ys = [y for y in test_pred]
            mol_images = [f'./mol_plots/{label}.png' for label in test_labels]
            test_clusters = [cluster_dict[label] for label in test_labels]
            colors = Category20c[20]
            test_colors = [colors[i] if i<19 else colors[19] for i in test_clusters]
            data_dict = {'x':xs,'y':ys,'color':test_colors,'imgs':mol_images,'label':test_labels,'clusters':test_clusters}
            source = ColumnDataSource(data=data_dict)
            s1 = p.circle('x','y',fill_color='color',source=source,\
                     line_color='black',size=5,line_width=0.1)            
            hover1 = HoverTool(tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="80" alt="@imgs" width="80"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 12px;font-weight: bold;">@label Cluster: @clusters</span>
                </div>
            </div>
            """,renderers=[s1])
            p.add_tools(hover1)
        else:
            p.circle(test_y,test_pred,line_color='white',fill_color='red',size=5,alpha=0.2)
        if test_error is not None:
            xs_err,ys_err = error_lines(test_y,test_pred,test_error)
            ml = p.multi_line(xs_err,ys_err,line_color='black',hover_line_color='red',\
                         line_width=1,line_alpha=0.2,hover_alpha=1,name='error')
            hover2 = HoverTool(renderers=[ml],tooltips=None)
            p.add_tools(hover2)
        m, b = np.polyfit(test_y, test_pred, deg=1)
        p.line([min(test_y)-1,max(test_y)+1],\
               [m*(min(test_y)-1)+b,m*(max(test_y)+1)+b],\
                line_color='red',line_dash='dashed')
        p.line([min(test_y)-1,max(test_y)+1],[min(test_y)-1,max(test_y)+1],\
                line_color='black')
        save(p)
        return None

    def calc_uncertainty(self,key):
        data_point = self.data[key]
        test_y = data_point['test_samples']['y']
        test_pred = data_point['test_samples']['pred']
        if 'pred_err' in data_point['test_samples'].keys():
            test_err = data_point['test_samples']['pred_err']
            # calculate average miscalibration error and 90% miscaliibration score
            if test_err is not None:
                ama,m_90 = calc_miscalibration_area(test_y,test_pred,test_err)
                self.data[key]['test_samples']['AMA'] = ama
                self.data[key]['test_samples']['m_90'] = m_90

    def process_data(self,key,plot_xy=True,plot_williams=True,\
                     plot_umap=True,plot_uncertainty=True,color_by_cluster=True,simple=False):
        # if simple is True, don't add plots information to tables
        if key not in self.data.keys():
            # print(f'key {key} not in in data keys {self.data.keys()}')
            return None
        data_point = self.data[key]
        mae = data_point['MAE']
        rmse = data_point['RMSE']
        r2 = data_point['R2']
        r2_pearson = data_point['Pearson_r2']                        
        train_y = data_point['train_samples']['y']
        train_pred = data_point['train_samples']['pred']
        train_labels = data_point['train_samples']['labels']
        test_y = data_point['test_samples']['y']
        test_pred = data_point['test_samples']['pred']
        test_labels = data_point['test_samples']['labels']
        if 'pred_err' in data_point['test_samples'].keys():
            test_err = data_point['test_samples']['pred_err']
            # calculate average miscalibration error and 90% miscaliibration score
        else:
            test_err = None
        # log feature importance
        # if feature_importance and 'top_features' in data_point.keys():
        #     top_features = data_point['top_features']
        #     top_importances = data_point['top_importances']
        #     fi_log = self.log_feature_importance(top_features,top_importances)
        # else:
        #     fi_log = ''
        # plot xy figure
        if plot_xy:
            # xy_fig_path = f'{key}_xy.png'
            # self.plot_xy(test_pred,test_y,xy_fig_path,train_pred=train_pred,train_y=train_y)
            # xy_fig_log = self.log_image(xy_fig_path)
            xy_fig_path = f'{key}_xy.html'
            if not os.path.exists(xy_fig_path):
                self.plot_xy_bokeh(test_pred,test_y,xy_fig_path,test_error=test_err,cluster_dict=self.scaffold_cluster,\
                                   test_labels=test_labels)
            xy_fig_log = self.log_bokeh_image(xy_fig_path)
        else:
            xy_fig_log = ''
        # plot williams figure
        if plot_williams and 'proj' in data_point['train_samples'].keys():
            train_proj = data_point['train_samples']['proj']
            test_proj = data_point['test_samples']['proj']
            williams_fig_path = f'{key}_williams.html'
            if not os.path.exists(williams_fig_path):
                # self.plot_williams(train_proj,train_y,train_pred,\
                #                     test_proj,test_y,test_pred,williams_fig_path)
                self.plot_williams_bokeh(train_proj,train_y,train_pred,train_labels,\
                                         test_proj,test_y,test_pred,test_labels,williams_fig_path)
            williams_fig_log = self.log_bokeh_image(williams_fig_path)
        else:
            williams_fig_log = ''
        # plot umap figure
        if plot_umap and 'proj' in data_point['train_samples'].keys():
            # umap_fig_path = f'{key}_umap.png'
            # train_proj = data_point['train_samples']['proj']
            # test_proj = data_point['test_samples']['proj']
            # self.plot_umap(train_proj,test_proj,umap_fig_path)
            # umap_fig_log = self.log_image(umap_fig_path)
            umap_fig_path = f'{key}_umap.html'
            train_proj = data_point['train_samples']['proj']
            test_proj = data_point['test_samples']['proj']
            if not os.path.exists(umap_fig_path):
                self.plot_umap_bokeh(train_proj,test_proj,umap_fig_path,\
                                     cluster_dict=self.scaffold_cluster,\
                                     test_labels=test_labels,train_labels=train_labels,\
                                     color_by_cluster=color_by_cluster)
            umap_fig_log = self.log_bokeh_image(umap_fig_path)
        else:
            umap_fig_log = ''
        # plot rmse binned uncertainty figure
        if test_err is not None and plot_uncertainty:
            rmse_bin_err_fig_path = f'{key}_rmse_bin_err.png'
            plot_rmse_binned_err(test_y,test_pred,test_err,rmse_bin_err_fig_path)
            rmse_bin_err_fig_log = self.log_image(rmse_bin_err_fig_path)
        else:
            rmse_bin_err_fig_log = ''
        # plot kde figure
        # if plot_kde and 'proj' in data_point['train_samples']:
        #     kde_fig_path = f'{key}_kde.png'
        #     train_proj = data_point['train_samples']['proj']
        #     test_proj = data_point['test_samples']['proj']
        #     self.plot_kde(train_proj,test_proj,kde_fig_path)
        #     kde_fig_log = self.log_image(kde_fig_path)
        # else:
        #     kde_fig_log = ''
        if simple:
            row = [mae,rmse,r2,r2_pearson]
            columns = ['MAE','RMSE','R2','Pearson_r2']
        else:
            row = [mae,rmse,r2,r2_pearson,xy_fig_log,williams_fig_log,umap_fig_log,rmse_bin_err_fig_log]
            columns = ['MAE','RMSE','R2','Pearson_r2','XY_Plot',\
                       'Williams_Plot','UMAP_plot','RMSE_uncertainty']
        index = [key]
        data_row = pd.DataFrame([row],index=index,columns=columns)
        return data_row

    def process_pred_results(self,train_data:QSARBaseDataset,test_data:QSARBaseDataset):
        # write the predicted results to a final csv file
        # Columns: GNumber, SMILES, pred1_mean, pred1_std, pred2_mean, pred2_std, in_train, train_value
        # pred1: predicted value from final models
        # pred2: predicted value from evaluation models
        pred_dict = {}
        train_raw = train_data.raw_data_to_dict()
        test_raw = test_data.raw_data_to_dict()
        pred_final = os.path.join(self.pred_dir,'pred.pkl')
        if os.path.exists(pred_final):
            f1 = open(pred_final,'rb')
            preds1 = pickle.load(f1)
            f1.close()
            labels1 = preds1['labels'] # length: n_samples
            values1 = preds1['pred_values']
            err1 = preds1['pred_err']
            for label,mean,err in zip(labels1,values1,err1):
                pred_dict[label]={}
                pred_dict[label]['label'] = label
                pred_dict[label]['pred_mean'] = mean
                pred_dict[label]['pred_err'] = err
        # pred_eval = os.path.join(self.pred_dir,'pred_eval_model.pkl')
        # if os.path.exists(pred_eval):
        #     f2 = open(pred_eval,'rb')
        #     preds2 = pickle.load(f2)
        #     f2.close()
        #     labels2 = preds2['labels'] # length: n_samples
        #     values2 = np.array(preds2['pred_values']) # shape: n_preds * n_samples
        #     if log:
        #         values2 = np.exp(values2)
        #     mean2 = np.mean(values2,axis=0)
        #     std2 = np.std(values2,axis=0)
        #     for label,mean,std in zip(labels2,mean2,std2):
        #         if label not in pred_dict.keys():
        #             pred_dict[label]={}
        #             pred_dict[label]['GNumber'] = label
        #         pred_dict[label]['pred2_mean'] = mean
        #         pred_dict[label]['pred2_std'] = std
        for label in pred_dict.keys():
            if label in train_raw.keys():
                pred_dict[label]['in_train'] = True
                pred_dict[label]['train_value'] = train_raw[label][train_data.end_point_name]
            else:
                pred_dict[label]['in_train'] = False
                pred_dict[label]['train_value'] = None
            pred_dict[label]['SMILES'] = test_raw[label]['SMILES']
        f_final_csv = os.path.join(self.pred_dir,'pred.csv')
        with open(f_final_csv,'w') as f:
            writer = csv.DictWriter(f,fieldnames=pred_dict[label].keys())
            writer.writeheader()
            for key,value in pred_dict.items():
                writer.writerow(value)
        return pred_dict

    @classmethod    
    def write_table(cls,df:pd.DataFrame):
        # write dataframe to html style table
        loginfo = ''
        # Write headers
        loginfo = '<table>\n<tr>\n'
        loginfo += '<th>Index</th>\n'
        for col in df.columns:
            loginfo += f'<th>{col}</th>\n'
        loginfo += '</tr>\n'
        # Write data
        for idx,row in df.iterrows():
            loginfo += '<tr>\n'
            loginfo += f'<td>{idx}</td>\n'
            for col in df.columns:
                value = row[col]
                if isinstance(value,float):
                    loginfo += f'<td>{value:.3f}</td>\n'
                else:
                    loginfo += f'<td>{row[col]}</td>\n'
            loginfo += '</tr>\n'
        loginfo += '</table>\n'
        return loginfo
    
    def write_log(self):
        # Write html style log file
        with open(self.log_file,'w') as f:
            f.write('<style>\n')
            f.write('table, th, td {\n')
            f.write('border: 1px solid black;\n')
            f.write('border-collapse: collapse;\n')
            f.write('padding: 8px;\n')
            f.write('text-align: center;\n')
            f.write('}\n')
            f.write('</style>\n')
            # sort the table keys
            keys_sorted = sorted(self.tables.keys())
            for key in keys_sorted:
                f.write(f'<h1>{key}</h1>\n')
                value = self.tables[key]
                f.write(self.write_table(value))
        return self.log_file
      
