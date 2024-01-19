import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seaborn import kdeplot,boxplot,stripplot,barplot
# from sklearn.decomposition import PCA
# from sklearn.metrics import mean_squared_error
from qsar.dataset import QSARBaseDataset
# import pickle
# import time
# from glob import glob
# import umap
# import csv
from typing import Optional
from bokeh.io import output_file, show
from bokeh.plotting import figure, save, ColumnDataSource
from bokeh.models import HoverTool,LinearColorMapper
from bokeh.palettes import Turbo256,Category20c
from bokeh.transform import cumsum
# from qsar.models.basemodel import BaseQSARModel
# from rdkit import Chem
# from rdkit.Chem import Draw
from qsar.logger import ResultsLogger
from typing import Mapping, Optional

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from glob import glob
import copy

class SummaryWriter():
    # Read results from all models and write summary page

    def __init__(self,base_dir:str,k:int,n:int,train_data:QSARBaseDataset):
        self.base_dir = base_dir
        self.k = k
        self.n_fold = n
        self.results_loggers : Optional(Mapping[str,ResultsLogger]) = {}
        self.tables = {}
        self.summary_file = 'summary.html'
        self.train_dataset = train_data
        ResultsLogger.init_plot_params()

    
    def add_logger(self,model_name:str,log_dir:str):
        assert model_name in ['KNN','RF','XRT','AttentiveFP','AttentiveFPEx','AttentiveFPMVE','GNEprop','RF_Gset','XRT_Gset'], 'Model name not recognized'
        if model_name in ['KNN','RF','XRT','RF_Gset','XRT_Gset']:
            n_fold = 0
        else:
            n_fold = self.n_fold
        results_logger = ResultsLogger(log_dir,self.k,n_fold,load=True)
        self.results_loggers[model_name] = results_logger
    
    def write_logger_summary(self,exps=['holdout','scaffold','timesplit']):
        for rl in self.results_loggers.values():
            if not os.path.exists(rl.log_file):
                rl.plot_mol_structures(self.train_dataset)
                # for exp in exps:
                #     rl.create_tables(exp)
            rl.write_log()
    
    def correlation_report(self,exps=['holdout','scaffold','timesplit']):
        # create correlation report for all models
        # create a separate html file for correlations
        def plot_bokeh_corr(data1:dict,model1:str,data2:dict,model2:str,save_path,cluster_dict,save_plot=False):
            output_file(save_path)
            # sort data by label
            # assert labels are the same
            # return pearson correlation of pred1 and pred2
            labels1 = data1['labels']
            labels2 = data2['labels']
            assert labels1 == labels2, 'Test data are not the same'
            pred1 = data1['pred']
            pred2 = data2['pred']
            if save_plot:
                p = figure(width=350,height=350,x_axis_label=model1,y_axis_label=model2)
                mol_images = [f'../mol_plots/{label}.png' for label in labels1]
                clusters = [cluster_dict[label] for label in labels1]
                category_colors = Category20c[20]
                colors = [category_colors[i] if i<19 else category_colors[19] for i in clusters]
                source = ColumnDataSource(data=dict(x=pred1,y=pred2,label=labels1,imgs=mol_images,cluster=clusters,color=colors))
                s1 = p.circle('x','y',source=source,size=5,color='color',line_color='black',line_width=0.2)
                hover1 = HoverTool(tooltips="""
                <div>
                    <div>
                        <img
                            src="@imgs" height="80" alt="@imgs" width="80"
                        ></img>
                    </div>
                    <div>
                        <span style="font-size: 12px;font-weight: bold;">@label Cluster: @cluster</span>
                    </div>
                </div>
                """,renderers=[s1])
                p.add_tools(hover1)
                m,b = np.polyfit(pred1,pred2,1)
                p.line([min(pred1)-1,max(pred1)+1],\
                    [m*(min(pred1)-1)+b,m*(max(pred1)+1)+b],\
                    line_width=2,color='red',line_dash='dashed')
                p.line([min(pred1)-1,max(pred1)+1],\
                    [min(pred1)-1,max(pred1)+1],\
                    line_width=2,color='black')
                save(p)
            # calculate pearson correlation
            pearson_corr = np.corrcoef(pred1,pred2)[0,1]
            return pearson_corr
        # create correlation folder
        corr_dir = os.path.join(self.base_dir,'corr')
        os.makedirs(corr_dir,exist_ok=True)
        os.chdir(self.base_dir)
        # Plot correlations for all models
        self.corr_df = pd.DataFrame()
        corr_dict = {}
        models = list(self.results_loggers.keys())
        for key1,rl1 in self.results_loggers.items():
            for key2,rl2 in self.results_loggers.items():
                if key1!=key2:
                    for exp in exps:
                        for i in range(self.k):
                            if i==0:
                                save_plot = True
                            else:
                                save_plot = False
                            key = f'{exp}_{i}'
                            data1 = rl1.data[key]['test_samples']
                            data2 = rl2.data[key]['test_samples']
                            cluster_dict = rl1.scaffold_cluster
                            save_path = f'./corr/corr_{key}_{key1}_{key2}.html'
                            corr_temp = plot_bokeh_corr(data1,key1,data2,key2,\
                                                        save_path,cluster_dict,save_plot=save_plot)
                            dict_temp ={'corr':corr_temp,'evaluation':exp,'fold':i,'key1':key1,'key2':key2}
                            corr_dict[f'{exp}_{i}_{key1}_{key2}'] = corr_temp
                            df_temp = pd.DataFrame(dict_temp,index=[0])
                            self.corr_df = pd.concat([self.corr_df,df_temp],axis=0,ignore_index=True)
        # convert corr_figs to dataframe
        # write html file
        corr_html = os.path.join(self.base_dir,'corr.html')
        with open(corr_html,'w') as f:
            loginfo1 = '<html>\n'
            loginfo1 += '<style>\n'
            loginfo1 += 'table, th, td {\n'
            loginfo1 += 'border: 1px solid black;\n'
            loginfo1 += 'border-collapse: collapse;\n'
            loginfo1 += 'padding: 8px;\ntext-align: center;\n}\n'
            loginfo1 += '</style>\n'
            loginfo1+= '<body>\n'
            for exp in exps:
                loginfo1 += f'<table>\n'
                loginfo1 += f'<tr><th>{exp}</th>'
                for model in models:
                    loginfo1 += f'<th>{model}</th>'
                loginfo1 += '</tr>\n'
                for model1 in models:
                    loginfo1 += f'<tr><td>{model1}</td>'
                    for model2 in models:
                        if model1==model2:
                            loginfo1 += f'<td></td>'
                        else:
                            corr_fig = f'./corr/corr_{exp}_0_{model1}_{model2}.html'
                            fig_log = ResultsLogger.log_bokeh_image(corr_fig)
                            loginfo1 += f'<td>{fig_log}</td>'
                    loginfo1 += '</tr>\n'
                loginfo1 += '</table>\n<br><br>'
                loginfo1 += '<table>\n'
            loginfo1 += '</table>\n</body>\n</html>\n'
            f.write(loginfo1)
        # plot box plot for correlations
        corr_box_plot = pd.DataFrame()
        for model in models:
            n_models = len(models)-1
            fig_name = f'corr_{model}.png'
            plt.figure(figsize=(20,10))
            temp_df = self.corr_df[self.corr_df['key1']==model]
            ax = boxplot(x='evaluation',y='corr',hue='key2',data=temp_df,meanline=True,showmeans=True)
            stripplot(x='evaluation',y='corr',hue='key2',data=temp_df,ax=ax,dodge=True)
            for patch in ax.patches:
                fc = patch.get_facecolor()
                patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.3))
            ax.set_xlabel('Evaluation Experiment')
            ax.set_ylabel('Pearson Correlation')
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:n_models], labels[:n_models])
            plt.savefig(fig_name)
            plt.close()
            fig_log = ResultsLogger.log_image(fig_name)
            row = pd.DataFrame({'Pearson Correlations':fig_log},index=[model])
            corr_box_plot = pd.concat([corr_box_plot,row],axis=0,ignore_index=False)
        self.tables['Corr_Box_Plot'] = corr_box_plot
        return None

    def uncertrainty_report(self,exps=['holdout','scaffold']):
        # plot Average Miscalibration area box plot 
        # And plot Miscalibration score at 90% confidence interval
        # Then write table in for summary html
        # exps = ['holdout','scaffold','timesplit']
        uncertanity_df = pd.DataFrame()
        print('Calculating Uncertainty')
        for key,rl in self.results_loggers.items():
            for exp in exps:
                print(f'Calculating Uncertainty for {key} {exp}')
                for i in range(self.k):
                    data_temp = rl.data[f'{exp}_{i}']['test_samples']
                    print(data_temp.keys())
                    if 'AMA' in data_temp.keys() and 'm_90' in data_temp.keys():
                        m_90 = data_temp['m_90']
                        ama = data_temp['AMA']
                        row_temp = pd.DataFrame({'AMA':ama,'m_90':m_90,'model':key,'evaluation':exp},index=[0])
                        uncertanity_df = pd.concat([uncertanity_df,row_temp],axis=0,ignore_index=True)
        models = [key for key in self.results_loggers.keys()]
        uncertrainty_models = uncertanity_df['model'].unique()
        hue_order = []
        for model in models:
            if model in uncertrainty_models:
                hue_order.append(model)
        n_models = len(hue_order)
        # plot box plot for AME
        plt.figure(figsize=(20,10))
        ax = boxplot(x='evaluation',y='AMA',hue='model',hue_order=hue_order,data=uncertanity_df,meanline=True,showmeans=True)
        stripplot(x='evaluation',y='AMA',hue='model',hue_order=hue_order,data=uncertanity_df,ax=ax,dodge=True)
        for patch in ax.patches:
            fc = patch.get_facecolor()
            patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.3))
        ax.set_xlabel('Evaluation Experiment')
        ax.set_ylabel('Average Miscalibration Area')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:n_models], labels[:n_models])
        plt.savefig('AMA.png')
        plt.close()
        # plot box plot for miscallibration score at 90% confidence interval
        plt.figure(figsize=(20,10))
        ax = boxplot(x='evaluation',y='m_90',hue='model',hue_order=hue_order,data=uncertanity_df,meanline=True,showmeans=True)
        stripplot(x='evaluation',y='m_90',hue='model',hue_order=hue_order,data=uncertanity_df,ax=ax,dodge=True)
        for patch in ax.patches:
            fc = patch.get_facecolor()
            patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.3))
        ax.set_xlabel('Evaluation Experiment')
        ax.set_ylabel('Miscalibration Score at 90% Confidence Interval')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:n_models], labels[:n_models])
        plt.savefig('m_90.png')
        plt.close()
        # create dataframe table for AMA and m_90
        row1 = pd.DataFrame({'Uncertainty Calibration':ResultsLogger.log_image('AMA.png')},index=['Average Miscalibration Area'])
        row2 = pd.DataFrame({'Uncertainty Calibration':ResultsLogger.log_image('m_90.png')},index=['Miscalibration Score at 90% Confidence'])
        uncertanity_table = pd.concat([row1,row2],axis=0,ignore_index=False)
        self.tables['Uncertainty_Calibration'] = uncertanity_table
        return None


    def create_summary_tables(self,exp:str,simple=True,**kwargs):
        # create summary tables for a given experiment exp can be ['holdout','scaffold','timesplie']
        # Box plot for the results
        # summary_dict = {}
        if not hasattr(self,'summary_df'):
            self.summary_df = pd.DataFrame()
        # create long form data frame
        for model_name,rl in self.results_loggers.items():
            print(model_name)
            rl.create_tables(exp,simple=simple,**kwargs)
            table_temp_name = exp
            row_temp = rl.tables[table_temp_name].head(self.k) # row has ['MAE','RMSE,'R2','Pearson_r2']
            mae_column = row_temp['MAE']
            rmse_column = row_temp['RMSE']
            r2_column = row_temp['R2']
            pearson_r2_column = row_temp['Pearson_r2']
            df_temp = pd.DataFrame({'MAE':mae_column,'RMSE':rmse_column,'R2':r2_column,\
                                    'Pearson_r2':pearson_r2_column,'model':model_name,'evaluation':exp})
            self.summary_df = pd.concat([self.summary_df,df_temp],axis=0,ignore_index=True)
        
    def create_dataset_summary(self,dataset):
        # Log The dataset information: 
        # 1. Number of molecules
        # 2. Number of scaffold clusters
        # 3. A figure show 10 largeest clusters and their size
        # Return a string that will be written in the summary html
        os.chdir(self.base_dir)
        loginfo = '<h1>Dataset Summary</h1>\n'
        loginfo += f'<h3>Number of molecules: {len(dataset)}</h3>\n'
        plot_dir = os.path.join(self.base_dir,'mol_plots')
        dataset.plot_mol_structures(plot_dir)
        # Plot Scaffold Pie Chart based on default scaffold cluster
        def create_pie_chart_dict(clusters):
            pie_chart_dict = {}
            label_dict = {}
            for i,cluster in enumerate(clusters):
                if i<=18:
                    pie_chart_dict[f'cluster_{i}'] = len(cluster)
                    label_dict[f'cluster_{i}'] = [dataset.raw_samples[idx]['label'] for idx in cluster]
                else:
                    if 'others' in pie_chart_dict.keys():
                        pie_chart_dict['others'] += len(cluster)
                        label_dict['others'].extend([dataset.raw_samples[idx]['label'] for idx in cluster])
                    else:
                        pie_chart_dict['others'] = len(cluster)
                        label_dict['others'] = [dataset.raw_samples[idx]['label'] for idx in cluster]
            return pie_chart_dict,label_dict
        def plot_bokeh_pie_chart(pic_chart_dict,label_dict,save_file,n_samples=20):
            # plot pie chart using bokeh
            clusters = []
            length = []
            samples_table = []
            data_dict = {}
            for key,value in pic_chart_dict.items():
                clusters.append(key)
                length.append(value)
                labels = label_dict[key]
                img_info = '<table>\n'
                img_info += '<tr>\n'
                if n_samples > value:
                    n_plot_samples = value
                else:
                    n_plot_samples = n_samples
                for i in range(n_plot_samples):
                    img_info += f'<td><img src="./mol_plots/{labels[i]}.png" alt="structure" width="70" height="70"></td>\n'
                    if i % 5 == 4:
                        img_info += '</tr>\n'
                        if i!= n_plot_samples-1:
                            img_info += '<tr>\n'
                if n_plot_samples%5 != 0:
                    img_info+= '</tr>\n'
                img_info += '</table>\n'
                samples_table.append(img_info)
                
            angle = [i/sum(length)*2*np.pi for i in length]
            ratio = [i*100/sum(length) for i in length]
            data_dict['clusters'] = clusters
            data_dict['angle'] = angle
            data_dict['ratio'] = ratio
            data_dict['length'] = length
            data_dict['samples'] = samples_table
            print(len(clusters))
            data_dict['color'] = Category20c[len(clusters)]
            output_file(save_file)
            source = ColumnDataSource(data_dict)
            p = figure(height=550,width=600, title="Pie Chart", x_range=(-0.5, 1.0))
            p.wedge(x=0, y=1, radius=0.4,\
                    start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),\
                    line_color="white", fill_color='color', legend_field='clusters', source=source)
            hover = HoverTool(tooltips="""
            <div>@samples</div>
            <div>
            <span style="font-size: 12px;font-weight: bold;">@length  @ratio%</span>
            </div>
            """)
            p.add_tools(hover)
            save(p)
        cluster1 = dataset.scaffold_cluster()
        pie_chart_dict1,label_dict1 = create_pie_chart_dict(cluster1)
        pie_fig1 = 'scaffold_pie_default.html'
        # cluster2 = dataset.scaffold_cluster(use_dc=True)
        # pie_chart_dict2,label_dict2 = create_pie_chart_dict(cluster2)
        # pie_fig2 = 'scaffold_pie_dc.html'
        plot_bokeh_pie_chart(pie_chart_dict1,label_dict1,pie_fig1)
        # plot_bokeh_pie_chart(pie_chart_dict2,label_dict2,pie_fig2)
        loginfo+='<table>\n<tr>\n'
        loginfo+='<th>Rdkit scaffold cluster</th>\n'
        # loginfo+='<th>DeepChem scaffold cluster</th>\n'
        loginfo+='</tr>\n<tr>\n'
        loginfo+=f'<td>{len(cluster1)} clusters</td>\n'
        # loginfo+=f'<td>{len(cluster2)} clusters</td>\n'
        loginfo+='</tr>\n<tr>\n'
        loginfo+=f'<td><embed type="text/html" src="./{pie_fig1}" width="650" height="600"></embed></td>\n'
        # loginfo+=f'<td><embed type="text/html" src="./{pie_fig2}" width="650" height="600"></embed></td>\n'
        loginfo+='</tr>\n</table>\n'
        return loginfo       

    def box_plot_summary(self,bar=False):
        # create box plot for the summary log
        # for each box plot, row is each exp, column is metric value, hue is model
        # Box plot tables: Row is each experiment, column is different metrics
        # Return a box plot table that will be shonw in the summary page
        # if bar is True, also show bar plot for mean of each metric
        os.chdir(self.base_dir)
        assert hasattr(self,'summary_df'), 'Summary table not created'
        metrics_list = ['R2','Pearson_r2','MAE','RMSE']
        hue_order = [key for key in self.results_loggers.keys()]
        boxplot_df = pd.DataFrame()
        n_models = len(self.results_loggers)
        for metric in metrics_list:
            fig_name = f'box_plot_{metric}.png'
            plt.figure(figsize=(12,10))
            ax = boxplot(x='evaluation',y=metric,hue='model',hue_order=hue_order,data=self.summary_df,meanline=True,showmeans=True)
            stripplot(x='evaluation',y=metric,hue='model',hue_order=hue_order,data=self.summary_df,ax=ax,dodge=True)
            for patch in ax.patches:
                fc = patch.get_facecolor()
                patch.set_facecolor(matplotlib.colors.to_rgba(fc, 0.3))
            ax.set_xlabel('Evaluation Experiment')
            ax.set_ylabel(metric)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:n_models], labels[:n_models])
            plt.savefig(fig_name)
            plt.close()
            fig_log = ResultsLogger.log_image(fig_name)
            print(fig_log)
            boxplot_df[metric] = [fig_log]
        boxplot_df.index = ['BoxPlot']
        if bar:
            barplot_data_df = self.summary_df.groupby(['evaluation','model']).mean()
            barplot_data_df = barplot_data_df.reset_index()
            barplot_df = pd.DataFrame()
            for metric in metrics_list:
                fig_name = f'bar_plot_{metric}.png'
                plt.figure(figsize=(12,10))
                ax = barplot(x='evaluation',y=metric,hue='model',hue_order=hue_order,\
                             data=barplot_data_df,alpha=0.5,linewidth=2.5,edgecolor='black')
                ax.set_xlabel('Evaluation Experiment')
                ax.set_ylabel(metric)
                plt.savefig(fig_name)
                plt.close()
                fig_log = ResultsLogger.log_image(fig_name)
                print(fig_log)
                barplot_df[metric] = [fig_log]
            barplot_df.index = ['BarPlot']
            boxplot_df = pd.concat([boxplot_df,barplot_df])
        self.tables['Box_Plot'] = boxplot_df
    
    def log_evaluation_summary(self):
        # return a string that write resutls summary for html
        self.summary_df_mean = self.summary_df.groupby(['evaluation','model']).mean()
        eval_list = self.summary_df['evaluation'].unique()
        model_list = self.summary_df['model'].unique()
        n_eval = len(eval_list)
        # write irregular header, each eval exp report MAE, R2 and pearson_r2
        loginfo = '<table>\n<tr>\n'
        loginfo += '<th></th>\n'
        for eval in eval_list:
            loginfo += f'<th colspan="3">{eval}</th>\n'
        loginfo += '</tr>\n<tr>\n'
        loginfo += '<th></th>\n'
        for eval in eval_list:
            loginfo += f'<th>MAE</th>\n'
            loginfo += f'<th>R2</th>\n'
            loginfo += f'<th>Pearson_r2</th>\n'
        loginfo += '</tr>\n'
        for model_name in model_list:
            # add href to the model name
            href_path = f'./{model_name}/log/results_log.html'
            loginfo += f'<tr>\n<td><a href="{href_path}">{model_name}</a></td>\n'
            for eval in eval_list:
                loginfo += f'<td>{self.summary_df_mean.loc[(eval,model_name),"MAE"]:.3f}</td>\n'
                loginfo += f'<td>{self.summary_df_mean.loc[(eval,model_name),"R2"]:.3f}</td>\n'
                loginfo += f'<td>{self.summary_df_mean.loc[(eval,model_name),"Pearson_r2"]:.3f}</td>\n'
            loginfo += '</tr>\n'
        loginfo += '</table>\n'
        return loginfo
            
    def write_summary(self):
        os.chdir(self.base_dir)
        with open(self.summary_file,'w') as f:
            f.write('<style>\n')
            f.write('table, th, td {\n')
            f.write('border: 1px solid black;\n')
            f.write('border-collapse: collapse;\n')
            f.write('padding: 8px;\n')
            f.write('text-align: center;\n')
            f.write('}\n')
            f.write('</style>\n')
            # write dataset info
            data_info = self.create_dataset_summary(self.train_dataset)
            f.write(data_info)

            # write evaluation info
            f.write('<h1>Evaluations</h1>\n')
            models = list(self.results_loggers.keys())
            models_str = ', '.join(models)
            f.write(f'<h3>Evaluated Models: {models_str}</h3>\n')
            evals = self.summary_df['evaluation'].unique()
            evals_str = ', '.join(evals)
            f.write(f'<h3>Evaluation Experiments: {evals_str}</h3>\n')
            f.write(f'Each split is performed {self.k} repeats\n')
            f.write('<br/>')
            f.write(f'For neural network models, ensemble size is {self.n_fold}\n')
            # write summary box plot table
            box_plots = self.tables['Box_Plot']
            f.write(ResultsLogger.write_table(box_plots))
            f.write('<br/><br/>')
            # Write metircs summary table
            f.write(self.log_evaluation_summary())
            f.write('<br/><br/><br/><br/>')

            # write uncertainty calibration info
            f.write('<h1>Uncertainty Calibration</h1>\n')
            if 'Uncertainty_Calibration' not in self.tables.keys():
                self.uncertrainty_report()
            f.write(ResultsLogger.write_table(self.tables['Uncertainty_Calibration']))
            f.write('<br/><br/>')

            # write correlation info
            if hasattr(self,'corr_df'):
                f.write('<h1>Correlations between models</h1>\n')
                f.write('<h3>Correlation between models for each evaluation experiment</h3>\n')
                f.write(ResultsLogger.write_table(self.tables['Corr_Box_Plot']))
                f.write('<br/><br/>')
                f.write('<a href="./corr.html">Correlation details</a>\n')

        return self.summary_file
        
    # def get_summary(self,exp,**kwargs):
    #     # Get summary of results for a given experiment
    #     self.create_tables(exp,**kwargs)
    #     pass     
        
class PredWriter(SummaryWriter):
    # Write prediction results to csv file
    # And write a html prediction file
    def __init__(self,base_dir:str,k:int,n:int,train_data:QSARBaseDataset,test_data:QSARBaseDataset):
        super().__init__(base_dir,k,n,train_data)
        self.test_dataset = test_data
        self.pred_dir = os.path.join(self.base_dir,'pred')
        os.makedirs(self.pred_dir,exist_ok=True)
        os.chdir(self.pred_dir)
        self.pred_csv = os.path.join(self.pred_dir,'pred.csv')
        self.pred_html = os.path.join(self.pred_dir,'pred.html')

    def process_pred_data(self):
        # read predition results from all the models
        # and save to a diction
        self.pred_data = {}
        for key,rl in self.results_loggers.items():
            if key=='KNN':
                continue
            pred_dict = rl.process_pred_results(self.train_dataset,self.test_dataset)
            # pred_dict contains dicts with keys as labels
            # each dict has keys: 'label','pred_mean','pred_err','in_train':bool,'train_value':Optional[float]=None
            self.pred_data[key] = pred_dict
        return self.pred_data
    
    def final_prediction(self,used_models:list):
        # aggregate the prediction results from all models and make final predictions
        for model in used_models:
            assert model in self.pred_data.keys(), f'{model} not in pred_data'
        final_pred = {}
        for label in self.pred_data[used_models[0]].keys():
            pred_means = []
            pred_errs = []
            for model in used_models:
                pred_means.append(self.pred_data[model][label]['pred_mean'])
                pred_errs.append(self.pred_data[model][label]['pred_err'])
            pred_means = np.array(pred_means)
            pred_errs = np.array(pred_errs)
            pred_vars = pred_errs**2
            pred_mean = np.mean(pred_means)
            pred_var = np.mean(pred_vars) + np.mean(np.power(pred_means,2)) - np.power(pred_mean,2)
            pred_err = np.sqrt(pred_var)
            in_train = self.pred_data[used_models[0]][label]['in_train']
            train_value = self.pred_data[used_models[0]][label]['train_value']
            final_pred[label] = {'label':label,'pred_mean':pred_mean,'pred_err':pred_err,'in_train':in_train,'train_value':train_value}
        self.pred_data['Final'] = final_pred
        return final_pred

    def plot_bokeh_preds(self):
        # plot the scatter plot of prediction with respect to uncertainty
        mol_plots_dir = os.path.join(self.pred_dir,'mol_plots')
        os.chdir(self.pred_dir)
        if os.path.exists(mol_plots_dir):
            n_mols = len(self.test_dataset.raw_samples)
            n_figs = len(glob(os.path.join(mol_plots_dir,'*.png')))
            if n_mols!=n_figs:
                self.test_dataset.plot_mol_structures(mol_plots_dir)
        else:
            self.test_dataset.plot_mol_structures(mol_plots_dir)
        distances = self.calc_pred_distances()
        def plot_pred_bokeh(model_name,pred_dict,distances):
            pred_means = []
            pred_errs = []
            labels = []
            images = []
            for label,preds in pred_dict.items():
                if preds['in_train']:
                    continue
                pred_means.append(preds['pred_mean'])
                pred_errs.append(preds['pred_err'])
                labels.append(label)
                img_path = f'./mol_plots/{label}.png'
                images.append(img_path)
            # print(pred_means)
            source_data1 = ColumnDataSource(data=dict(x=pred_means,y=pred_errs,labels=labels,imgs=images,\
                                                     distances=distances))
            source_data2 = copy.deepcopy(source_data1)
            fig_path1 = f'{model_name}_pred_err.html'
            output_file(fig_path1)
            hover1 = HoverTool(tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="80" alt="@imgs" width="80"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 12px; font-weight: bold;">@labels</span>
                </div>
            </div>
            """)
            p = figure(width=600, height=600,\
                       x_axis_label='Prediction',y_axis_label='Uncertainty')
            p.circle('x','y',source=source_data1)
            p.add_tools(hover1)
            save(p)

            fig_path2 = f'{model_name}_pred_dist.html'
            output_file(fig_path2)
            hover2 = HoverTool(tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs" height="80" alt="@imgs" width="80"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 12px; font-weight: bold;">@labels</span>
                </div>
            </div>
            """)
            p2 = figure(width=600, height=600,\
                          x_axis_label='Prediction',y_axis_label='Distance')
            p2.circle('x','distances',source=source_data2)
            p2.add_tools(hover2)
            save(p2)
            fig_log1 = ResultsLogger.log_bokeh_image(fig_path1,width=620,height=620)
            fig_log2 = ResultsLogger.log_bokeh_image(fig_path2,width=620,height=620)
            return fig_log1,fig_log2
        pred_err_figs = []
        pred_dist_figs = []
        for model_name,pred_dict in self.pred_data.items():
            pred_err_fig,pred_dist_fig = plot_pred_bokeh(model_name,pred_dict,distances)
            pred_err_figs.append(pred_err_fig)
            pred_dist_figs.append(pred_dist_fig)
        pred_err_table = pd.DataFrame()
        pred_err_table['Prediction vs Uncertainty'] = pred_err_figs
        pred_err_table['Prediction vs Distance'] = pred_dist_figs
        pred_err_table.index = [model_name for model_name in self.pred_data.keys()]
        self.tables['pred_err_figs'] = pred_err_table
        
    def calc_pred_distances(self):
        # plot the scatter plot of prediction with respect to distance
        # calc cloest distance between pred and train data
        # calc fingerprint similarity between pred and train data
        train_smis = [sample['SMILES'] for sample in self.train_dataset.raw_samples]
        ms = [Chem.MolFromSmiles(smi) for smi in train_smis]
        train_fps = [AllChem.GetMorganFingerprintAsBitVect(m,2,2048) for m in ms]
        models = list(self.pred_data.keys())
        distances = []
        dict_temp = self.pred_data[models[0]]
        pred_data_dict = self.test_dataset.raw_data_to_dict()
        for label,pred_dict in dict_temp.items():
            if pred_dict['in_train']:
                continue
            pred_smi = pred_data_dict[label]['SMILES']
            pred_mol = Chem.MolFromSmiles(pred_smi)
            pred_fp = AllChem.GetMorganFingerprintAsBitVect(pred_mol,2,2048)
            similarities = DataStructs.BulkTanimotoSimilarity(pred_fp,train_fps)
            max_sim = np.max(similarities)
            dis = 1-max_sim
            distances.append(dis)
        return distances

    def write_pred_log(self):
        # write predictions to csv file
        # And visualze results in html
        assert 'Final' in self.pred_data.keys(), 'Final prediction not calculated'
        with open(self.pred_csv,'w') as f:
            f.write('label,pred,pred_err,in_train,train_value\n')
            for label,pred_dict in self.pred_data['Final'].items():
                f.write(f'{label},{pred_dict["pred_mean"]},{pred_dict["pred_err"]},{pred_dict["in_train"]},{pred_dict["train_value"]}\n')
        n_in_train = 0
        n_in_test = 0
        for label,pred_dict in self.pred_data['Final'].items():
            if pred_dict['in_train']:
                n_in_train += 1
            else:
                n_in_test += 1

        with open(self.pred_html,'w') as f:
            loginfo = '<style>\n'
            loginfo += 'table, th, td {\n'
            loginfo += 'border: 1px solid black;\n'
            loginfo += 'border-collapse: collapse;\n'
            loginfo += 'padding: 8px;\n'
            loginfo += 'text-align: center;\n'
            loginfo += '}\n'
            loginfo += '</style>\n'
            loginfo += '<h1>Prediction Data</h1>\n'   
            loginfo += f'<h3>Number of molecules in training set: {n_in_train}</h3>\n'
            loginfo += f'<h3>Number of molecules not in training set: {n_in_test}</h3>\n'
            # log prediction data information
            loginfo += '<h1>Prediction Results</h1>\n'
            loginfo += ResultsLogger.write_table(self.tables['pred_err_figs'])
            f.write(loginfo)
