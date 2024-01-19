import os
import pickle
# import time
# import subprocess
from tqdm import tqdm

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MLP
from sklearn.decomposition import PCA

from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP

from qsar.dataset import split_dataset,split_dataset_by_index
from qsar.models.basemodel import BaseQSARModel
from qsar.logger import ResultsLogger



class TorchModel(BaseQSARModel):
    # Base class for all torch models, for example, CNN, RNN, etc.

    def __init__(self,model_dir:str,dataset,pl_model,env_conifg_path:str,k:int=5,n_fold:int=2):
        super().__init__(model_dir,dataset)
        self.model = pl_model
        self.model_name = pl_model.model_name
        self.batch_size = pl_model.batch_size
        self.logdir = os.path.join(self.model_dir, 'log')
        self.k = k
        self.n_fold = n_fold
        assert os.path.exists(env_conifg_path), f'env_config_path {env_conifg_path} does not exist'
        self.env_config_path = env_conifg_path
        self.logger = ResultsLogger(self.logdir,k,n_fold)

    # def train(self,pl_model,trainer,dataset:torch.utils.data.Dataset,batch_size:int=128):
    #     loader = DataLoader(dataset,batch_size=batch_size)
    #     pl_model.reset_parameters()
    #     pl_model.train()
    #     trainer.fit(pl_model,loader)

    def eval(self):
        pass

    def save_kwargs(self):
        dataset_args = self.dataset.get_kwargs()
        model_args = self.model.hparams
        # print(dataset_args)
        # print(model_args)
        kwargs_file = os.path.join(self.model_dir, 'kwargs.pkl')
        with open(kwargs_file, 'wb') as f:
            pickle.dump([dataset_args,model_args], f)
        return kwargs_file


    def write_job_scripts(self,f_scripts:str,cmd:str):
        pass

    # def write_cancel_scripts(self,slurm_file,job_id:str):
    #     with open(slurm_file,'w') as f:
    #         f.write('#!/bin/bash\n')
    #         f.write(f'#SBATCH --job-name=cancel\n')
    #         f.write('#SBATCH --qos=veryshort\n')
    #         f.write('#SBATCH -N 1\n')
    #         f.write('#SBATCH -c 1\n')
    #         f.write('#SBATCH -t 0-00:01\n\n')
    #         f.write(f'scancel {job_id}\n')

    def write_slurm_scripts(self,run_file:str,slurm_file:str,job_name:str,account=None): 
        with open(slurm_file,'w') as f:
            f.write(f'#!/bin/bash\n')
            f.write(f'#SBATCH --job-name={job_name}\n')
            if account is not None:
                f.write(f'#SBATCH --account={account}\n')
            f.write('#SBATCH --nodes=1 --ntasks-per-node=8 --gpus-per-node=1\n')
            f.write('#SBATCH -t 04:00:00\n\n')
            f.write(f'#SBATCH -o {job_name}.out\n')
            # f.write('source /gstore/home/wangs238/miniconda3/etc/profile.d/conda.sh\n')
            # f.write(f'conda activate QSAR\n')
            # f.write('module load CUDA/10\n')
            f.write(f'source {self.env_config_path}\n')
            f.write(f'python {run_file}\n')

    def train(self,train_loader,val_loader,model,model_name,max_epochs:int=5000,patience=150):
        temp_dir = os.path.join(self.model_dir,model_name)
        tf_log_dir = os.path.join(self.model_dir,'tf_logs')
        tflogger = TensorBoardLogger(tf_log_dir,name=model_name)
        earlystop_callbacks = EarlyStopping(monitor='val_loss',patience=patience)
        checkpoint_callback = ModelCheckpoint(dirpath=temp_dir,\
                                              filename='bestmodel',\
                                              monitor='val_loss',\
                                              save_top_k=1)
        trainer = pl.Trainer(default_root_dir=temp_dir,\
                             max_epochs=max_epochs,\
                             logger=tflogger,\
                             reload_dataloaders_every_n_epochs=10,\
                             callbacks=[earlystop_callbacks,checkpoint_callback])
        model.reset_parameters()
        model.train()
        checkpoint_model_path = os.path.join(temp_dir,'bestmodel.ckpt')
        if os.path.exists(checkpoint_model_path):
            os.remove(checkpoint_model_path)
        trainer.fit(model,train_loader,val_loader)
        return checkpoint_model_path
    
    def holdout_split_eval(self,max_epochs:int=5000):
        for i in range(self.k):
            for j in range(self.n_fold):
                self.holdout_split_step(i,j,max_epochs=max_epochs)
    
                
    def holdout_split_step(self,i:int,j:int,max_epochs:int=5000):
        pass
    
    def time_split_eval(self,max_epochs:int=5000):
        # time split eval
        for i in range(self.n_fold):
            self.time_split_step(i,max_epochs=max_epochs)
    
    def k_split_eval_jobs(self,mode:str,max_epochs:int=5000,patience:int=150,account:str=None,sbatch:bool=True):
        # submit k split evaluation jobs for holdout, scaffold and timesplit
        assert mode in ['holdout','scaffold','time']
        job_dir = os.path.join(self.model_dir,f'{mode}_jobs')
        os.makedirs(job_dir,exist_ok=True)
        setattr(self,f'{mode}_job_dir',job_dir)
        os.chdir(job_dir)
        for i in range(self.k):
            for j in range(self.n_fold):
                script_file = os.path.join(job_dir,f'{mode}_{i}_fold_{j}.py')
                cmd = f'{mode}_split_step({i},{j},max_epochs={max_epochs},patience={patience})'
                self.write_job_scripts(script_file,cmd)
                if sbatch:
                    slurm_file = os.path.join(job_dir,f'subjob{i}_{j}.slurm')
                    job_name = f'{mode}_{i}_fold_{j}'
                    self.write_slurm_scripts(script_file,slurm_file,job_name,account=account)
                    os.system(f'sbatch {slurm_file}')
                else:
                    os.system(f'python {script_file}')

    def rerun_err_jobs(self):
        # rerun the job that has failed due to exceptions on cluster
        # check raw data dir for error jobs
        raw_data_dir = os.path.join(self.logdir,'raw_data')
        assert os.path.exists(raw_data_dir),f'run evaluation first'
        holdout_jobdir = os.path.join(self.model_dir,'holdout_jobs')
        scaffold_jobdir = os.path.join(self.model_dir,'scaffold_jobs')
        time_jobdir = os.path.join(self.model_dir,'time_jobs')
        for i in range(self.k):
            for j in range(self.n_fold):
                holdout_data = os.path.join(raw_data_dir,f'holdout_{i}_{j}.pkl')
                scaffold_data = os.path.join(raw_data_dir,f'scaffold_{i}_{j}.pkl')
                time_data = os.path.join(raw_data_dir,f'timesplit_{i}_{j}.pkl')
                if not os.path.exists(holdout_data):
                    os.chdir(holdout_jobdir)
                    holdout_job_file = os.path.join(holdout_jobdir,f'subjob{i}_{j}.slurm')
                    os.system(f'sbatch {holdout_job_file}')
                if not os.path.exists(scaffold_data):
                    os.chdir(scaffold_jobdir)
                    scaffold_job_file = os.path.join(scaffold_jobdir,f'subjob{i}_{j}.slurm')
                    os.system(f'sbatch {scaffold_job_file}')
                if not os.path.exists(time_data):
                    os.chdir(time_jobdir)
                    time_job_file = os.path.join(time_jobdir,f'subjob{i}_{j}.slurm')
                    os.system(f'sbatch {time_job_file}')
        return None                   

    def train_final_jobs(self,sbatch:bool=True):
        job_dir = os.path.join(self.model_dir,'train_final_jobs')
        os.makedirs(job_dir,exist_ok=True)
        setattr(self,'train_final_job_dir',job_dir)
        os.chdir(job_dir)
        for i in range(self.n_fold):
            script_file = os.path.join(job_dir,f'train_final{i}.py')
            cmd = f'final_train_step({i})'
            self.write_job_scripts(script_file,cmd)
            if sbatch:
                slurm_file = os.path.join(job_dir,f'subjob{i}.slurm')
                job_name = f'train_final{i}'
                self.write_slurm_scripts(script_file,slurm_file,job_name)
                os.system(f'sbatch {slurm_file}')
            else:
                os.system(f'python {script_file}')


    def time_split_step(self,i:int,max_epochs:int=5000):
        pass

    def scaffold_split_eval(self,max_epochs:int=5000):
        for i in range(self.k):
            for j in range(self.n_fold):
                self.scaffold_split_step(i,j,max_epochs=max_epochs)
    

    def scaffold_split_step(self,i:int,j_fold:int,max_epochs:int=5000):
        pass

    
class TorchGraphModel(TorchModel):
    def __init__(self, model_dir: str,dataset, pl_model,env_config_path,k:int=5,n_fold:int=5):
        super().__init__(model_dir,dataset, pl_model, env_config_path, k=k,n_fold=n_fold)

    def write_job_scripts(self,f_scripts:str,cmd:str):
        kwargs_file = self.save_kwargs()
        # write individual job scripts
        if self.model_name == 'AttentiveFP':
            lit_model = 'LitAttentiveFP'
        elif self.model_name == 'AttentiveFPMVE':
            lit_model = 'LitAttentiveFPMVE'
        elif self.model_name == 'AttentiveFPEx':
            lit_model = 'LitAttentiveFPEx'
        f = open(f_scripts,'w')
        # Load packages
        f.write('from qsar.dataset import SDFDataset\n')
        f.write(f'from qsar.models.torch_models import TorchGraphModel,{lit_model}\n\n')
        # Load kwargs:
        f.write(f'data_kwargs,model_kwargs = TorchGraphModel.load_kwargs(\'{kwargs_file}\')\n')
        # Reproduce dataset and model
        f.write('dataset = SDFDataset(**data_kwargs)\n')
        f.write(f'lit_model = {lit_model}(**model_kwargs)\n')
        f.write(f'model = TorchGraphModel(\'{self.model_dir}\',dataset,lit_model,\'{self.env_config_path}\')\n')
        # Run command
        f.write(f'model.{cmd}\n')
        f.close()


    def projection(self,model,loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        projs = []
        for batch in loader:
            batch.to(device)
            x_proj = model.projection(batch).detach().cpu()
            projs.append(x_proj)
        return torch.cat(projs,dim=0).numpy()

    def final_train_step(self,i_fold,max_epochs:int=5000,patience=200):
        train_data,val_data,_ = split_dataset(self.dataset,ratio=[0.8,0.2,0.0],\
                                                seed=self.random_seeds[i_fold])
        train_loader = DataLoader(train_data,batch_size=self.batch_size)
        val_loader = DataLoader(val_data,batch_size=self.batch_size)
        model_name = f'final_{i_fold}'
        ckpt_model_path = self.train(train_loader,val_loader,self.model,model_name,max_epochs=max_epochs,patience=patience)
        return ckpt_model_path

    def holdout_split_step(self,i_holdout:int,j_fold:int,max_epochs:int=5000,patience=150):
        # run individual steps
        train_data,_,test_data = split_dataset(self.dataset,ratio=[0.8,0.0,0.2],\
                                                seed=self.random_seeds[i_holdout])
        train_data,val_data,_ = split_dataset(train_data,ratio=[0.85,0.15,0.0],\
                                                seed=self.random_seeds[j_fold])
        train_loader = DataLoader(train_data,batch_size=self.batch_size)
        val_loader = DataLoader(val_data,batch_size=self.batch_size)
        test_loader = DataLoader(test_data,batch_size=self.batch_size)
        self.eval(train_loader,val_loader,test_loader,self.model,\
                  f'holdout_{i_holdout}_{j_fold}',\
                  max_epochs=max_epochs,patience=patience)

    def scaffold_split_step(self,i:int,j_fold:int,max_epochs:int=5000,patience:int=150):
        # run scaffold split evaluation step
        train_idx,_,test_idx = self.dataset.get_scaffold_splits(ratio=[0.8,0.0,0.2],\
                                                                seed=self.random_seeds[i])
        train_data_temp,test_data = split_dataset_by_index(self.dataset,train_idx,test_idx)
        train_data,val_data,_ = split_dataset(train_data_temp,ratio=[0.85,0.15,0.0],\
                                                seed=self.random_seeds[j_fold])
        train_loader = DataLoader(train_data,batch_size=self.batch_size)
        val_loader = DataLoader(val_data,batch_size=self.batch_size)
        test_loader = DataLoader(test_data,batch_size=self.batch_size)
        model_name = f'scaffold_{i}_{j_fold}'
        self.eval(train_loader,val_loader,test_loader,self.model,\
                    model_name,max_epochs=max_epochs,patience=patience)
        
    def time_split_step(self,i:int,j_fold,max_epochs:int=5000,patience:int=150):
        # time split individual step
        train_temp,_,test_data = split_dataset(self.dataset,ratio=[0.8,0.0,0.2],seed=self.random_seeds[i],mode='date')
        train_data,val_data,_ = split_dataset(train_temp,ratio=[0.85,0.15,0.0],\
                                                seed=self.random_seeds[j_fold])
        train_loader = DataLoader(train_data,batch_size=self.batch_size)
        val_loader = DataLoader(val_data,batch_size=self.batch_size)
        test_loader = DataLoader(test_data,batch_size=self.batch_size)
        self.eval(train_loader,val_loader,test_loader,self.model,\
                    f'timesplit_{i}_{j_fold}',max_epochs=max_epochs,patience=patience)
        
    # def eval_scores(self,train_y,train_pred,test_y,test_pred,model_name):
    #     logger_row = []
    #     scores = self.calc_scores(test_y,test_pred)
    #     logger_row.extend(scores)
    #     # plot xy
    #     xy_fig_path = f'{model_name}_xy.png'
    #     xy_fig_log = ResultsLogger.log_image(xy_fig_path)
    #     xy_plot = self.plot_xy(train_pred,train_y,test_pred,test_y,xy_fig_path)
    #     logger_row.extend(['',xy_fig_log,'',''])
    #     return logger_row



    def eval(self,train_loader,val_loader,test_loader,\
             model:pl.LightningModule,model_name,max_epochs:int=5000,patience:int=150):
        # Evaluate model and log results
        # check cuda avaliability
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        checkpoint_model_path = self.train(train_loader,val_loader,model,model_name,max_epochs=max_epochs,patience=patience)        
        model = model.load_from_checkpoint(checkpoint_model_path)
        model.eval().to(device)
        train_y,train_pred,train_labels = [],[],[]
        for batch_train in train_loader:
            batch_train.to(device)
            train_y_temp = batch_train.y
            if self.model_name in ['AttentiveFPMVE']:
                train_pred_mu,train_pred_sigma2 = model(batch_train)
                train_pred_temp = train_pred_mu.detach().cpu()
            else:
                train_pred_temp = model(batch_train).detach().cpu()
            train_y.append(train_y_temp.cpu())
            train_labels.extend(batch_train.label)
            train_pred.append(train_pred_temp)
        train_proj = self.projection(model,train_loader)
        train_y = torch.cat(train_y,dim=0).numpy()
        train_pred = torch.cat(train_pred,dim=0).numpy()
        test_y,test_pred,test_labels,test_var = [],[],[],[]
        for batch_test in test_loader:
            batch_test.to(device)
            test_y_temp = batch_test.y.cpu()
            if self.model_name in ['AttentiveFPMVE']:
                test_pred_mu,test_pred_sigma2 = model(batch_test)
                test_pred_temp = test_pred_mu.detach().cpu()
                test_pred_sigma2 = test_pred_sigma2.detach().cpu()
                test_var.append(test_pred_sigma2)
            else:
                test_pred_temp = model(batch_test).detach().cpu()
            test_y.append(test_y_temp)
            test_labels.extend(batch_test.label)
            test_pred.append(test_pred_temp)
        test_proj = self.projection(model,test_loader)
        test_y = torch.cat(test_y,dim=0).numpy()
        test_pred = torch.cat(test_pred,dim=0).numpy()
        data_dict = self.calc_scores(test_y,test_pred)
        train_samples = {}
        test_samples = {}
        train_samples['labels'] = train_labels
        train_samples['y'] = train_y
        train_samples['pred'] = train_pred
        train_samples['proj'] = train_proj
        test_samples['labels'] = test_labels
        test_samples['y'] = test_y
        test_samples['pred'] = test_pred
        test_samples['proj'] = test_proj
        if self.model_name in ['AttentiveFPMVE']:
            test_var = torch.cat(test_var,dim=0).numpy()
            test_err = np.sqrt(test_var)
            test_samples['pred_var'] = test_var
            test_samples['pred_err'] = test_err
        else:
            test_samples['pred_var'] = None
            test_samples['pred_err'] = None
        data_dict['train_samples'] = train_samples
        data_dict['test_samples'] = test_samples
        self.logger.log_data(model_name,data_dict)

    def prediction(self,use_eval=False):
        # eval_model set to True, prediction with evaluation models
        # Else, use final models
        pred_loader = DataLoader(self.dataset,batch_size=self.batch_size,shuffle=False)
        label_list = []
        for batch in pred_loader:
            label_list.extend(batch.label)
        pred_list = []
        var_list = []
        if use_eval:
            # predict with k holdout models
            for i in range(self.k):
                for j in range(self.n_fold):
                    model_name = f'holdout_{i}_{j}'
                    holdout_model_path = os.path.join(self.model_dir,model_name,'bestmodel.ckpt')
                    if os.path.exists(holdout_model_path):
                        pred,sigma2 = self.predict_with_ckpt(holdout_model_path,pred_loader)
                        pred_list.append(pred)
                        if sigma2 is not None:
                            var_list.append(sigma2)
            # pred with scaffold models:
            # for i in range(self.k):
            #     for j in range(self.n_fold):
            #         model_name = f'scaffold_{i}_{j}'
            #         scaffold_model_path = os.path.join(self.model_dir,model_name,'bestmodel.ckpt')
            #         if os.path.exists(scaffold_model_path):
            #             pred,sigma2 = self.predict_with_ckpt(scaffold_model_path,pred_loader)
            #             pred_list.append(pred)
        else:
            # predict with final models
            for i in range(self.n_fold):
                model_name = f'final_{i}'
                final_model_path = os.path.join(self.model_dir,model_name,'bestmodel.ckpt')
                if os.path.exists(final_model_path):
                    pred,sigma2 = self.predict_with_ckpt(final_model_path,pred_loader)
                    pred_list.append(pred)
                    if sigma2 is not None:
                        var_list.append(sigma2)
        pred_array = np.array(pred_list)
        pred_mean = np.mean(pred_array,axis=0)
        if len(var_list)==0:
            pred_err = np.std(pred_array,axis=0)
        else:
            var_array = np.array(var_list)
            pred_var = np.mean(var_array,axis=0)+np.mean(np.power(pred_array,2),axis=0)-pred_mean**2
            pred_err = np.sqrt(pred_var)
        # log the predictions
        self.logger.log_pred(label_list,pred_mean,pred_err,use_eval=use_eval)
        return label_list,pred_mean,pred_err

        
    def predict_with_ckpt(self,checkpt_path:str,data_loader):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = self.model.load_from_checkpoint(checkpt_path,map_location=device)
        model.to(device).eval()
        preds = []
        sigma2s = []
        for batch in tqdm(data_loader):
            batch.to(device)
            if self.model_name in ['AttentiveFPMVE']:
                pred,sigma2 = model(batch)
                pred = pred.detach().cpu().numpy()
                sigma2 = sigma2.detach().cpu().numpy()
                sigma2s.append(sigma2)
            else:
                pred = model(batch).detach().cpu().numpy()
            preds.append(pred)
            #print(pred)
        preds = np.concatenate(preds,axis=0)
        if len(sigma2s)==0:
            sigma2s = None
        else:
            sigma2s = np.concatenate(sigma2s,axis=0)
        return preds,sigma2s


class LitAttentiveFP(pl.LightningModule):
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,\
                 edge_dim:int,num_layers:int,num_timesteps=2,dropout=0.0,batch_size=128):
        super().__init__()
        self.model_name = 'AttentiveFP'
        self.attentiveFP = AttentiveFP(in_channels,hidden_channels,out_channels,edge_dim,\
                                       num_layers,num_timesteps,dropout)
        self.linear = nn.Linear(out_channels,1,bias=False)
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self,batch):
        x = self.attentiveFP(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
        x = self.linear(x)
        x = x.view(-1)
        return x

    def projection(self,batch):
        return self.attentiveFP(batch.x,batch.edge_index,batch.edge_attr,batch.batch)

    def training_step(self,batch,batch_idx):
        # training_step defined the train loop.
        y = batch.y
        x = self.forward(batch)
        loss1 = F.mse_loss(x,y)
        loss2 = F.l1_loss(x,y)
        # Logging to TensorBoard by default
        self.log_dict({'mse_loss':loss1,'mae_loss':loss2},on_step=False,on_epoch=True,\
                       prog_bar=True,batch_size=self.batch_size)
        return loss1

    def validation_step(self,batch,batch_idx):
        y = batch.y
        x = self.forward(batch)
        val_loss = F.mse_loss(x,y)
        self.log('val_loss',val_loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.batch_size)

    def test_step(self, batch):
        y = batch.y
        x = self.forward(batch)
        test_loss = F.mse_loss(x,y)
        self.log('test_loss',test_loss,on_step=False,on_epoch=True,batch_size=self.batch_size)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        return optimizer

    def reset_parameters(self):
        self.attentiveFP.reset_parameters()
        self.linear.reset_parameters()

class LitAttentiveFPEx(LitAttentiveFP):
    def __init__(self,in_channels:int,feature_length:int,hidden_channels:int,\
                 out_channels:int,edge_dim:int,num_layers:int,num_timesteps=2,\
                 dropout=0.0,batch_size=128):
        super().__init__(in_channels,hidden_channels,out_channels,\
                         edge_dim,num_layers,num_timesteps,dropout,batch_size)
        self.model_name = 'AttentiveFPEx'
        self.mlp1 = MLP(feature_length,[2*out_channels,out_channels],\
                       norm_layer=nn.LazyBatchNorm1d,activation_layer=nn.PReLU,bias=False)
        self.mlp2 = MLP(2*out_channels,[out_channels],\
                        activation_layer=nn.PReLU,bias=False)
        # self.bn = nn.BatchNorm1d(2*out_channels)
        # self.activation = nn.PReLU()
        self.linear = nn.Linear(out_channels,1,bias=False)

    def forward(self,batch):
        x1 = self.attentiveFP(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
        x2 = self.mlp1(batch.global_features)
        # x2 = self.activation(x2)
        x = torch.cat([x1,x2],dim=1)
        x = torch.sigmoid(x)
        x = self.mlp2(x)
        # x = self.bn(x)
        x = self.linear(x)
        x = x.view(-1)
        return x
    
    def projection(self, batch):
        x1 = self.attentiveFP(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
        x2 = self.mlp1(batch.global_features)
        x = torch.cat([x1,x2],dim=1)
        x = self.mlp2(x)
        return x
    
    def reset_parameters(self):
        self.attentiveFP.reset_parameters()
        for m in self.mlp1.modules():
            if hasattr(m,'reset_parameters'):
                m.reset_parameters()
        for m in self.mlp2.modules():
            if hasattr(m,'reset_parameters'):
                m.reset_parameters()
        self.linear.reset_parameters()

class LitAttentiveFPMVE(LitAttentiveFP):
    # Mean Variance Estimation Network for AttentiveFP
    def __init__(self,in_channels:int,hidden_channels:int,out_channels:int,\
                    edge_dim:int,num_layers:int,num_timesteps=2,dropout=0.0,batch_size=128,warmup_epochs=100):
        super().__init__(in_channels,hidden_channels,out_channels,\
                         edge_dim,num_layers,num_timesteps,dropout,batch_size)
        self.model_name = 'AttentiveFPMVE'
        self.mlp1 = MLP(out_channels,[out_channels//2,out_channels//4,1],\
                        norm_layer=nn.BatchNorm1d,bias=False)
        self.mlp2 = MLP(out_channels,[out_channels//2,out_channels//4,1],\
                        norm_layer=nn.BatchNorm1d,bias=False)
        self.linear = None
        self.softplus = nn.Softplus()
        self.warmup_epochs = warmup_epochs
        self.warmup_step = 0
        
    def forward(self,batch):
        x = self.attentiveFP(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
        mu = self.mlp1(x)
        sigma2 = self.mlp2(x)
        sigma2 = self.softplus(sigma2)+0.01
        mu = mu.view(-1)
        sigma2 = sigma2.view(-1)
        return mu,sigma2
    
    def projection(self,batch):
        x = self.attentiveFP(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
        return x
    
    def reset_parameters(self):
        self.attentiveFP.reset_parameters()
        for m in self.mlp1.modules():
            if hasattr(m,'reset_parameters'):
                m.reset_parameters()
        for m in self.mlp2.modules():
            if hasattr(m,'reset_parameters'):
                m.reset_parameters()
    
    def training_step(self,batch,batch_idx):
        y = batch.y
        mu,sigma2 = self.forward(batch)
        mse_loss = F.mse_loss(mu,y)
        nll_loss = 0.5*torch.mean((y-mu)**2/sigma2 + torch.log(sigma2))
        loss = 0
        if self.warmup_step<self.warmup_epochs:
            loss = loss + mse_loss
        else:
            loss = loss + nll_loss + 0.2*mse_loss
        if self.trainer.is_last_batch:
            self.warmup_step += 1
        self.log_dict({'NLL_loss':nll_loss,'mse_loss':mse_loss,'loss':loss},on_step=False,on_epoch=True,\
                      prog_bar=True,batch_size=self.batch_size)
        return loss
    
    def validation_step(self,batch,batch_idx):
        y = batch.y
        mu,sigma2 = self.forward(batch)
        val_loss = F.mse_loss(mu,y)
        #val_loss = 0.5*torch.mean((y-mu)**2/sigma2+torch.log(sigma2))
        self.log('val_loss',val_loss,prog_bar=True,on_step=False,on_epoch=True,batch_size=self.batch_size)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4)
        return optimizer

        

    
