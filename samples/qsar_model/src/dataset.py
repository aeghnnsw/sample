import csv
import os
import pickle
# import sys
from copy import deepcopy
import time
import numpy as np
# import deepchem as dc

from rdkit import DataStructs
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.ML.Cluster import Butina

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm
from glob import glob
# Global dataset class, read from raw csv file 
# calculate all descriptors, but only retrieve user-defined data-type

class QSARBaseDataset(Dataset):
    def __init__(self,  end_point_property:str,label_tag:str,\
                        data_type:str,data_dir:str,end_point_name=None,\
                        sort_by_date:bool=True,save=True,load=True,\
                        raw_file=None,reverse=False,\
                        log=False,negative=False,offset:float=0,predict=False,\
                        desp_upper_limit:float=3000):
        self.end_point_property = end_point_property
        self.label_tag = label_tag
        data_type_list = ['Descriptor_FP','Graph','Graph_Descriptor_FP','Graph_Descriptor','SMILES']
        assert data_type in data_type_list,f'Data type not supported! Please choose from {data_type_list}'
        self.data_type = data_type
        self.raw_file = raw_file
        self.upper_limit = desp_upper_limit
        self.data_dir = data_dir
        if os.path.exists(self.data_dir) is False:
            os.makedirs(self.data_dir)
        self.end_point_name = end_point_name if end_point_name is not None else self.end_point_property
        self.sort_by_date = sort_by_date
        self.dataset_file = os.path.join(data_dir,f'{self.end_point_name}_{data_type}.pkl')
        if load and os.path.exists(self.dataset_file):
            self.load()
            return None
        
        self.save_data = save    
        self.raw_samples = self.get_raw_samples(log=log,negative=negative,offset=offset,predict=predict) # list of dict (key: 'mol','label','[end_point_name]')
        print('number of molecules before removing enantiomer:',len(self.raw_samples))
        if predict==False:
            self.remove_enantiomer(reverse=reverse)
        print('number of molecules after removing enantiomer:',len(self.raw_samples))
        # sort the raw_samples by key: label
        if self.sort_by_date:
            self.raw_samples = sorted(self.raw_samples,key=lambda x:x['label'])
        print(len(self.raw_samples))
        self.data = []
        self.descriptor_keys = []
        print('get_dataset')
        self.get_dataset()
        if self.save_data:
            with open(self.dataset_file,'wb') as f_dataset:
                pickle.dump([self.raw_samples,self.data,self.descriptor_keys],f_dataset)
    
    def get_kwargs(self,load:bool=True):
        # get kwargs for reproduce the dataset
        kwargs = {'end_point_property':self.end_point_property,\
                    'label_tag':self.label_tag,\
                    'data_type':self.data_type,\
                    'data_dir':self.data_dir,\
                    'end_point_name':self.end_point_name,\
                    'sort_by_date':self.sort_by_date,\
                    'load':load,\
                    'raw_file':self.raw_file}
        return kwargs

    def __len__(self):
        return len(self.data)
    
    def remove_enantiomer(self,reverse=False):
        # remove duplicated enantiomer from raw_samples
        # only keep the one with larger end_point_property, 
        #   if reverse=True, keep the smaller one
        assert len(self.raw_samples)>0,'No raw samples!'
        # sort by end_point_property
        remove_list = []
        n_samples = len(self.raw_samples)
        for i in range(n_samples):
            for j in range(i+1,n_samples):
                if self.raw_samples[i]['SMILES']==self.raw_samples[j]['SMILES']:
                    end_pointi = self.raw_samples[i][self.end_point_name]
                    end_pointj = self.raw_samples[j][self.end_point_name]
                    if reverse:
                        if end_pointi>end_pointj:
                            remove_list.append(i)
                        else:
                            remove_list.append(j)
                    else:
                        if end_pointi<end_pointj:
                            remove_list.append(i)
                        else:
                            remove_list.append(j)
        remove_list = list(set(remove_list))
        self.raw_samples = [self.raw_samples[i] for i in range(n_samples) if i not in remove_list]
        return self.raw_samples
    
    def get_dataset(self):
        if self.data_type=='Descriptor_FP':
            self.get_Descriptor_FP_Dataset()
        elif self.data_type=='Graph':
            self.get_Graph_Dataset()
        elif self.data_type=='GSet':
            self.get_GSet_Dataset()
        elif self.data_type=='Graph_Descriptor':
            self.get_Graph_Descriptor_Dataset()
        elif self.data_type=='Graph_Descriptor_FP':
            self.get_Graph_Descriptor_FP_Dataset()
        elif self.data_type=='SMILES':
            self.get_SMILES_Dataset()

    def get_descriptro_keys(self):
        if self.data_type=='Graph':
            print('Graph dataset has no descriptor keys!')
        return self.descriptor_keys

    def get_Descriptor_FP_Dataset(self):
        fpgen = AllChem.GetMorganGenerator(radius=2,fpSize=1024)
        for sample in tqdm(self.raw_samples):
            mol_temp = sample['mol']
            fps = fpgen.GetFingerprint(mol_temp)
            arr = np.zeros((0,),dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fps,arr)
            fp_descriptor_keys = [f'fp_{i}' for i in range(len(arr))]
            descriptors_dict = Descriptors.CalcMolDescriptors(mol_temp,missingVal=0)
            x = list(descriptors_dict.values())
            # Replace NaN with 0
            x = [0 if np.isnan(i) else i for i in x]
            x = [self.upper_limit if i>self.upper_limit else i for i in x]
            x = np.array(x)
            x = np.concatenate((x,arr))
            y = sample[self.end_point_name]
            label = sample['label']
            self.data.append([x,y,label])
        self.descriptor_keys = list(descriptors_dict.keys())+fp_descriptor_keys
        return self.data

    def get_Graph_Dataset(self):
        for sample in tqdm(self.raw_samples):
            mol_temp = sample['mol']
            atom_x,edge_index,bond_x = self.get_molgraph(mol_temp)
            y = sample[self.end_point_name]
            label = sample['label']
            graph_temp = Data(x=atom_x,edge_index=edge_index,edge_attr=bond_x,y=y,label=label)
            self.data.append(graph_temp)
   
    def get_Graph_Descriptor_FP_Dataset(self):
        fpgen = AllChem.GetMorganGenerator(radius=2,fpSize=1024)
        for sample in tqdm(self.raw_samples):
            mol_temp = sample['mol']
            atom_x,edge_index,bond_x = self.get_molgraph(mol_temp)
            fps = fpgen.GetFingerprint(mol_temp)
            arr = np.zeros((0,),dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fps,arr)
            fp_descriptor_keys = [f'fp_{i}' for i in range(len(arr))]
            descriptors_dict = Descriptors.CalcMolDescriptors(mol_temp,missingVal=0)
            x = list(descriptors_dict.values())
            # # Replace NaN with 0
            # x = [0 if np.isnan(i) else i for i in x]
            x = [self.upper_limit if i>self.upper_limit else i for i in x]
            x = np.array(x)
            x = np.concatenate((x,arr))
            x = torch.from_numpy(x).float().view(1,-1)
            y = sample[self.end_point_name]
            label = sample['label']
            graph_temp = Data(x=atom_x,edge_index=edge_index,\
                              edge_attr=bond_x,y=y,label=label,
                              global_features=x)
            self.data.append(graph_temp)
        self.descriptor_keys = list(descriptors_dict.keys())+fp_descriptor_keys

    def get_Graph_Descriptor_Dataset(self):
        for sample in tqdm(self.raw_samples):
            mol_temp = sample['mol']
            atom_x,edge_index,bond_x = self.get_molgraph(mol_temp)
            descriptors_dict = Descriptors.CalcMolDescriptors(mol_temp,missingVal=0)
            x = list(descriptors_dict.values())
            # Replace NaN with 0
            # x = [0 if np.isnan(i) else i for i in x]
            x = [self.upper_limit if i>self.upper_limit else i for i in x]
            x = torch.from_numpy(np.array(x)).float().view(1,-1)
            y = sample[self.end_point_name]
            label = sample['label']
            graph_temp = Data(x=atom_x,edge_index=edge_index,\
                              edge_attr=bond_x,y=y,label=label,
                              global_features=x)
            self.data.append(graph_temp)
        self.descriptor_keys = list(descriptors_dict.keys())


    def get_SMILES_Dataset(self):
        for sample in tqdm(self.raw_samples):
            mol_temp = sample['mol']
            y = sample[self.end_point_name]
            label = sample['label']
            smiles = sample['SMILES']
            self.data.append([smiles,y,label])
        if self.save_data:
            f_dataset = open(self.dataset_file,'wb')
            pickle.dump(self.data,f_dataset)
            f_dataset.close()
    

    def get_molgraph(self,mol):
        # Get mol graph from rdkit mol object and return torch_geometric Data
        # Reference: "Pushing the Boundaries of Molecular Representation for Drug Discovery 
        # with the Graph Attention Mechanism"
        # Atom features include (11+6+1+6+1+1=26): 
        #   1. Atomic symbol    (one-hot encoding) [B,C,N,O,F,P,S,Cl,Br,I,others][0-10]
        #   2. degree           (one-hot encoding) [0-5]
        #   3. formal charge    (integer)
        #   4. hybridization    (one-hot encoding) (sp,sp2,sp3,sp3d,sp3d2)[0-5]
        #   5. aromatic         (boolean)   
        #   6. Hydrogens        (integer) number of connected hydrogens
        # Bond features include (5+1+1=7):
        #   1. Bond type        (one-hot encoding) (single,double,triple,aromatic)[0-4]
        #   2. Conjugated       (boolean) whether the bond is conjugated
        #   3. Ring             (boolean) whether the bond is in a ring

        # Atom type use one-hot encoding with
        atom_type_dict = {'B':0,'C':1,'N':2,'O':3,'F':4,'P':5,'S':6,'Cl':7,'Br':8,'I':9,'others':10}
        hybridization_dict = {Chem.rdchem.HybridizationType.SP:0,Chem.rdchem.HybridizationType.SP2:1,Chem.rdchem.HybridizationType.SP3:2,Chem.rdchem.HybridizationType.SP3D:3,Chem.rdchem.HybridizationType.SP3D2:4,'others':5}
        bondtype_dict = {Chem.rdchem.BondType.SINGLE:0,Chem.rdchem.BondType.DOUBLE:1,Chem.rdchem.BondType.TRIPLE:2,Chem.rdchem.BondType.AROMATIC:3,'others':4}
        atom_x = torch.zeros((mol.GetNumAtoms(),26))
        bond_x = torch.zeros((mol.GetNumBonds(),7))
        edge_index = torch.zeros((2,mol.GetNumBonds()),dtype=torch.long)
        for idx,atom_temp in enumerate(mol.GetAtoms()):
            atom_symbol = atom_temp.GetSymbol()
            if atom_symbol in atom_type_dict.keys():
                atom_x[idx,atom_type_dict[atom_symbol]] = 1
            else:
                atom_x[idx,atom_type_dict['others']] = 1
            atom_x[idx,11+atom_temp.GetDegree()] = 1
            charge = atom_temp.GetFormalCharge()
            atom_x[idx,16] = charge
            hybridization = atom_temp.GetHybridization()
            if hybridization in hybridization_dict.keys():
                atom_x[idx,17+hybridization_dict[hybridization]] = 1
            else:
                atom_x[idx,17+hybridization_dict['others']] = 1
            if atom_temp.GetIsAromatic():
                atom_x[idx,24] = 1
            atom_x[idx,25] = atom_temp.GetTotalNumHs()
        for idx,bond_temp in enumerate(mol.GetBonds()):
            id1 = bond_temp.GetBeginAtomIdx()
            id2 = bond_temp.GetEndAtomIdx()
            edge_index[0,idx] = id1
            edge_index[1,idx] = id2
            bond_type = bond_temp.GetBondType()
            if bond_type in bondtype_dict.keys():
                bond_x[idx,bondtype_dict[bond_type]] = 1
            else:
                bond_x[idx,bondtype_dict['others']] = 1
            if bond_temp.GetIsConjugated():
                bond_x[idx,5] = 1
            if bond_temp.IsInRing():
                bond_x[idx,6] = 1
        edge_index,bond_x = to_undirected(edge_index,bond_x) # convert to undirected graph
        # graph = Data(x=atom_x,edge_index=edge_index,edge_attr=bond_x)
        return atom_x,edge_index,bond_x
    

    def get_raw_samples(self):
        # rewrite this in child class
        pass

    def load(self):
        # Load dataset
        open_file = True
        while open_file:
            try: 
                with open(self.dataset_file,'rb') as f:
                    self.raw_samples,self.data,self.descriptor_keys = pickle.load(f)
                open_file = False
            except:
                time.sleep(1)
        return None

    def plot_mol_structures(self,plot_dir:str):
        # plot the molecular structures
        os.makedirs(plot_dir,exist_ok=True)
        # check if the mol_plots has been created
        raw_samples = self.raw_samples
        n_samples = len(raw_samples)
        png_files = os.path.join(plot_dir,'*.png')
        n_plots = glob(png_files)
        if len(n_plots)==n_samples:
            return None
        for sample in raw_samples:
            fig_file = os.path.join(plot_dir,f'{sample["label"]}.png')
            mol = Chem.MolFromSmiles(sample['SMILES'])
            Draw.MolToFile(mol,fig_file)
        return None
    

    def show_info(self):
        print(f'Raw data file: {self.raw_file}')
        print(f'Data type: {self.data_type}')
        print(f'End Point Property: {self.end_point_name}')
        print(f'Sort by Date: {self.sort_by_date}')
        print(f'Number of samples: {len(self.data)}')
        if self.data_type=='Graph':
            print(f'Number of atoms features: {self.data[0].x.shape[1]}')
            print(f'Number of bonds features: {self.data[0].edge_attr.shape[1]}')
        elif self.data_type=='Graph_Descriptor_FP' or self.data_type=='Graph_Descriptor':
            print(f'Number of atoms features: {self.data[0].x.shape[1]}')
            print(f'Number of bonds features: {self.data[0].x.shape[1]}')
            print(f'Number of global features: {self.data[0].global_features.shape[1]}')
        elif self.data_type=='SMILES':
            print('SMILES dataset')
        else:
            print(f'Number of features: {len(self.data[0][0])}')
        if self.data_type=='Descriptor_FP':
            print(f'Number of descriptors: {len(self.descriptor_keys)}')
            print(f'Descriptors: {self.descriptor_keys}')

    def cluster_split(self,ratio=[0.7,0.1,0.2],seed=0):
        # Inplement Butina clustering using RDKit
        # Return a list of splits
        #     0, Random split for all clusters
        #     1-n_clusters, excluding the cluster i in the train and val set
        # Returned list has length 1+n_clusters, each element has 3 lists
        assert len(self.raw_samples)>0,'No data to cluster'
        assert len(ratio)==3,'Ratio should be a list of length 3'
        assert sum(ratio)==1,'Sum of ratio should be 1'
        dists = []
        fps = []
        for sample in self.raw_samples:
            mol = Chem.MolFromSmiles(sample['SMILES'])
            fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
            fps.append(fp)
        for i in range(1,len(fps)):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
            dists.extend([1-x for x in sims])
        mol_clusters = Butina.ClusterData(dists,len(fps),0.5,isDistData=True)
        mol_clusters = [list(cluster) for cluster in mol_clusters]
        # combine clusters with small number of samples
        # iteratively combine clusters with smallest number of samples
        # stop when the smallest cluster has more than 10% of the total samples
        mol_clusters = sorted(mol_clusters,key=lambda x:len(x),reverse=True)
        while len(mol_clusters[-1])<0.1*len(fps):
            mol_clusters[-2].extend(mol_clusters[-1])
            mol_clusters.pop()
            mol_clusters = sorted(mol_clusters,key=lambda x:len(x),reverse=True)
        n_clusters = len(mol_clusters)
        def get_cluster_index(cluster,ratio):
            n_samples = len(cluster)
            n_train = int(n_samples*ratio[0])
            n_val = int(n_samples*ratio[1])
            train_idx = np.random.choice(cluster,n_train,replace=False)
            val_idx = np.random.choice(list(set(cluster)-set(train_idx)),\
                                            n_val,replace=False)
            test_idx = list(set(cluster)-set(train_idx)-set(val_idx))
            return train_idx,val_idx,test_idx
        splits = []
        # Randomly split each cluster:
        train_idx_rand,val_idx_rand,test_idx_rand = [],[],[]
        np.random.seed(seed)
        for cluster in mol_clusters:
            train_idx,val_idx,test_idx = get_cluster_index(cluster,ratio)
            train_idx_rand.extend(train_idx)
            val_idx_rand.extend(val_idx)
            test_idx_rand.extend(test_idx)
        splits.append([train_idx_rand,val_idx_rand,test_idx_rand])
        # Remove each cluster from training and validation set
        # And output N_cluster different splits
        for i in range(n_clusters):
            train_idx_temp,val_idx_temp,test_idx_temp = [],[],[]
            for j in range(n_clusters):
                train_idx,val_idx,test_idx = get_cluster_index(mol_clusters[j],ratio)
                if j!=i:
                    train_idx_temp.extend(train_idx)
                    val_idx_temp.extend(val_idx)
                test_idx_temp.extend(test_idx)
            splits.append([train_idx_temp,val_idx_temp,test_idx_temp])
        return splits

    def scaffold_cluster(self,use_dc=False,threshold=0.6):
        # Inplement Butina clustering using RDKit
        assert len(self.raw_samples)>0,'No data to cluster'
        if use_dc:
        #     smis = [sample['SMILES'] for sample in self.raw_samples]
        #     x_temp = np.arange(len(smis))
        #     dc_np_dataset = dc.data.NumpyDataset(x_temp,ids=smis)
        #     scaffold_splitter = dc.splits.ScaffoldSplitter()
        #     mol_clusters = scaffold_splitter.generate_scaffolds(dc_np_dataset)
        #     self.dc_clusters = mol_clusters
            print('DeepChem clustering not recommended')
            raise NotImplementedError
        else:
            dists = []
            fps = []
            for sample in self.raw_samples:
                mol = Chem.MolFromSmiles(sample['SMILES'])
                fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
                fps.append(fp)
            for i in range(1,len(fps)):
                sims = DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
                dists.extend([1-x for x in sims])
            # decrease the threshold until there are at least 20 clusters
            # And make sure the largest cluster is smaller than 50% of the total samples
            while True:
                mol_clusters = Butina.ClusterData(dists,len(fps),threshold,isDistData=True)
                mol_clusters = [list(cluster) for cluster in mol_clusters]
                mol_clusters = sorted(mol_clusters,key=lambda x:len(x),reverse=True)
                n_largest = len(mol_clusters[0])
                if len(mol_clusters)>=20 and n_largest<0.5*len(fps):
                    break
                threshold -= 0.05
            mol_clusters = Butina.ClusterData(dists,len(fps),threshold,isDistData=True)
            mol_clusters = [list(cluster) for cluster in mol_clusters]
            # sort the clusters by size
            mol_clusters = sorted(mol_clusters,key=lambda x:len(x),reverse=True)
            self.clusters = mol_clusters
        cluster_dict = {}
        for i,cluster in enumerate(mol_clusters):
            for idx in cluster:
                cluster_dict[self.raw_samples[idx]['label']] = i
        self.cluster_dict = cluster_dict
        return mol_clusters
        
    def get_scaffold_splits(self,ratio=[0.7,0.1,0.2],seed=0,use_dc=False,threshold=0.6):
        # Get scaffold splits for train, val and test
        # if use_dc set to True, use deepchem scaffold split
        assert len(ratio)==3,'Ratio should be a list of length 3'
        assert sum(ratio)==1,'Sum of ratio should be 1'
        train_ratio = ratio[0]
        val_ratio = ratio[1]
        test_ratio = ratio[2]
        if use_dc:
            if not hasattr(self,'dc_clusters'):
                self.scaffold_cluster(use_dc=use_dc)
            mol_clusters = deepcopy(self.dc_clusters)
        else:
            if not hasattr(self,'clusters'):
                self.scaffold_cluster(use_dc=use_dc,threshold=threshold)
            mol_clusters = deepcopy(self.clusters)
        # print('number of cluster: ',len(mol_clusters))
        # randomly choose clusters to add to the test set until reach ratio
        test_idx = []
        # if seed is 0, do not random choose, but use largest clusters for training
        if seed==0:
            while len(test_idx)/len(self.raw_samples)<test_ratio:
                test_idx.extend(mol_clusters.pop())
        else:
            np.random.seed(seed)
            while len(test_idx)/len(self.raw_samples)<test_ratio:
                cluster_idx = np.random.choice(len(mol_clusters))
                test_idx.extend(mol_clusters[cluster_idx])
                mol_clusters.pop(cluster_idx)
        # Randomly split all rest idx to train and val
        rest_idx = [idx for cluster in mol_clusters for idx in cluster]
        train_length = int(len(rest_idx)*train_ratio/(train_ratio+val_ratio))
        train_idx = np.random.choice(rest_idx,train_length,replace=False)
        val_idx = list(set(rest_idx)-set(train_idx))
        return train_idx,val_idx,test_idx

    def raw_data_to_dict(self):
        assert len(self.raw_samples)!=0,'No data to process'
        raw_dict = {}
        for sample in self.raw_samples:
            raw_dict[sample['label']] = sample
        return raw_dict

    
    def __getitem__(self, index):
        if self.data_type=='Graph' or self.data_type=='Graph_Descriptor' \
           or self.data_type=='Graph_Descriptor_FP':
            return self.data[index]
        else:
            x,y,label = self.data[index]
            return x,y,label
        

class CSVDataset(QSARBaseDataset):
    def __init__(self,  raw_file:str,end_point_property:str,label_tag:str,\
                        data_type:str,data_dir:str,end_point_name=None,sort_by_date=True,\
                        save=True,load=False,predict=False,**kwargs):
        super().__init__(end_point_property,label_tag,data_type,\
                         data_dir,end_point_name=end_point_name,sort_by_date=sort_by_date,\
                         save=save,load=load,predict=predict,raw_file=raw_file,**kwargs)
    def get_raw_samples(self,log:bool=False,negative:bool=False,offset:float=0,predict=False):
        # read raw csv file
        assert os.path.exists(self.raw_file),'Raw csv file does not exist!'
        raw_samples = list()
        f = open(self.raw_file,'r')
        reader = csv.DictReader(f)
        for row in reader:
            if len(row[self.end_point_property])==0:
                continue
            if len(row[self.label_tag])==0:
                continue
            if len(row['SMILES'])==0:
                continue
            mol_temp = Chem.MolFromSmiles(row['SMILES'])
            if mol_temp is not None:
                sample = dict()
                sample['mol'] = mol_temp
                sample['label'] = row[self.label_tag]
                sample['SMILES'] = Chem.MolToSmiles(mol_temp,isomericSmiles=False)
                if predict:
                    sample[self.end_point_name] = None
                else:
                    try:
                        end_point_value = float(row[self.end_point_property])
                    except:
                        continue
                    if log:
                        end_point_value = float(np.log10(end_point_value))
                    if negative:
                        end_point_value = -end_point_value
                    end_point_value += offset
                    sample[self.end_point_name] = end_point_value
                raw_samples.append(sample)
        f.close()
        return raw_samples


class SDFDataset(QSARBaseDataset):
    def __init__(self, raw_file:str,end_point_property:str,label_tag:str,\
                        data_type:str,data_dir:str,end_point_name=None,sort_by_date=True,\
                        save=True,load=False,predict=False,**kwargs):
        super().__init__(end_point_property,label_tag,data_type,\
                         data_dir,end_point_name=end_point_name,sort_by_date=sort_by_date,\
                         save=save,load=load,predict=predict,raw_file=raw_file,**kwargs)
    
    def get_raw_samples(self,log:bool=False,negative:bool=False,offset:float=0,predict=False):
        # read raw sdf file
        assert os.path.exists(self.raw_file),'Raw sdf file does not exist!'
        raw_samples = list()
        suppl = Chem.SDMolSupplier(self.raw_file)
        for mol_temp in suppl:
            if mol_temp is not None:
                sample = dict()
                sample['mol'] = mol_temp
                try:
                    sample['label'] = mol_temp.GetProp(self.label_tag)
                except:
                    continue
                if predict:
                    sample[self.end_point_name] = None
                else:
                    try:
                        end_point_value = float(mol_temp.GetProp(self.end_point_property))
                    except:
                        continue
                    if log:
                        end_point_value = float(np.log10(end_point_value))
                    if negative:
                        end_point_value = -end_point_value
                    end_point_value += offset
                    sample[self.end_point_name] = end_point_value
                sample['SMILES'] = Chem.MolToSmiles(mol_temp,isomericSmiles=False)
                raw_samples.append(sample)
        return raw_samples
    

class subdataset(Dataset):
    def __init__(self,data):
        self.data = data
    def save_to_csv(self,file_name:str,property:str):
        if len(self.data)==0:
            return None
        with open(file_name,'w') as f:
            writer = csv.DictWriter(f,fieldnames=['SMILES',property])
            writer.writeheader()
            for data_temp in self.data:
                smiles_temp,y,_ = data_temp
                writer.writerow({'SMILES':smiles_temp,property:y})
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


# Split dataset into train, val, test
# Two modes, random or split by date
def split_dataset(dataset,ratio=[0.6,0.2,0.2],seed=0,mode='random'):
    assert mode in ['random','date'],'Choose a mode from [random,date]'
    assert len(ratio)==3,'Ratio should be a list of 3 numbers'
    assert sum(ratio)==1,'Sum of ratio should be 1'
    train_data = list()
    val_data = list()
    test_data = list()
    if mode=='random':
        # Random split
        test_num = int(len(dataset)*ratio[2])
        val_num = int(len(dataset)*ratio[1])
        np.random.seed(seed)
        test_idx = np.random.choice(len(dataset),test_num,replace=False)
        val_idx = np.random.choice(list(set(range(len(dataset)))-set(test_idx)),val_num,replace=False)
        train_idx = list(set(range(len(dataset)))-set(test_idx)-set(val_idx))
        for idx in train_idx:
            train_data.append(dataset[idx])
        for idx in val_idx:
            val_data.append(dataset[idx])
        for idx in test_idx:
            test_data.append(dataset[idx])
    if mode=='date':
        # Split by date, if seed==0, split from start
        # else, split from a random point
        train_num = int(len(dataset)*ratio[0])
        val_num = int(len(dataset)*ratio[1])
        if seed==0:
            start = 0
        else:
            np.random.seed(seed)
            start = np.random.randint(0,len(dataset))
        for idx in range(train_num):
            idx_temp = (start+idx)%len(dataset)
            train_data.append(dataset[idx_temp])
        for idx in range(train_num,train_num+val_num):
            idx_temp = (start+idx)%len(dataset)
            val_data.append(dataset[idx_temp])
        for idx in range(train_num+val_num,len(dataset)):
            idx_temp = (start+idx)%len(dataset)
            test_data.append(dataset[idx_temp])
    train_data = subdataset(train_data)
    val_data = subdataset(val_data)
    test_data = subdataset(test_data)
    return train_data,val_data,test_data

def split_dataset_by_index(dataset,*index):
    # Split dataset by index
    # index: list of index
    # assert len(dataset)==sum([len(idx) for idx in index]),\
    #     'Sum of index should be equal to the length of dataset'
    splits_dataset = []
    for idx in index:
        data_temp = []
        for i in idx:
            data_temp.append(dataset[i])
        splits_dataset.append(subdataset(data_temp))
    return splits_dataset

# def kholdout_split(dataset:QSARBaseDataset,k,data_path:str,end_point:str,ratio=[0.8,0.0,0.2]):
#     if not os.path.exists(data_path):
#         os.mkdir(data_path)


def scaffold_split(dataset:QSARBaseDataset,k,data_path:str,end_point:str,ratio=[0.8,0.0,0.2]):
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    for i in range(k):
        trian_idx,val_idx,test_idx = dataset.get_scaffold_splits(ratio=ratio,seed=i)
        train_temp,val_temp,test_temp = split_dataset_by_index(dataset,trian_idx,val_idx,test_idx)
        train_f = os.path.join(data_path,'train_'+str(i)+'.csv')
        val_f = os.path.join(data_path,'val_'+str(i)+'.csv')
        test_f = os.path.join(data_path,'test_'+str(i)+'.csv')
        train_temp.save_to_csv(train_f,end_point)
        val_temp.save_to_csv(val_f,end_point)
        test_temp.save_to_csv(test_f,end_point)





