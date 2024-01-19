#!/usr/bin/env python3
import argparse
import multiprocessing
import os
import pickle
from time import time

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

from pbdd.models.ma_models import mol_assign_GAN
from pbdd.post_processing.sample import assign_mols, assign_mols_match
from pbdd.post_processing.scoring import vina_score


def convert_sdf_to_pdbqt(assign_sdf_path,assign_pdbqt_path):
    if os.path.exists(assign_pdbqt_path):
        return None
    else:
        os.system(f'obabel -isdf {assign_sdf_path} -opdbqt -O {assign_pdbqt_path} -p 7.4')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_lig_file',type=str,\
                        help='path to the original ligand mol2 file')
    parser.add_argument('--pocket_pdbqt',type=str,\
                        help='path to the pocket pdbqt file')
    parser.add_argument('--wkdir',type=str,\
                        help='path to the working directory')

    # sample top parameters
    # # parser.add_argument('--DGT_ckpt_path',type=str,\
    # #                     help='path to the DGT GAN checkpoint')
    # # parser.add_argument('--n_tops',type=int,default=1024,\
    # #                     help='number of topologies to sample')
    # # parser.add_argument('--sample_repeats',type=int,default=100,\
    # #                     help='number of repeats for sampling one stacking')
    # # parser.add_argument('--top_freq_lib',type=str,default=None,\
    #                     help='path to the topology frequency library')
    parser.add_argument('--top_save_path',type=str,default=None,\
                        help='path to save the sampled tops')
    # assgin mol parameters
    parser.add_argument('--assign_method',type=str,default='GAN',choices=['GAN','Matching'],\
                        help='method to assign mols, choose from GAN and Matching')
    parser.add_argument('--mol_assign_ckpt_path',type=str,default=None,\
                        help='path to the mol assign GAN checkpoint')
    parser.add_argument('--n_assigns',type=int,default=100,\
                        help='number of assignments for each top to sample')
    parser.add_argument('--load',action='store_true',\
                        help='whether to load the assigned mols from pickle, do this is'+\
                        'you already have assigned mols')
    parser.add_argument('--save_batch_size',type=int,default=10000,\
                        help='batch size for saving assigned mols')
    parser.add_argument('--frag_lib_path',type=str,default=None,\
                        help='path to the fragment library')
    parser.add_argument('--n_subjobs',type=int,default=1,\
                        help='number of subjobs for assigning mols')
    parser.add_argument('--subjob_rank',type=int,default=0,\
                        help='rank of the current subjob')

    # filtering parameters
    parser.add_argument('--vina_threshold',type=float,default=-8.0,\
                        help='vina score threshold for saving poses')
    parser.add_argument('--qed_threshold',type=float,default=0.4,\
                        help='QED threshold for filtering, value between 0 and 1')
    # parser.add_argument('--sas_threshold',type=float,default=8.0,\
    #                     help='SAS threshold for filtering')
    parser.add_argument('--box_length',type=float,default=30.0,\
                        help='box length for vina docking')
    parser.add_argument('--write_pose',action='store_true',\
                        help='whether to write the pose')

    t0 = time()
    args = parser.parse_args()


    ori_lig_file = args.ori_lig_file
    wkdir = args.wkdir

    pocket_pdbqt = args.pocket_pdbqt
    # DGT_GAN_ckpt_path = args.DGT_ckpt_path
    # n_tops = args.n_tops
    # sample_repeats = args.sample_repeats
    # top_freq_lib = args.top_freq_lib

    top_save_path = args.top_save_path
    if top_save_path is None:
        top_save_path = os.path.join(wkdir,'tops.pkl')   
    assert os.path.exists(top_save_path),f'{top_save_path} does not exist'
    assert args.qed_threshold>=0 and args.qed_threshold<=1,\
        f'qed_threshold should be between 0 and 1, got {args.qed_threshold}'

    assign_method = args.assign_method
    if assign_method=='GAN':
        mol_assign_ckpt_path = args.mol_assign_ckpt_path
        assert os.path.exists(mol_assign_ckpt_path),f'{mol_assign_ckpt_path} does not exist'
        n_assigns = args.n_assigns

    if assign_method=='Matching':
        frag_lib_path = args.frag_lib_path
        assert os.path.exists(frag_lib_path),f'{frag_lib_path} does not exist'
        with open(frag_lib_path,'rb') as f:
            frag_lib = pickle.load(f)

    save_batch_size = args.save_batch_size
    n_subjobs = args.n_subjobs
    subjob_rank = args.subjob_rank

    print('n_subjobs: ',n_subjobs)
    print('subjob_rank: ',subjob_rank)

    mol_assign_dir = os.path.join(wkdir,f'mols_assign_{subjob_rank+1}_of_{n_subjobs}')
    os.makedirs(mol_assign_dir,exist_ok=True)
    mol_assign_path = os.path.join(mol_assign_dir,'mols_assign.pkl')

    vina_threshold = args.vina_threshold

    os.makedirs(wkdir,exist_ok=True)

    # ### Get original ligand info
    assert os.path.exists(ori_lig_file),f'{ori_lig_file} does not exist'
    lig_file_type = ori_lig_file.split('.')[-1]
    if lig_file_type=='pdb':
        ori_mol = Chem.MolFromPDBFile(ori_lig_file, removeHs=False)
    elif lig_file_type=='mol2':
        ori_mol = Chem.MolFromMol2File(ori_lig_file, removeHs=False)
    else:
        raise ValueError(f'unsupported ligand file type: {lig_file_type}')
    num_atoms = ori_mol.GetNumAtoms()
    num_rot = Descriptors.NumRotatableBonds(ori_mol)
    pos = ori_mol.GetConformer().GetPositions()
    pos_min = np.min(pos,axis=0)
    pos_max = np.max(pos,axis=0)
    lig_pos_center = (pos_min+pos_max)/2

    print('Original ligand Info:')
    print('num_atoms: ',num_atoms)
    print('num_rotatble bonds: ',num_rot)
    print('pos_center: ',lig_pos_center)


    if args.load == False:
        with open(top_save_path,'rb') as f:
            top_list = pickle.load(f)


        # get subjob top_list
        n_tops = len(top_list)
        print('number of tops: ',n_tops)

        top_list_subjob = []
        batch_top = int(n_tops/n_subjobs)+1
        start = int(batch_top*subjob_rank)
        end = min(batch_top*(subjob_rank+1),n_tops)
        top_list = top_list[start:end]
        print('number of tops in subjob: ',len(top_list))

        if assign_method == 'GAN':
            mol_assign_trained_model = mol_assign_GAN.load_GAN_from_checkpoint(mol_assign_ckpt_path)
            assigned_mols = assign_mols(top_list,mol_assign_trained_model,\
                                        n_samples=n_assigns,qed_threshold=args.qed_threshold)
        elif assign_method == 'Matching':
            assigned_mols = assign_mols_match(top_list,frag_lib,num_tol=1)

        # save assigned mols as pickle
        with open(mol_assign_path,'wb') as f:
            pickle.dump(assigned_mols,f)
    else:
        #load assigned mols from pickle
        with open(mol_assign_path,'rb') as f:
            assigned_mols = pickle.load(f)
    # calculate SAS and QED and filter based on quentile
    # n_mols = len(assigned_mols)
    # SAS_list = []
    # QED_list = []
    # for i,mol in enumerate(assigned_mols):
    #     sas_temp = calc_sas(mol)
    #     qed_temp = calc_qed(mol)
    #     SAS_list.append([i,sas_temp])
    #     QED_list.append([i,qed_temp])
    #     mol.SetProp('qed',f'{qed_temp:.3f}')
    #     mol.SetProp('sas',f'{sas_temp:.3f}')
    # SAS_list.sort(key=lambda x:x[1])
    # QED_list.sort(key=lambda x:x[1],reverse=True)
    # SAS_keep_n = int(n_mols*(1-args.SAS_quentile))
    # QED_keep_n = int(n_mols*(1-args.QED_quentile))
    # SAS_keep_index = [i for i,_ in SAS_list[:SAS_keep_n]]
    # QED_keep_index = [i for i,_ in QED_list[:QED_keep_n]]
    # keep_index = list(set(SAS_keep_index).intersection(set(QED_keep_index)))

    batch_id = 0
    assign_sdf_path = os.path.join(mol_assign_dir,f'mol_batch_{batch_id:03d}.sdf')
    w = Chem.SDWriter(assign_sdf_path)
    mol_dict = {}
    for i,mol in enumerate(assigned_mols):
        if i%save_batch_size==0 and i>0:
            batch_id += 1
            assign_sdf_path = os.path.join(mol_assign_dir,\
                                        f'mol_batch_{batch_id:03d}.sdf')
            w = Chem.SDWriter(assign_sdf_path)
        name_temp = f'mol_{subjob_rank}_{batch_id:03d}_{i%save_batch_size:04d}'
        mol.SetProp('_Name',name_temp)
        w.write(mol)
        mol_dict[name_temp] = mol
    
    w.close()
    t1 = time()

    # convert sdf to pdbqt, multiprocess
 
    convert_task_list = []
    num_process = os.cpu_count()
    num_process = min(num_process,batch_id+1)
    for i in range(batch_id+1):
        assign_sdf_path = os.path.join(mol_assign_dir,f'mol_batch_{i:03d}.sdf')
        assign_pdbqt_path = os.path.join(mol_assign_dir,f'mol_batch_{i:03d}.pdbqt')
        convert_task_list.append((assign_sdf_path,assign_pdbqt_path))
    with multiprocessing.get_context('fork').Pool(num_process) as pool:
        pool.starmap(convert_sdf_to_pdbqt,convert_task_list)

    # vina scoring and save poses with score<threshold
    pos_dir = os.path.join(wkdir,f'pos_{subjob_rank}')
    os.makedirs(pos_dir,exist_ok=True)

    score_list = []
    for i in range(batch_id+1):
        temp_pdbqt_path = os.path.join(mol_assign_dir,f'mol_batch_{i:03d}.pdbqt')
        # read pdbqt and split to string list
        with open(temp_pdbqt_path,'r') as f:
            lines = f.readlines()
            mol_temp = ''
            mol_str_list = []
            for line in lines:
                if line.startswith('MODEL'):
                    continue
                if line.startswith('ENDMDL'):
                    mol_str_list.append(mol_temp)
                    mol_temp = ''
                    continue
                mol_temp += line
        # vina scoring
        n_mols = len(mol_str_list)
        vina_task_list = []
        num_process = os.cpu_count()-2
        for j,lig_str in enumerate(mol_str_list):
            pose_name = os.path.join(pos_dir,f'mol_{i:03d}_{j:04d}')
            vina_task_list.append((lig_str,pocket_pdbqt,lig_pos_center,\
                                pose_name,vina_threshold,args.box_length,args.write_pose))
        with multiprocessing.get_context('fork').Pool(num_process) as pool:
            score_temp = pool.starmap(vina_score,vina_task_list)
        score_list.append(score_temp)

    # # save scores and ranking

    # collect all scores and corresponding poses
    # print(score_list)
    print(len(score_list))
    collect_scores = []
    for batch_i,scores in enumerate(score_list):
        for mol_j,score in enumerate(scores):
            collect_scores.append([subjob_rank,batch_i,mol_j,score[0]])
    # sort by score
    collect_scores.sort(key=lambda x:x[3])

    # save scores
    score_dir = os.path.join(wkdir,'scores')
    os.makedirs(score_dir,exist_ok=True)
    score_path = os.path.join(score_dir,f'scores_{subjob_rank}.pkl')
    with open(score_path,'wb') as f:
        pickle.dump(collect_scores,f)


    # read top poses and write to sdf (only score<threshold)
    top_mols = []
    n_poses = 0
    # print(collect_scores)
    for _,batch_i,mol_j,score in collect_scores:
        if score>=vina_threshold:
            break
        mol_name = f'mol_{subjob_rank}_{batch_i:03d}_{mol_j:04d}'
        mol_temp = mol_dict[mol_name]
        mol_temp.SetProp('vina_score',str(score))
        top_mols.append(mol_temp)
        n_poses += 1

    # write top poses sdf
    final_dir = os.path.join(wkdir,'final')
    os.makedirs(final_dir,exist_ok=True)
    final_sdf = os.path.join(final_dir,f'{subjob_rank}_top{n_poses}_poses.sdf')
    final_pdbqt = os.path.join(final_dir,f'{subjob_rank}_top{n_poses}_poses.pdbqt')
    sdf_writer = Chem.SDWriter(final_sdf)

    for mol in top_mols:
        sdf_writer.write(mol)
    os.system(f'obabel -isdf {final_sdf} -opdbqt -O {final_pdbqt} -p 7.4')

    t2 = time()
    print(f'assign time: {t1-t0}')
    print(f'scoring time: {t2-t1}')

if __name__ == "__main__":
    main()

