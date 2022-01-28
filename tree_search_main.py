# +
import os, sys
sys.path.append('descriptor_generation_essentials')
sys.path.append('11182021_SMILES_SMARTS_Hierarchy')

from joblib import load
from math import exp,log
from rdkit import Chem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import numpy
import numpy as np
import pickle

import pandas as pd

from rdkit.Chem import Draw
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import SaltRemover

from rdkit.Chem import PandasTools
from rdkit.Chem import QED
from rdkit.Chem import Fragments
from rdkit import ML
import argparse
import psycopg2

from sshtunnel import SSHTunnelForwarder

import openpyxl

from datetime import datetime
from anytree import Node, RenderTree 
from anytree import Node, RenderTree, AsciiStyle, ZigZagGroupIter, findall,findall_by_attr
import multiprocessing as mp
from functools import partial
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# -

# # UTIL FUNCTIONS

# +
#def convert_SMILES_to_SMARTS(x):
#    mol = Chem.MolFromSmiles(x)
#    smarts = Chem.MolToSmarts(mol)
#    return smarts

def convert_SMILES_to_MOL(x):
    mol = Chem.MolFromSmiles(x)
    remover = SaltRemover.SaltRemover(defnData="[Cl,Br,HCl]")
    mol = remover.StripMol(mol1)
    return mol

def convert_SMARTS_to_SMILES(x):
    smarts = overall_df.loc[overall_df['SMARTS']==x, 'SMILES']
    return smarts

def convert_SMILES_to_SMARTS(x):
    smarts = overall_df.loc[overall_df['SMILES']==x, 'SMARTS']
    return smarts

def node_search(level1, level0): 
        try:
            mol = Chem.MolFromSmiles(level1)
            patt = Chem.MolFromSmarts(level0)
            hit_ats = int(mol.HasSubstructMatch(patt))

            if hit_ats > 0:
                return 1
            else:
                return 0
        except:
            return 0
        
def rdkit_tanimoto(mol1, mol2):
    try:
        mol1 =  rdkit.Chem.MolFromSmiles(mol1)
        mol2 =  rdkit.Chem.MolFromSmiles(mol2)
        fp1 = rdkit.Chem.RDKFingerprint(mol1)
        fp2 = rdkit.Chem.RDKFingerprint(mol2)

        tanimoto_score = rdkit.DataStructs.TanimotoSimilarity(fp1,fp2)
        return tanimoto_score
    except:
         return None
        
from ast import literal_eval
def literal(x):
    return literal_eval(x)

def split(final, split_col):
    
    list_cols = {split_col}
    other_cols = list(set(final.columns) - set(list_cols))
    exploded = [final[col].explode() for col in list_cols]
    df2 = pd.DataFrame(dict(zip(list_cols, exploded)))
    df2 = final[other_cols].merge(df2, how="right", left_index=True, right_index=True)
    df2 = df2.reset_index()
    
    return df2
    
def smile_search_outside(input_smile, start_node):
    matches =[]
    for i in list(start_node.children):
        if node_search(input_smile, i.name) != 0:
            matches.append(i.name)
            node_list = list(i.children)
            for j in node_list:
                if node_search(input_smile, j.name) != 0:
                    matches.append(j.name)
                    node_list2 = list(j.children)
                    if len(node_list2) >= 1: 
                        for x in node_list2:
                            if node_search(input_smile, x.name) != 0:
                                matches.append(x.name)
                    
                    
    matches = list(set(matches))
    
    try:
        matches.remove(None)
    except:
        'none'
                    
    for i in range(len(matches)):
            matches[i] = convert_SMARTS_to_SMILES(matches[i])
    
    return matches

def smile_search_outside_clustering(input_smile, cluster_list):
    
    matches= []
    parents = []
    for i in cluster_list:
            for j in list(i.children):
                if node_search(input_smile, j.name) != 0:
                    node_list = list(j.children)
                    parents.append(j.name)
                    for x in node_list:
                        matches.append(x.name)

    matches = list(set(matches))

    try:
        matches.remove(None)
    except:
        'none'

    for i in range(len(matches)):
            matches[i] = convert_SMARTS_to_SMILES(matches[i])
    
    return matches


# -

# # Load Data

# +
# file structure 

# level 0 and level 1
start0 = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/D1/D1_Node_0a_mono_unsub_NH_to_N_nodup.csv')
start1 = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/D1/D1_Node_0a_1a_mono_sub_NH_to_N_nodup.csv')
start2 = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/D1/D1_Node_0a_1b_bicyclic_unsub_NH_to_N_nodup.csv')
#start1a = pd.read_csv('11182021_SMILES_SMARTS_Hierarchy/D1/D1_Node_0a_1a_2a_bicyclic_sub_NH_to_N_nodup.csv')
#start1 = start1.append(start2)

# load 730 wildcards for node 0 search 
wildcard_730 = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/wildcards_mono_connodup_730.csv')

# top right
start43 = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/D2/D2_Node_0a_SMARTS_FGs_2_left_c.txt',sep='/t')
start43_kids = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/D2/D2_Node_0a_1a_FGs_SMARTS_FGs_c.csv',sep=',')

# middle block
start27 = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/3_Ar_SMARTS_SMILES_linker_Ar/1_MM/D1_Node_0a_1c_MM_CN_all_Xe_enumeration__SMARTS_linkers_NH_to_N_nodup.csv')
start27_kids = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/3_Ar_SMARTS_SMILES_linker_Ar/1_MM/D1_Node_0a_1c_2c_MM_CN_all_Xe_enumeration_SMILES_SMARTS_nodup_NH_to_N_nodup.csv')

# wildcards
wild_73 = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/wildcards_mono_connodup_730.csv')
wild_73k = pd.read_csv('descriptor_generation_essentials/11182021_SMILES_SMARTS_Hierarchy/wildcards_bicyclic_connodup_73191.csv')

print('DONE HERE')
# -

# create overall_df for smarts to smiles conversion
overall_df = pd.concat([start0, start1, start2, start27, start27_kids, start43, start43_kids, wild_73, wild_73k])
overall_df['SMILES'] = overall_df['SMILES'].fillna(overall_df['SMARTS'])
start0['SMARTS'] =  start0['SMILES'].apply(lambda x: convert_SMILES_to_SMARTS(x))

# +
# wildcard files and precomputed searches - ORIGINAL FILES 

# 1400 -> 45k
#searched_45k = pd.read_csv()
#searched_45k.columns=['SMARTS','CHILDREN']

# 150k -> 830
searched_830 = pd.read_csv('descriptor_generation_essentials/search_files/searched_150k_832.csv')
searched_830.columns=[0,'SMARTS','CHILDREN']

# 52 -> 730 
searched_730 = pd.read_csv('descriptor_generation_essentials/search_files/searched_52_730.csv')
searched_730.columns=[0,'SMARTS','CHILDREN']

# 1400 -> 73k 
searched_14k = pd.read_csv('descriptor_generation_essentials/search_files/searched_1k_73k.csv')
searched_14k.columns = [0,'SMARTS','CHILDREN']

# 23000 -> 53000
start23_searched = pd.read_csv('descriptor_generation_essentials/search_files/middle_block_searches.csv')
start23_searched.columns=[0,'SMARTS','CHILDREN']

# 150k -> 114k 
searched_150k = pd.read_csv('descriptor_generation_essentials/search_files/114k_150k.csv')
searched_150k.columns = [0, 'SMARTS','CHILDREN'] 

# +
# CLEAN ORIGINAL FILES 
searched_830['SMARTS'] = searched_830['SMARTS'].apply(lambda x: literal(x))
searched_830['CHILDREN'] = searched_830['CHILDREN'].apply(lambda x: literal(x))
searched_830 = split(searched_830,'SMARTS')
searched_830 = pd.merge(start1[['SMILES','SMARTS']], searched_830, left_on='SMILES', right_on='SMARTS')
searched_830 = searched_830[['SMILES','SMARTS_x','CHILDREN']]
searched_830.columns = ['SMILES','SMARTS','CHILDREN']


start27['index'] = start27.index
start23_searched = pd.merge(start27[['index','SMARTS']], start23_searched, left_on='index', right_on='SMARTS')
start23_searched = start23_searched[['SMARTS_x','CHILDREN']]
start23_searched.columns =['SMARTS','CHILDREN']
start23_searched['CHILDREN'] = start23_searched['CHILDREN'].apply(lambda x: literal(x))


searched_150k = searched_150k[searched_150k['CHILDREN']!='[]']
searched_150k['CHILDREN'] = searched_150k['CHILDREN'].apply(lambda x: literal(x))
searched_150k = split(searched_150k, 'CHILDREN')
searched_150k = pd.merge(start1, searched_150k, left_on='SMILES', right_on='CHILDREN')
searched_150k = searched_150k[['SMILES','SMARTS_x','SMARTS_y']]
searched_150k.columns = ['SMILES','SMARTS','CHILDREN']


searched_14k = pd.merge(searched_14k, start2, left_on='SMARTS', right_on='SMILES')
searched_14k = searched_14k[['SMILES','SMARTS_y','CHILDREN']]
searched_14k.columns = ['SMILES', 'SMARTS','CHILDREN']


# -

# # TREE BUILDING FUNCTIONS

# +
# creates tree -  DONEEEE
# 52 -> 150k
# 52 -> 1400
# 1400 -> 45k - removed

def tree_building_1():

    list1 = start0['SMARTS'].tolist()
    start_0 = Node('START')
    
    for i in range(len(list1)):
        print(i)
        a = Node(list1[i], parent=start_0)

        start_smiles = start1['SMILES'].tolist()
        pool = mp.Pool(processes = (mp.cpu_count()-1))
        results = pool.map(partial(node_search, level0=list1[i]), start_smiles)
        pool.close()
        pool.join()

        start1['CHECK'] = results
        # select smiles that have matches to list1[i]
        temp = start1.loc[start1['CHECK']!=0, 'SMARTS' ]
        temp = temp.drop_duplicates()
        temp = temp.tolist()

        keep_temp = []
        if temp:
            for j in range(len(temp)):
                b = Node(temp[j], parent=a)
                
                
        #1.4k        
        start_smiles = start2['SMILES'].tolist()
        pool = mp.Pool(processes = (mp.cpu_count()-1))
        results = pool.map(partial(node_search, level0=list1[i]), start_smiles)
        pool.close()
        pool.join()

        start2['CHECK'] = results
        # select smiles that have matches to list1[i]
        temp = start2.loc[start2['CHECK']!=0, 'SMARTS' ]
        temp = temp.drop_duplicates()
        temp = temp.tolist()

        keep_temp = []
        if temp:
            for j in range(len(temp)):
                b = Node(temp[j], parent=a)
                #searches = searched_45k.loc[searched_45['SMARTS']==temp[j],'CHILDREN'].tolist()
                #for x in searches:
                #    c = Node(x, parent=b)

    return start_0


# +
# creates tree - DONEEEEEE
# 44 -> 114k

def node_string_search(level1, level0): 
    return level0 in level1 

def tree_building_2():
    list1 = start43['SMARTS'].tolist()
    start_043 = Node('START')
    superset = []
    for i in range(len(list1)):
        print(i)
        a = Node(list1[i], parent=start_043)

        start_smiles = start43_kids['SMARTS'].tolist()
        pool = mp.Pool(processes = (mp.cpu_count()-1))
        results = pool.map(partial(node_string_search, level0=list1[i]), start_smiles)
        pool.close()
        pool.join()

        start43_kids['CHECK'] = results
        temp = start43_kids.loc[start43_kids['CHECK']==True, 'SMARTS' ]

        temp = temp.drop_duplicates()
        temp = temp.tolist()
        for j in temp:
            if j not in superset:
                superset.append(j)
            else:
                temp.remove(j)

        if temp:
            for j in range(len(temp)):
                b = Node(temp[j], parent=a)

    return start_043



# +
# Creates tree - OLD because no way to connect 52 -> 27k
# 52 -> 23000 -> 53000

# load pre-searched tree and create it for middle block
def tree_building_middleblock_main_w52():
    list1 = start0['SMARTS'].tolist()
    start_027 = Node('START')
    
    # maps the 52 to the 27000
    for i in range(len(list1)):
        print(i)
        a = Node(list1[i], parent=start_027)
        
        start_smiles = start27['SMILES'].tolist()
        pool = mp.Pool(processes = (mp.cpu_count()-1))
        results = pool.map(partial(node_search, level0=list1[i]), start_smiles)
        pool.close()
        pool.join()

        start27['CHECK'] = results
        # select smiles that have matches to list1[i]
        temp = start27.loc[start27['CHECK']!=0, 'SMARTS' ]
        temp = temp.drop_duplicates()
        temp = temp.tolist()

        keep_temp = []
        if temp:
            for j in range(len(temp)):
                b = Node(temp[j], parent=a)
                # maps presearched children from function tree_building_middleblock_1
                searches = start23_searched.loc[start23_searched['SMARTS']==temp[j], 'CHILDREN'].tolist()
                for x in searches:
                    for child in x:
                        c = Node(child, parent=b)

    return start_027


# -

# load pre-searched tree and create it for middle block - DONEEE
def tree_building_middleblock_main():
    list1 = start27['SMARTS'].tolist()
    start_027 = Node('START')
    
    for i in range(len(list1)):
        b = Node(list1[i], parent=start_027)
        # maps presearched children from function tree_building_middleblock_1
        searches = start23_searched.loc[start23_searched['SMARTS']==list1[i], 'CHILDREN'].tolist()
        for x in searches:
            for child in x:
                c = Node(child, parent=b)

    return start_027


# # CLUSTERING TREES 

# +
# clustering 
# 52 -> 730 
# 150k -> 114k
# 150k -> 830
# 1400 -> 73k 
# -

def cluster_52_730():
    
    list1 = start0['SMARTS'].tolist()
    start_027 = Node('START')
    
    # maps the 52 to the 730
    for i in range(len(list1)):
        print(i)
        a = Node(list1[i], parent=start_027)
        searches = searched_730.loc[searched_730['SMARTS']==list1[i], 'CHILDREN'].tolist()
        for x in searches: 
            b = Node(x, parent=a)
            
    return start_027


def cluster_150k_114k_and_150k_830():
    
    list1 = start1['SMARTS'].tolist()
    start = Node('START')
    
    # maps the 150k to the 114k
    for i in range(len(list1)):
        print(i)
        a = Node(list1[i], parent=start)
        
        searches = searched_150k.loc[searched_150k['SMARTS']==list1[i], 'CHILDREN'].tolist()
        for x in searches: 
            b = Node(x, parent=a)
            
        searches = searched_830.loc[searched_830['SMARTS']==list1[i], 'CHILDREN'].tolist()
        for x in searches: 
            for child in x: 
                b = Node(child, parent=a)
        
        
    return start


def cluster_1k_73k():
    list1 = start2['SMARTS'].tolist()
    start = Node('START')
    
    # maps the 1.4k to the 73k
    for i in range(len(list1)):
        print(i)
        a = Node(list1[i], parent=start)
        searches = searched_14k.loc[searched_14k['SMARTS']==list1[i], 'CHILDREN'].tolist()
        for x in searches: 
            b = Node(x, parent=a)
        
    return start

# # QSAR FUNCTION


# +
# feed in dataset and output identified functional groups and what they are similar to -> USE IN QSAR

# define all smiles that need to be searched in tanimoto_matrix 
all_values= []
all_values.extend(start0['SMILES'].tolist())
all_values.extend(start1['SMILES'].tolist())
all_values.extend(start2['SMILES'].tolist())
values = pd.DataFrame(all_values)
values = values.drop_duplicates()

# build trees necessary for tree_search
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


def initialize_nodes():
    
    # 52->150k,1.4k
    node1 = tree_building_1()
    #43-> 114k
    node2 = tree_building_2()
    # 23k x 53k
    node3 = tree_building_middleblock_main()
    
    list_of_nodes = [node1, node2, node3]
    
    cluster1 =  cluster_52_730()
    cluster2 =  cluster_150k_114k_and_150k_830()
    cluster3 =  cluster_1k_73k()
    
    cluster_list = [cluster1, cluster2, cluster3]
    
    return list_of_nodes, cluster_list


# main funtion to generate functional groups and selected tanimoto scored similar ones 
def generate_fg_hierarchy_descriptors_no_clustering(data, list_of_nodes):
    
    # find fgs for all smile strings in dataset
    def all_fgs(smile):
        keep = []
        for i in list_of_nodes:
            list1 = smile_search_outside(smile, i)
            keep.extend(list1)
        return keep 

    data['FGS'] = data['SMILES'].parallel_apply(lambda x: all_fgs(x))
    
    return data['FGS']


# main funtion to generate functional groups and selected tanimoto scored similar ones 
def generate_fg_hierarchy_descriptors(data, list_of_nodes, cluster_list):
    
    # find fgs for all smile strings in dataset
    def all_fgs(smile):
        keep = []
        for i in listofnodes:
            list1 = smile_search_outside(smile, i)
            keep.extend(list1)
        return keep 

    data['FGS'] = data['SMILES'].parallel_apply(lambda x: all_fgs(x))
    
    # identify all functional groups in the dataset 
    # so that the tanimoto and coulomb matrices are only made as needed
    collect_all_fgs = []
    fgs = data['FGS'].tolist()
    for i in fgs:
        collect_all_fgs.extend(i)
    collect_all_fgs = list(set(collect_all_fgs))
    
    # create tanimoto matrix for only fgs that exist in the dataset
    start_smiles = values[0].tolist()
    tanimoto_matrix = []
    for i in collect_all_fgs:
        print(i)
        current = []
        pool = mp.Pool(processes = (mp.cpu_count()-1))
        results = pool.map(partial(rdkit_tanimoto, mol2=i),start_smiles)
        pool.close()
        pool.join()
        tanimoto_matrix.append(results)
        
    tanimoto_matrix = pd.DataFrame(tanimoto_matrix)
    tanimoto_matrix.columns = start_smiles
    tanimoto_matrix.index = collect_all_fgs
    tanimoto_matrix = tanimoto_matrix.T
    tanimoto_matrix = tanimoto_matrix.fillna(0)
    
    # create coulomb matrix
    #create eigenvectors
    values['eigenvectors'] = values[0].parallel_apply(lambda x: coulomb_calculations.create_eigenvectors(x))
    all_fgs = pd.DataFrame(collect_all_fgs)
    all_fgs['eigenvectors'] = collect_all_fgs[0].parallel_apply(lambda x: coulomb_calculations.create_eigenvectors(x))
    
    # identify pad length for vectors and pad all eigenvectors
    length = []
    all_fgs['length'] = all_fgs['eigenvectors'].apply(lambda x: check_length(x))
    max_length = all_fgs['eigenvector'].sort_values(ascending = False).iloc[0]
    length.append(max_length)
    
    values['length'] = values['eigenvectors'].apply(lambda x: check_length(x))
    max_length = values['eigenvector'].sort_values(ascending = False).iloc[0]
    length.append(max_length)
    
    max_length = max(length)
    all_fgs['eigenvectors'] = all_fgs['eigenvectors'].parallel_apply(lambda x: coulomb_calculations.pad(x, max_length))
    values['eigenvectors'] = values['eigenvectors'].parallel_apply(lambda x: coulomb_calculations.pad(x, max_length))
    value_eigen = values['eigenvectors'].tolist()
    
    # create the actual matrix - 
    coulomb_matrix = []
    for i in all_fgs['eigenvectors'].tolist():
        current = []
        pool = mp.Pool(processes = (mp.cpu_count()-1))
        results = pool.map(partial(coulomb_calculations.DoPairwiseVectorComparison, constant_vector=i), values_eigen)
        pool.close()
        pool.join()
        coulomb_matrix.append(results)
    
    coulomb_matrix = pd.DataFrame(coulomb_matrix)
    # min max normalize values between 0 and 1 - change to mean normalization
    coulomb_matrix=(coulomb_matrix-coulomb_matrix.min())/(coulomb_matrix.max()-coulomb_matrix.min())
    coulomb_matrix.columns = start_smiles
    coulomb_matrix.index = collect_all_fgs
    coulomb_matrix = coulomb_matrix.T
    coulomb_matrix = coulomb_matrix.fillna(0)

    
    # cluster using coulomb matrix - lower the better
    def coulomb_clustered(fgs_list):
            associated = fgs_list.copy()
            for i in fgs_list: 
                coulomb_identified = [i+'_COULOMB' for i in coulomb_matrix.loc[coulomb_matrix[i]<=.2].index.tolist()]
                associated.extend(coulomb_identified)
            return associated

    # cluster using coulomb matrix - higher the better
    def tanimoto_clustered(fgs_list):
            associated = fgs_list.copy()
            for i in fgs_list: 
                tanimoto_identified = [i+'_TANIMOTO' for i in tanimoto_matrix.loc[tanimoto_matrix[i]>.6].index.tolist()]
                associated.extend(tanimoto_identified)
            return associated

    # returns original and additionally identified ones using Tanimoto Matrix
    data['FGS_tanimoto'] = data['FGS'].parallel_apply(lambda x: tanimoto_clustered(x))
    
    # returns original and additionally identified ones using coulomb Matrix
    data['FGS_coulomb'] = data['FGS'].parallel_apply(lambda x: coulomb_clustered(x))
    
    # finds matches from smile through clusters
    data['clustered_fgs'] = data['SMILES'].parallel_apply(lambda x: smile_search_outside_clustering(x, cluster_list))
    
    # combines existing FGS with newly found clustered fgs
    def con(original,x1, x2, y):
        y =[i+'_WILDCARD' for i in y]
        x1 = [i+'_TANIMOTO' for i in y]
        x2 = [i+'_COULOMB' for i in y]
        original.extend(x1)
        original.extend(x2)
        original.extend(y)
        return original
    
    data['FGS'] = data.parallel_apply(lambda x: con(x['FGS'], x['FGS_tanimoto'], x['FGS_coulomb'], x['clustered_fgs']), axis=1)
    
    return data['FGS']


# -

# this is for inputting new data through QSAR Models that have functional groups as variables 
def generate_needed_fg_hierarchy_descriptors(data, desc):
    
    # take list of desc and do a smarts, smile search
    def check_desc(smile, smarts): 
        return node_search(smile, smarts)

    remaining = []
    for i in desc:
        temp = convert_SMILES_to_SMARTS(i)
        if len(temp) >= 1:
            data[i] = data['SMILES'].parallel_apply(lambda x: check_desc(x, temp))
        else:
            remaining.append(i)
        
    return data, remaining
