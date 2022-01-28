# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys
sys.path.append('descriptor_generation_essentials')
from math import exp,log
from rdkit import Chem, DataStructs
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#import numpy as np
import torch
import autograd.numpy as np
import pickle

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
from autograd.extend import primitive, defvjp
from rdkit.Chem import rdqueries
from rdkit import Chem

import openpyxl
import torch.nn as nn
from rdkit import Chem
from collections import namedtuple
# -

import joblib
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, PandasTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors
from rdkit.Chem import SaltRemover
from rdkit.Chem import PandasTools
from rdkit.Chem import QED
from rdkit.Chem import Fragments

from sklearn import preprocessing 
import scipy
import argparse
import json
import os
import warnings
import logging
import importlib
import pandas as pd
from collections import defaultdict
from aizynthfinder.aizynthfinder import AiZynthFinder
from aizynthfinder.utils.logging import logger, setup_logger
import predict_category
import torch


# +
def shape_score(smiles):
    result = predict_category.predict_score(smiles)
    result = result['pred_cat'].iloc[0]
    return result
        
def CalculateLogD(mol):
    try:
        mol = Chem.MolFromSmiles(mol)
        res=calc.CalculateLogD(mol)
        if isinstance(res, float):
            return res
        else:
            return('error')
    except:
        return('error')

def CalculateLogP(mol): 
    try:
        mol = Chem.MolFromSmiles(mol)
        res=calc.CalculateLogP(mol)
        if isinstance(res, float):
            return res
        else:
            return('error')
    except:
        return('error')

def CalculateAcid(mol):
    try:
        mol = Chem.MolFromSmiles(mol)
        result = calc.CheckAcid(mol)
        if result == 'base':
            return 1
        else:
            return 0
    except:
        return('error')

def CalculatepKa(mol):
    try:
        mol = Chem.MolFromSmiles(mol)
        res = calc.CalculatepKa(mol)
        if isinstance(res, float):
            return res
        else:
            return('error')
    except:
        return('error')

def CalculateFsp3(mol):
    try:
        mol = Chem.MolFromSmiles(mol)
        res= calc.CalculateFsp3(mol)
        if isinstance(res, float):
            return res
        else:
            return('error')
    except:
        return('error')

def CalculateHetCarbonRatio(mol):
    try:
        mol = Chem.MolFromSmiles(mol)
        res=calc.CalculateHetCarbonRatio(mol)
        if isinstance(res, float):
            return res
        else:
            return('error')
    except:
        return('error')
        


# +
def ads(x,a,b,c,d,e,f,dx_max):

    # Asymmetric Double Sigmoidal functions
    
    return((a+(b/(1+exp(-1*(x-c+d/2)/e))*(1-1/(1+exp(-1*(x-c-d/2)/f)))))/dx_max)

def ads_params():
    # ADS parameter sets for 8 molecular properties
    params = { 'MW'    : {  'A'    : 0.2224,
                            'B'    : 3625.3240,
                            'C'    : 396.2089,
                            'D'    : -233.5103,
                            'E'    : 57.4179,
                            'F'    : 38.4751,
                            'DMAX' : 32.68
                          },
               'ALOGP' : {  'A'    : 3.172690585,
                            'B'    : 137.8624751,
                            'C'    : 2.534937431,
                            'D'    : 4.581497897,
                            'E'    : 0.822739154,
                            'F'    : 0.576295591,
                            'DMAX' : 131.31866035,
                          },
               'HBA'   : {  'A'    : 4.5657,
                            'B'    : 138.6334,
                            'C'    : 4.2816,
                            'D'    : 5.5734,
                            'E'    : 0.1548,
                            'F'    : 0.6764,
                            'DMAX' : 143.0
                          },
               'HBD'   : {  'A'    : 1.618662227,
                            'B'    : 1010.051101,
                            'C'    : 0.985094388,
                            'D'    : 0.000000000001,
                            'E'    : 0.713820843,
                            'F'    : 0.920922555,
                            'DMAX' : 258.16326158
                          },
               'PSA'   : {  'A'    : 2.4616,
                            'B'    : 145.5062,
                            'C'    : 69.1875,
                            'D'    : -92.6749,
                            'E'    : -23.8381,
                            'F'    : -3.9658,
                            'DMAX' : 141
                          },
               'ROTB'  : {  'A'    : 12.3569,
                            'B'    : 129772.6367,
                            'C'    : 5.3212,
                            'D'    : -6.9402,
                            'E'    : 1.0397,
                            'F'    : 0.7131,
                            'DMAX' : 100
                          },
               'AROM'  : {  'A'    : -2.8282,
                            'B'    : 363.1550,
                            'C'    : 3.5979,
                            'D'    : 0.9020,
                            'E'    : 0.6492,
                            'F'    : 0.3600,
                            'DMAX' : 185.0
                          },
               'KI'    : {  'A'    : 1.7450,
                            'B'    : 120.0451,
                            'C'    : 4.1084,
                            'D'    : 3.9719,
                            'E'    : 0.3909,
                            'F'    : 0.9318,
                            'DMAX' : 112
                          }
                }
    return params
def weights():

    # unweighted "weights" included for ease of modification - set to zero to exclude a term
    unweights={ 'MW'    : 1.0,
                'ALOGP' : 1.0,
                'HBA'   : 1.0,
                'HBD'   : 1.0,
                'PSA'   : 1.0,
                'ROTB'  : 1.0,
                'AROM'  : 1.0,
                'KI'    : 1.0
    }

    weights = { 'MW'    : 0.66,
                'ALOGP' : 0.46,
                'HBA'   : 0.05,
                'HBD'   : 0.61,
                'PSA'   : 0.06,
                'ROTB'  : 0.65,
                'AROM'  : 0.48,
                'KI'    : 0.95
    }
    return (unweights,weights)
def remove_boc(x):
    try:
        reactants_1 = [Chem.MolFromSmiles(x)]
        reaction = AllChem.ReactionFromSmarts('[#7:1]C(=O)OC(C)(C)C>>[#7H:1]')
        products = reaction.RunReactants(reactants_1)
        productmol = products[0][0]
        product = Chem.MolToSmiles(productmol)
        
    except:
        product = x

    try:
        reactants_2 = [Chem.MolFromSmiles(product)]
        reaction = AllChem.ReactionFromSmarts('[#7:1]C(=O)OC(C)(C)C>>[#7H:1]')
        products_2 = reaction.RunReactants(reactants_2)
        productmol_2 = products_2[0][0]
        product_2 = Chem.MolToSmiles(productmol_2)
        
    except:
        product_2 = product
        
        
    return pd.Series([product_2])
def identify_amines(x):
    try:
        mol1 = Chem.MolFromSmiles(x)
        remover = SaltRemover.SaltRemover(defnData="[Cl,Br,HCl]")
        mol = remover.StripMol(mol1)
        
        NH1_patt = Chem.MolFromSmarts('[NX3;H1;!$(NC=O);!$(Nc);!$(NS);!$(NC=S);!$(N-N);!$(N=N)]')
        NH0_patt = Chem.MolFromSmarts('[NX3;H0;!$(NC=O);!$(Nc);!$(NS);!$(NC=S);!$(N-N);!$(N=N)]')
        
        NH2_patt = Chem.MolFromSmarts('[NX3;H2;!$(NC=O);!$(Nc);!$(NS);!$(NC=S);!$(N-N);!$(N=N)]')
        NH2_patt2 = Chem.MolFromSmarts('[NX2;H1;!$(NC=O);!$(Nc);!$(NS);!$(NC=S);!$(N-N);!$(N=N)]')

        
        a = int(mol.HasSubstructMatch(NH1_patt))
        b = int(mol.HasSubstructMatch(NH0_patt))
 
        f = int(mol.HasSubstructMatch(NH2_patt))
        e = int(mol.HasSubstructMatch(NH2_patt2))
        c = f + e
        
        d = a + b + c
 
    except:
        a = ''
        b = ''
        c = ''
        d = ''

    return pd.Series([a,b,c,d])
def calculate_QRD(x):
    try:
        mol = Chem.MolFromSmiles(x)
        QED_props = QED.properties(mol)

        a = QED_props[0]
        b = QED_props[1]
        c = QED_props[2]
        d = QED_props[3]
        e = QED_props[4]
        f = QED_props[5]
        g = QED_props[6]

        k1 = Chem.GraphDescriptors.Kappa1(mol)
        k2 = Chem.GraphDescriptors.Kappa2(mol)
        heavy_atom = Chem.rdchem.Mol.GetNumHeavyAtoms(mol)
        if heavy_atom > 0:
            kier_index = k1*k2/heavy_atom
            h = kier_index
        else:
            kier_index = 0

        p = ads_params()
        (u,w) = weights()

        mw = a
        alogp = b
        hba = c
        hbd = d
        psa = e
        rotb = f
        arom = g
        ki = h  

        desirability_functions = {
        'MW'     : ads(float(mw),p['MW']['A'],p['MW']['B'],p['MW']['C'],p['MW']['D'],p['MW']['E'],p['MW']['F'],p['MW']['DMAX']),   
        'ALOGP'  : ads(float(alogp),p['ALOGP']['A'],p['ALOGP']['B'],p['ALOGP']['C'],p['ALOGP']['D'],p['ALOGP']['E'],p['ALOGP']['F'],p['ALOGP']['DMAX']),
        'HBA'    : ads(int(hba),p['HBA']['A'],p['HBA']['B'],p['HBA']['C'],p['HBA']['D'],p['HBA']['E'],p['HBA']['F'],p['HBA']['DMAX']),
        'HBD'    : ads(int(hbd),p['HBD']['A'],p['HBD']['B'],p['HBD']['C'],p['HBD']['D'],p['HBD']['E'],p['HBD']['F'],p['HBD']['DMAX']),
        'PSA'    : ads(float(psa),p['PSA']['A'],p['PSA']['B'],p['PSA']['C'],p['PSA']['D'],p['PSA']['E'],p['PSA']['F'],p['PSA']['DMAX']),
        'ROTB'   : ads(int(rotb),p['ROTB']['A'],p['ROTB']['B'],p['ROTB']['C'],p['ROTB']['D'],p['ROTB']['E'],p['ROTB']['F'],p['ROTB']['DMAX']),
        'AROM'   : ads(int(arom),p['AROM']['A'],p['AROM']['B'],p['AROM']['C'],p['AROM']['D'],p['AROM']['E'],p['AROM']['F'],p['AROM']['DMAX']),
        'KI'     : ads(float(ki),p['KI']['A'],p['KI']['B'],p['KI']['C'],p['KI']['D'],p['KI']['E'],p['KI']['F'],p['KI']['DMAX'])
        }

        unweighted_numerator = 0
        weighted_numerator = 0

        for df in desirability_functions.keys():

            if desirability_functions[df] < 0:
                des_function = 0.0000000001
            else:
                des_function = desirability_functions[df]

            unweighted_numerator+=u[df]*log(des_function)
            weighted_numerator  +=w[df]*log(des_function)

        # Unweighted QED
        qed_uw  = exp(unweighted_numerator/sum(u.values()))
        i = qed_uw

        # Weighted QED
        qed_w   = exp(weighted_numerator/sum(w.values()))    
        j = qed_w
        
    except:
        a = ''
        b = ''
        c = ''
        d = ''
        e = ''
        f = ''
        g = ''
        h = ''
        i = ''
        j = '' 
    
    return pd.Series([a,b,c,d,e,f,g,h,i,j])
