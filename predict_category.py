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

import os
import sys
sys.path.append('descriptor_generation_essentials')

sys.path.append('shape_scoring_reqs')

import joblib
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

import openpyxl


def optimized_hyperparameters(descriptors_file):
    file = open(descriptors_file, 'rt')
    text = file.read()
    file.close()
    sel_descriptors = text.split()
    
    return sel_descriptors

def descriptorslist(dataframe):
    descriptorslist = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 
                       'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 
                       'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11', 
                       'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 
                       'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 
                       'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2', 
                       'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 
                       'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 
                       'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 
                       'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 
                       'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge', 
                       'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 
                       'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 
                       'NumAliphaticRings', 'NumAromaticCarbocycles', 
                       'NumAromaticHeterocycles', 'NumAromaticRings', 
                       'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 
                       'NumRadicalElectrons', 'NumRotatableBonds', 
                       'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 
                       'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1', 
                       'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 
                       'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 
                       'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 
                       'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 
                       'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 
                       'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 
                       'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 
                       'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7', 
                       'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 
                       'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 
                       'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 
                       'VSA_EState9']
    
    mol = Chem.MolFromSmiles(dataframe)
    value = []
    for descriptor in descriptorslist:

        try:
            
            package = 'rdkit.Chem.Descriptors'
            name = descriptor
            desc = getattr(__import__(package, fromlist=[name]), name)

            desstr = descriptor
            function = getattr(Descriptors, desstr)
            calc_desc2 = function(mol)
            value.append(calc_desc2)
        
        except:
            value.append('NA')
        
        
    BalabanJ,BertzCT,Chi0,Chi0n,Chi0v,Chi1,Chi1n,Chi1v,Chi2n,Chi2v,Chi3n,Chi3v,Chi4n,Chi4v,EState_VSA1,EState_VSA10,EState_VSA11,EState_VSA2,EState_VSA3,EState_VSA4,EState_VSA5,EState_VSA6,EState_VSA7,EState_VSA8,EState_VSA9,ExactMolWt,FpDensityMorgan1,FpDensityMorgan2,FpDensityMorgan3,FractionCSP3,HallKierAlpha,HeavyAtomCount,HeavyAtomMolWt,Ipc,Kappa1,Kappa2,Kappa3,LabuteASA,MaxAbsEStateIndex,MaxAbsPartialCharge,MaxEStateIndex,MaxPartialCharge,MinAbsEStateIndex,MinAbsPartialCharge,MinEStateIndex,MinPartialCharge,MolLogP,MolMR,MolWt,NHOHCount,NOCount,NumAliphaticCarbocycles,NumAliphaticHeterocycles,NumAliphaticRings,NumAromaticCarbocycles,NumAromaticHeterocycles,NumAromaticRings,NumHAcceptors,NumHDonors,NumHeteroatoms,NumRadicalElectrons,NumRotatableBonds,NumSaturatedCarbocycles,NumSaturatedHeterocycles,NumSaturatedRings,NumValenceElectrons,PEOE_VSA1,PEOE_VSA10,PEOE_VSA11,PEOE_VSA12,PEOE_VSA13,PEOE_VSA14,PEOE_VSA2,PEOE_VSA3,PEOE_VSA4,PEOE_VSA5,PEOE_VSA6,PEOE_VSA7,PEOE_VSA8,PEOE_VSA9,RingCount,SMR_VSA1,SMR_VSA10,SMR_VSA2,SMR_VSA3,SMR_VSA4,SMR_VSA5,SMR_VSA6,SMR_VSA7,SMR_VSA8,SMR_VSA9,SlogP_VSA1,SlogP_VSA10,SlogP_VSA11,SlogP_VSA12,SlogP_VSA2,SlogP_VSA3,SlogP_VSA4,SlogP_VSA5,SlogP_VSA6,SlogP_VSA7,SlogP_VSA8,SlogP_VSA9,TPSA,VSA_EState1,VSA_EState10,VSA_EState2,VSA_EState3,VSA_EState4,VSA_EState5,VSA_EState6,VSA_EState7,VSA_EState8,VSA_EState9 = value
    
    return pd.Series([BalabanJ,BertzCT,Chi0,Chi0n,Chi0v,Chi1,Chi1n,Chi1v,Chi2n,Chi2v,Chi3n,Chi3v,Chi4n,Chi4v,EState_VSA1,EState_VSA10,EState_VSA11,EState_VSA2,EState_VSA3,EState_VSA4,EState_VSA5,EState_VSA6,EState_VSA7,EState_VSA8,EState_VSA9,ExactMolWt,FpDensityMorgan1,FpDensityMorgan2,FpDensityMorgan3,FractionCSP3,HallKierAlpha,HeavyAtomCount,HeavyAtomMolWt,Ipc,Kappa1,Kappa2,Kappa3,LabuteASA,MaxAbsEStateIndex,MaxAbsPartialCharge,MaxEStateIndex,MaxPartialCharge,MinAbsEStateIndex,MinAbsPartialCharge,MinEStateIndex,MinPartialCharge,MolLogP,MolMR,MolWt,NHOHCount,NOCount,NumAliphaticCarbocycles,NumAliphaticHeterocycles,NumAliphaticRings,NumAromaticCarbocycles,NumAromaticHeterocycles,NumAromaticRings,NumHAcceptors,NumHDonors,NumHeteroatoms,NumRadicalElectrons,NumRotatableBonds,NumSaturatedCarbocycles,NumSaturatedHeterocycles,NumSaturatedRings,NumValenceElectrons,PEOE_VSA1,PEOE_VSA10,PEOE_VSA11,PEOE_VSA12,PEOE_VSA13,PEOE_VSA14,PEOE_VSA2,PEOE_VSA3,PEOE_VSA4,PEOE_VSA5,PEOE_VSA6,PEOE_VSA7,PEOE_VSA8,PEOE_VSA9,RingCount,SMR_VSA1,SMR_VSA10,SMR_VSA2,SMR_VSA3,SMR_VSA4,SMR_VSA5,SMR_VSA6,SMR_VSA7,SMR_VSA8,SMR_VSA9,SlogP_VSA1,SlogP_VSA10,SlogP_VSA11,SlogP_VSA12,SlogP_VSA2,SlogP_VSA3,SlogP_VSA4,SlogP_VSA5,SlogP_VSA6,SlogP_VSA7,SlogP_VSA8,SlogP_VSA9,TPSA,VSA_EState1,VSA_EState10,VSA_EState2,VSA_EState3,VSA_EState4,VSA_EState5,VSA_EState6,VSA_EState7,VSA_EState8,VSA_EState9])

def calc_all_descriptors(dataframe):
    
    rdmoldescriptors = ['CalcAsphericity', 'CalcAUTOCORR2D', 'CalcAUTOCORR3D', 
                    'CalcCrippenDescriptors', 'CalcEccentricity', 'CalcExactMolWt', 
                    'CalcFractionCSP3', 'CalcGETAWAY', 'CalcHallKierAlpha', 
                    'CalcInertialShapeFactor', 'CalcKappa1', 'CalcKappa2', 
                    'CalcKappa3', 'CalcLabuteASA', 'CalcMORSE', 'CalcMolFormula', 
                    'CalcNPR1', 'CalcNPR2', 'CalcNumAliphaticCarbocycles', 
                    'CalcNumAliphaticHeterocycles', 'CalcNumAliphaticRings', 
                    'CalcNumAmideBonds', 'CalcNumAromaticCarbocycles', 
                    'CalcNumAromaticHeterocycles', 'CalcNumAromaticRings', 
                    'CalcNumAtomStereoCenters', 'CalcNumBridgeheadAtoms', 
                    'CalcNumHBA', 'CalcNumHBD', 'CalcNumHeteroatoms', 
                    'CalcNumHeterocycles', 'GetMorganFingerprintAsBitVect',
                    'CalcNumLipinskiHBA', 
                    'CalcNumLipinskiHBD', 'CalcNumRings', 'CalcNumRotatableBonds', 
                    'CalcNumSaturatedCarbocycles', 'CalcNumSaturatedHeterocycles', 
                    'CalcNumSaturatedRings', 'CalcNumSpiroAtoms', 'CalcPBF', 
                    'CalcPMI1', 'CalcPMI2', 'CalcPMI3', 'CalcRDF', 
                    'CalcRadiusOfGyration', 'CalcSpherocityIndex', 'CalcTPSA', 
                    'CalcWHIM', 'GetHashedTopologicalTorsionFingerprintAsBitVect', 
                    'GetMACCSKeysFingerprint',  
                    'GetUSR', 'GetUSRCAT'] 
                    ##REMOVED 'GetConnectivityInvariants', 'GetFeatureInvariants'
    
    #rdmoldescriptors = ['GetMorganFingerprintAsBitVect']
    try:
        
        x = Chem.MolFromSmiles(dataframe)
        x2 = Chem.AddHs(x)
        AllChem.Compute2DCoords(x2)
        AllChem.EmbedMolecule(x2, useRandomCoords=True)
        AllChem.MMFFOptimizeMolecule(x2)
        mol2 = Chem.MolToMolBlock(x2)
        mol = Chem.MolFromMolBlock(mol2)
    
    except:
        
        mol2 = 'error'
        mol = 'error'
        
    value = []
    des_label = []

    for descriptor6 in rdmoldescriptors:
        
        try:

            package = 'rdkit.Chem.rdMolDescriptors'
            name = descriptor6
            desc = getattr(__import__(package, fromlist=[name]), name)
            desstr = str(descriptor6)
            function = getattr(rdMolDescriptors, desstr)

            if (descriptor6 == 'GetMorganFingerprintAsBitVect'):
                ints = function(mol, 6, nBits=2048)
                calc_desc1 = np.array(ints)

            elif (descriptor6 == 'GetHashedTopologicalTorsionFingerprintAsBitVect'):
                ints = function(mol, targetSize = 6, nBits=2048)
                calc_desc1 = np.array(ints)
                #print(calc_desc1)

            elif (descriptor6 == 'GetMACCSKeysFingerprint'):
                ints = function(mol)
                calc_desc1 = np.array(ints)

            else:
                calc_desc1 = function(mol)

        except:
            calc_desc1 = 'error'
           # print('error')

        
        des_label.append(desstr)
        value.append(calc_desc1)

    CalcAsphericity, CalcAUTOCORR2D, CalcAUTOCORR3D, CalcCrippenDescriptors, CalcEccentricity, CalcExactMolWt, CalcFractionCSP3, CalcGETAWAY, CalcHallKierAlpha, CalcInertialShapeFactor, CalcKappa1, CalcKappa2, CalcKappa3, CalcLabuteASA, CalcMORSE, CalcMolFormula, CalcNPR1, CalcNPR2, CalcNumAliphaticCarbocycles, CalcNumAliphaticHeterocycles, CalcNumAliphaticRings, CalcNumAmideBonds, CalcNumAromaticCarbocycles, CalcNumAromaticHeterocycles, CalcNumAromaticRings, CalcNumAtomStereoCenters, CalcNumBridgeheadAtoms, CalcNumHBA, CalcNumHBD, CalcNumHeteroatoms, CalcNumHeterocycles, GetMorganFingerprintAsBitVect, CalcNumLipinskiHBA, CalcNumLipinskiHBD, CalcNumRings, CalcNumRotatableBonds, CalcNumSaturatedCarbocycles, CalcNumSaturatedHeterocycles, CalcNumSaturatedRings, CalcNumSpiroAtoms, CalcPBF, CalcPMI1, CalcPMI2, CalcPMI3, CalcRDF, CalcRadiusOfGyration, CalcSpherocityIndex, CalcTPSA, CalcWHIM, GetHashedTopologicalTorsionFingerprintAsBitVect, GetMACCSKeysFingerprint, GetUSR, GetUSRCAT = value    

    return pd.Series([CalcAsphericity, CalcAUTOCORR2D, CalcAUTOCORR3D, CalcCrippenDescriptors, CalcEccentricity, CalcExactMolWt, CalcFractionCSP3, CalcGETAWAY, CalcHallKierAlpha, CalcInertialShapeFactor, CalcKappa1, CalcKappa2, CalcKappa3, CalcLabuteASA, CalcMORSE, CalcMolFormula, CalcNPR1, CalcNPR2, CalcNumAliphaticCarbocycles, CalcNumAliphaticHeterocycles, CalcNumAliphaticRings, CalcNumAmideBonds, CalcNumAromaticCarbocycles, CalcNumAromaticHeterocycles, CalcNumAromaticRings, CalcNumAtomStereoCenters, CalcNumBridgeheadAtoms, CalcNumHBA, CalcNumHBD, CalcNumHeteroatoms, CalcNumHeterocycles, GetMorganFingerprintAsBitVect, CalcNumLipinskiHBA, CalcNumLipinskiHBD, CalcNumRings, CalcNumRotatableBonds, CalcNumSaturatedCarbocycles, CalcNumSaturatedHeterocycles, CalcNumSaturatedRings, CalcNumSpiroAtoms, CalcPBF, CalcPMI1, CalcPMI2, CalcPMI3, CalcRDF, CalcRadiusOfGyration, CalcSpherocityIndex, CalcTPSA, CalcWHIM, GetHashedTopologicalTorsionFingerprintAsBitVect, GetMACCSKeysFingerprint, GetUSR, GetUSRCAT, mol2, des_label])

def construct_scoring_array(dataframe):
    descriptors5 = ['BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
                   'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v','EState_VSA1',
                   'EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3',
                   'EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8',
                   'EState_VSA9','ExactMolWt','FpDensityMorgan1','FpDensityMorgan2',
                   'FpDensityMorgan3','FractionCSP3','HallKierAlpha','HeavyAtomCount',
                   'HeavyAtomMolWt','Ipc','Kappa1','Kappa2','Kappa3','LabuteASA',
                   'MaxAbsEStateIndex','MaxAbsPartialCharge','MaxEStateIndex',
                   'MaxPartialCharge','MinAbsEStateIndex','MinAbsPartialCharge',
                   'MinEStateIndex','MinPartialCharge','MolLogP','MolMR','MolWt',
                   'NHOHCount','NOCount','NumAliphaticCarbocycles',
                   'NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles',
                   'NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors',
                   'NumHDonors','NumHeteroatoms','NumRadicalElectrons',
                   'NumRotatableBonds','NumSaturatedCarbocycles',
                   'NumSaturatedHeterocycles','NumSaturatedRings','NumValenceElectrons',
                   'PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13',
                   'PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5',
                   'PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','RingCount',
                   'SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5',
                   'SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SlogP_VSA1',
                   'SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3',
                   'SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8',
                   'SlogP_VSA9','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',
                   'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
                   'VSA_EState8','VSA_EState9','CalcAsphericity', 'CalcAUTOCORR2D', 
                   'CalcAUTOCORR3D', 'CalcCrippenDescriptors', 'CalcEccentricity', 
                   'CalcExactMolWt', 'CalcFractionCSP3', 'CalcGETAWAY', 
                   'CalcHallKierAlpha', 'CalcInertialShapeFactor', 'CalcKappa1', 
                   'CalcKappa2', 'CalcKappa3', 'CalcLabuteASA', 'CalcMORSE', 
                   'CalcMolFormula', 'CalcNPR1', 'CalcNPR2', 
                   'CalcNumAliphaticCarbocycles', 'CalcNumAliphaticHeterocycles', 
                   'CalcNumAliphaticRings', 'CalcNumAmideBonds', 
                   'CalcNumAromaticCarbocycles', 'CalcNumAromaticHeterocycles', 
                   'CalcNumAromaticRings', 'CalcNumAtomStereoCenters', 
                   'CalcNumBridgeheadAtoms', 'CalcNumHBA', 'CalcNumHBD', 
                   'CalcNumHeteroatoms', 'CalcNumHeterocycles', 
                   'GetMorganFingerprintAsBitVect', 'CalcNumLipinskiHBA', 
                   'CalcNumLipinskiHBD', 'CalcNumRings', 'CalcNumRotatableBonds', 
                   'CalcNumSaturatedCarbocycles', 'CalcNumSaturatedHeterocycles', 
                   'CalcNumSaturatedRings', 'CalcNumSpiroAtoms', 'CalcPBF', 
                   'CalcPMI1', 'CalcPMI2', 'CalcPMI3', 'CalcRDF', 
                   'CalcRadiusOfGyration', 'CalcSpherocityIndex', 'CalcTPSA', 
                   'CalcWHIM', 'GetHashedTopologicalTorsionFingerprintAsBitVect', 
                   'GetMACCSKeysFingerprint', 'GetUSR', 'GetUSRCAT']
    
    array = dataframe
    array_values = []
    array_id = []

    o_hyperparameters = optimized_hyperparameters(DESCRIPTORS_FILE)  
    num_of_desc = len(array)
    
    for num in range(num_of_desc):
        des_array = array[num]
        #print(des_array)
        
        if (isinstance(des_array, (list, tuple, np.ndarray)) == False):
            descriptor = descriptors5[num]
         #   print(descriptors5[num])
            label = descriptor+'_0'
            
            if label in o_hyperparameters:
                array_values.append(array[num])
                array_id.append(label)

            else:
                continue
        else:
            descriptor = descriptors5[num]        
            length_of_desc = len(des_array)
    
            for bit in range(length_of_desc):
            
                position = des_array[bit]
                descriptor = descriptors5[num]
                label = descriptor+'_'+str(bit)
                
                if label in o_hyperparameters:
                
                    array_values.append(des_array[bit])
                    array_id.append(label)

                else:
                    continue
                    
    return pd.Series([array_values, array_id])

def score_FP(fps):
    rf = joblib.load(MODEL_FILE).set_params(n_jobs=1)
    try:
        pred_num = rf.predict([fps])
        cat_id = CATEGORIES.loc[CATEGORIES['ID'] == pred_num[0], 'category'].iloc[0]
    except:
        pred_num = 'none'
        cat_id ='none'
    return pd.Series([pred_num, cat_id])

descriptors_final = ['BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
               'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v','EState_VSA1',
               'EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3',
               'EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8',
               'EState_VSA9','ExactMolWt','FpDensityMorgan1','FpDensityMorgan2',
               'FpDensityMorgan3','FractionCSP3','HallKierAlpha','HeavyAtomCount',
               'HeavyAtomMolWt','Ipc','Kappa1','Kappa2','Kappa3','LabuteASA',
               'MaxAbsEStateIndex','MaxAbsPartialCharge','MaxEStateIndex',
               'MaxPartialCharge','MinAbsEStateIndex','MinAbsPartialCharge',
               'MinEStateIndex','MinPartialCharge','MolLogP','MolMR','MolWt',
               'NHOHCount','NOCount','NumAliphaticCarbocycles',
               'NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles',
               'NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors',
               'NumHDonors','NumHeteroatoms','NumRadicalElectrons',
               'NumRotatableBonds','NumSaturatedCarbocycles',
               'NumSaturatedHeterocycles','NumSaturatedRings','NumValenceElectrons',
               'PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13',
               'PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5',
               'PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','RingCount',
               'SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5',
               'SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SlogP_VSA1',
               'SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3',
               'SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8',
               'SlogP_VSA9','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',
               'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
               'VSA_EState8','VSA_EState9','CalcAsphericity', 'CalcAUTOCORR2D', 
               'CalcAUTOCORR3D', 'CalcCrippenDescriptors', 'CalcEccentricity', 
               'CalcExactMolWt', 'CalcFractionCSP3', 'CalcGETAWAY', 
               'CalcHallKierAlpha', 'CalcInertialShapeFactor', 'CalcKappa1', 
               'CalcKappa2', 'CalcKappa3', 'CalcLabuteASA', 'CalcMORSE', 
               'CalcMolFormula', 'CalcNPR1', 'CalcNPR2', 
               'CalcNumAliphaticCarbocycles', 'CalcNumAliphaticHeterocycles', 
               'CalcNumAliphaticRings', 'CalcNumAmideBonds', 
               'CalcNumAromaticCarbocycles', 'CalcNumAromaticHeterocycles', 
               'CalcNumAromaticRings', 'CalcNumAtomStereoCenters', 
               'CalcNumBridgeheadAtoms', 'CalcNumHBA', 'CalcNumHBD', 
               'CalcNumHeteroatoms', 'CalcNumHeterocycles', 
               'GetMorganFingerprintAsBitVect', 'CalcNumLipinskiHBA', 
               'CalcNumLipinskiHBD', 'CalcNumRings', 'CalcNumRotatableBonds', 
               'CalcNumSaturatedCarbocycles', 'CalcNumSaturatedHeterocycles', 
               'CalcNumSaturatedRings', 'CalcNumSpiroAtoms', 'CalcPBF', 
               'CalcPMI1', 'CalcPMI2', 'CalcPMI3', 'CalcRDF', 
               'CalcRadiusOfGyration', 'CalcSpherocityIndex', 'CalcTPSA', 
               'CalcWHIM', 'GetHashedTopologicalTorsionFingerprintAsBitVect', 
               'GetMACCSKeysFingerprint', 'GetUSR', 'GetUSRCAT']

# +
#COMPOUNDS = os.path.join(COMPOUND_PATH, 'generator_092920.csv')
# -

workers = 8 ##CPUS
n = 50000     ##chunk row size
COMPOUND_PATH = 'descriptor_generation_essentials/shape_scoring_reqs/'
OUTPUT_DESCRIPTOR = 'initial_output'
MODEL_PATH = 'descriptor_generation_essentials/shape_scoring_reqs/'
MODEL_FILE = os.path.join(MODEL_PATH, 'shape_category_OPTIMIZED_ALL_22620.joblib')
DESCRIPTORS_FILE = os.path.join(MODEL_PATH, 'shape_category_descriptors_LATEST.log')
CATEGORIES = pd.read_csv('descriptor_generation_essentials/shape_scoring_reqs/shape_catagories_NEW.csv')
rf = joblib.load(MODEL_FILE).set_params(n_jobs=1)

def predict_score(smiles):

    total = pd.DataFrame(smiles)
    total.columns = ['SMILES']

    list_df = [total[i:i+n] for i in range(0,total.shape[0],n)]
    total_split_dfs = len(list_df)
    
    for x in range(0, total_split_dfs):

        total_top = list_df[x].copy()

        total_top[['BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v','Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','Chi4v','EState_VSA1','EState_VSA10','EState_VSA11','EState_VSA2','EState_VSA3','EState_VSA4','EState_VSA5','EState_VSA6','EState_VSA7','EState_VSA8','EState_VSA9','ExactMolWt','FpDensityMorgan1','FpDensityMorgan2','FpDensityMorgan3','FractionCSP3','HallKierAlpha','HeavyAtomCount','HeavyAtomMolWt','Ipc','Kappa1','Kappa2','Kappa3','LabuteASA','MaxAbsEStateIndex','MaxAbsPartialCharge','MaxEStateIndex','MaxPartialCharge','MinAbsEStateIndex','MinAbsPartialCharge','MinEStateIndex','MinPartialCharge','MolLogP','MolMR','MolWt','NHOHCount','NOCount','NumAliphaticCarbocycles','NumAliphaticHeterocycles','NumAliphaticRings','NumAromaticCarbocycles','NumAromaticHeterocycles','NumAromaticRings','NumHAcceptors','NumHDonors','NumHeteroatoms','NumRadicalElectrons','NumRotatableBonds','NumSaturatedCarbocycles','NumSaturatedHeterocycles','NumSaturatedRings','NumValenceElectrons','PEOE_VSA1','PEOE_VSA10','PEOE_VSA11','PEOE_VSA12','PEOE_VSA13','PEOE_VSA14','PEOE_VSA2','PEOE_VSA3','PEOE_VSA4','PEOE_VSA5','PEOE_VSA6','PEOE_VSA7','PEOE_VSA8','PEOE_VSA9','RingCount','SMR_VSA1','SMR_VSA10','SMR_VSA2','SMR_VSA3','SMR_VSA4','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA8','SMR_VSA9','SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2','SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7','SlogP_VSA8','SlogP_VSA9','TPSA','VSA_EState1','VSA_EState10','VSA_EState2','VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7','VSA_EState8','VSA_EState9']] = total_top['SMILES'].apply(descriptorslist)
        total_top[['CalcAsphericity', 'CalcAUTOCORR2D', 'CalcAUTOCORR3D', 'CalcCrippenDescriptors', 'CalcEccentricity', 'CalcExactMolWt', 'CalcFractionCSP3', 'CalcGETAWAY', 'CalcHallKierAlpha', 'CalcInertialShapeFactor', 'CalcKappa1', 'CalcKappa2', 'CalcKappa3', 'CalcLabuteASA', 'CalcMORSE', 'CalcMolFormula', 'CalcNPR1', 'CalcNPR2', 'CalcNumAliphaticCarbocycles', 'CalcNumAliphaticHeterocycles', 'CalcNumAliphaticRings', 'CalcNumAmideBonds', 'CalcNumAromaticCarbocycles', 'CalcNumAromaticHeterocycles', 'CalcNumAromaticRings', 'CalcNumAtomStereoCenters', 'CalcNumBridgeheadAtoms', 'CalcNumHBA', 'CalcNumHBD', 'CalcNumHeteroatoms', 'CalcNumHeterocycles', 'GetMorganFingerprintAsBitVect', 'CalcNumLipinskiHBA', 'CalcNumLipinskiHBD', 'CalcNumRings', 'CalcNumRotatableBonds', 'CalcNumSaturatedCarbocycles', 'CalcNumSaturatedHeterocycles', 'CalcNumSaturatedRings', 'CalcNumSpiroAtoms', 'CalcPBF', 'CalcPMI1', 'CalcPMI2', 'CalcPMI3', 'CalcRDF', 'CalcRadiusOfGyration', 'CalcSpherocityIndex', 'CalcTPSA', 'CalcWHIM', 'GetHashedTopologicalTorsionFingerprintAsBitVect', 'GetMACCSKeysFingerprint', 'GetUSR', 'GetUSRCAT','mol2','des_label']] = total_top['SMILES'].apply(calc_all_descriptors)
        total_top[['array_values', 'array_id']] = total_top[descriptors_final].apply(construct_scoring_array, axis=1)

        fp_array = total_top[['SMILES','array_values','array_id']].copy()

        fp_array[['pred_num', 'pred_cat']] = fp_array['array_values'].apply(score_FP)

        output = fp_array[['SMILES','array_values','pred_cat']]
        return output
