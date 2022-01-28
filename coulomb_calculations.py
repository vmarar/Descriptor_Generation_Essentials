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

import pandas as pd
import numpy as np
import multiprocessing as mp
import numpy as np
import os
from pandarallel import pandarallel
pandarallel.initialize()
import time
from rdkit import Chem
from functools import partial


# +
def create_eigenvectors(smile):
    
    try:
        mfs = Chem.MolFromSmiles(smile)  # Construct molecule (only heavy atoms)
        mwh = Chem.AddHs(mfs)             # Append hydrogens to the molecule

        AllChem.EmbedMolecule(mwh)        # Create an geometry of the molecule

        try:
        AllChem.MMFFOptimizeMolecule(mwh) # Optimize the geometry of the molecule
        except (ValueError, RuntimeError) as error:
            "none"

        cm = np.array(AllChem.CalcCoulombMat(mwh)) # Construct the Coulomb Matrix

        eigenvalues, eigenvectors = np.linalg.eig(cm) # Diagonalize the CM

        sorted_eigenvalues = list(eigenvalues) # Eigenvalues must be list to 'sort'

        # Sort the eigenvalues in order of "decreasing absolute value".
        # This first sort is done to guarantee the same sort in the case 
        # that two eigenvalues are the same magnitude but different sign.
        sorted_eigenvalues.sort(key = lambda x: -x)
        sorted_eigenvalues.sort(key = lambda x: -abs(x))

        chem_spc_eigen_values = np.array(sorted_eigenvalues) # Back to a np-array

        return chem_spc_eigen_values
    except:
        return None
    
    
def DoPairwiseVectorComparison(EigenvalueVector, constant_vector):
    try:
        distances = (np.linalg.norm(np.subtract(EigenvalueVector, constant_vector)))
    except:
        distances = None
    return distances

def check_length(x):
    try:
        return len(x)
    except:
        return 0

def pad(x, max_length):
    try:
        length = max_length-(len(x)-1)
        x = np.pad(x, (0,length),'constant', constant_values = (0))
        return x
    except:
        return None
