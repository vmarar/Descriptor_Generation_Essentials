import sys
sys.path.append('descriptor_generation_essentials')
import pandas as pd
from sklearn import preprocessing
pd.set_option('display.max_colwidth',50000)
import rdkit.ML.Descriptors.MoleculeDescriptors as Calc
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import f1_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import itertools
import warnings
warnings.filterwarnings('ignore')
import pickle as pickle
import joblib
from rdkit.Chem import AllChem
import scoring, Amine
#import tree_search_main
from pandarallel import pandarallel
from rdkit import Chem
import joblib
from mordred import Calculator, descriptors
from rdkit.Chem import MACCSkeys
# Initialization
pandarallel.initialize()
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler

# +
from PyBioMed.PyMolecule import connectivity
from PyBioMed.PyMolecule import topology
from PyBioMed.PyMolecule.cats2d import CATS2D
from PyBioMed import Pymolecule

def pybiomed_get_keys():
    smi = 'N[C@@H](CO)C(=O)O'
    mol = Pymolecule.PyMolecule()
    mol.ReadMolFromSmile(smi)
    alldes = mol.GetAllDescriptor()
    
    return list(alldes.keys())

def pybiomed_descriptors(smile):
    mol = Pymolecule.PyMolecule()
    mol.ReadMolFromSmile(smile)
    alldes = mol.GetAllDescriptor()
    
    return pd.Series(list(alldes.values()))

def pybiomed_needed_keys(desc):
    smi = 'N[C@@H](CO)C(=O)O'
    mol = Pymolecule.PyMolecule()
    mol.ReadMolFromSmile(smi)
    mol.set_needed_desc(desc)
    needed = mol.GetNeededDescriptors()
    return list(needed.keys())

def pybiomed_needed_descriptors(desc, smi):
    mol = Pymolecule.PyMolecule()
    mol.ReadMolFromSmile(smi)
    mol.set_needed_desc(desc)
    needed = mol.GetNeededDescriptors()

from rdkit.Chem import AllChem
def ecfps_gen(x):
    mol = Chem.MolFromSmiles(x)
    fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol,2)
    return pd.Series(list(fingerprints))

def ecfps_gen_hashed(x):
    mol = Chem.MolFromSmiles(x)
    fingerprints = AllChem.GetHashedMorganFingerprint(mol,2)
    return list(fingerprints)

def maccs_sample():
    x = 'C1CCC1OCC'
    mol = Chem.MolFromSmiles(x)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return len(list(fp))

def maccs_key(x):
    mol = Chem.MolFromSmiles(x)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return pd.Series(list(fp))
 

from PyBioMed.PyDNA.PyDNAac import GetDAC
def dna_desc_names():
    dac = GetDAC('GACTGAACTGCACTTTGGTTTCATATTATTTGCTC', phyche_index=['Twist','Tilt'], all_property=True)
    return list(dac.keys())


def dna_desc(x):
    dac = GetDAC(x, phyche_index=['Twist','Tilt'], all_property=True)
    return pd.Series(list(dac.values()))


def calculate_mordred(x):
        mol = Chem.MolFromSmiles(x)
        return pd.Series(calc(mol))
    
mordred_dict = dict()
calc = Calculator(descriptors, ignore_3D=True)
mordred_columns = list(calc.descriptors)
for i in mordred_columns:
    mordred_dict[str(i)] = i 


# -

def calc_fingerprints(data):
    fingerprint_sets = []

    # Morgan Fingerprint Radius = 3
    def calcfp(mol,radius, nBits=2048, useFeatures=False, useChirality=False):
            mol = Chem.MolFromSmiles(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, useFeatures=useFeatures, useChirality=useChirality)
            fp = pd.Series(np.asarray(fp))
            fp = fp.add_prefix('Bit_radius{}'.format(radius))
            return fp
    
    desc_values = pd.DataFrame(calcfp(data['SMILES'].iloc[0], 3))
    desc_values = desc_values.index.tolist()
    data[desc_values] = data['SMILES'].apply(lambda x: calcfp(x,3))
    fingerprint_sets.append([desc_values])

    # MACCS fingerprints
    maccs_key_names = [f'BitMAC_{i}' for i in range(maccs_sample())]
    data[maccs_key_names] = data['SMILES'].apply(lambda x: maccs_key(x))
    fingerprint_sets.append([maccs_key_names])
    
    from PyFingerprint.All_Fingerprint import get_fingerprint
    
    def calc_all_fps(x, i):
        fps = get_fingerprint(x, fp_type=i ,output='vec')
        fps = pd.Series(np.asarray(fps))
        fps = fps.add_prefix('{}_'.format(i))
        return fps
    
    
    all_fps = ["daylight", "extended", "graph", "pubchem", "estate",
                   "hybridization", "lingo", "klekota-roth", "shortestpath",
                   "signature", "circular", "Morgan", "cdk","rdkit", "maccs", "AtomPair", 
               "TopologicalTorsion", "Avalon","FP2", "FP3", "FP4"]
    
    for i in all_fps:
        try:
            sample = pd.DataFrame(calc_all_fps(data['SMILES'].iloc[0], i))
            sample = sample.index.tolist()
            data[sample] = data['SMILES'].apply(lambda x: calc_all_fps(x, i))
            fingerprint_sets.append(sample)
        except:
            print('ERROR',i)
            
    descs = []
    for i in fingerprint_sets:
        if len(i) == 1:
            descs.append(i[0])
        else:descs.append(i)
                                              
    return data, descs

# +
rdmoldescriptors = ['CalcAUTOCORR2D', 'CalcAUTOCORR3D', 'CalcAsphericity', 'CalcCrippenDescriptors', 
                    'CalcEccentricity', 'CalcExactMolWt', 'CalcFractionCSP3', 'CalcGETAWAY', 
                    'CalcHallKierAlpha', 'CalcInertialShapeFactor', 'CalcKappa1', 'CalcKappa2', 
                    'CalcKappa3', 'CalcLabuteASA', 'CalcMORSE', 'CalcMolFormula', 'CalcNPR1', 'CalcNPR2', 
                    'CalcNumAliphaticCarbocycles', 'CalcNumAliphaticHeterocycles', 'CalcNumAliphaticRings',
                    'CalcNumAmideBonds', 'CalcNumAromaticCarbocycles', 'CalcNumAromaticHeterocycles', 
                    'CalcNumAromaticRings', 'CalcNumAtomStereoCenters', 'CalcNumBridgeheadAtoms',
                    'CalcNumHBA', 'CalcNumHBD', 'CalcNumHeteroatoms', 'CalcNumHeterocycles', 
                    'CalcNumLipinskiHBA', 'CalcNumLipinskiHBD', 'CalcNumRings', 'CalcNumRotatableBonds', 
                    'CalcNumSaturatedCarbocycles', 'CalcNumSaturatedHeterocycles', 'CalcNumSaturatedRings', 
                    'CalcNumSpiroAtoms', 'CalcPBF', 'CalcPMI1', 'CalcPMI2', 'CalcPMI3', 'CalcRDF', 
                    'CalcRadiusOfGyration', 'CalcSpherocityIndex', 'CalcTPSA', 'CalcWHIM',
                    'GetConnectivityInvariants', 'GetFeatureInvariants',
                    'GetHashedTopologicalTorsionFingerprintAsBitVect', 'GetMACCSKeysFingerprint', 
                    'GetMorganFingerprintAsBitVect', 'GetUSR', 'GetUSRCAT']

descriptorslist = ['BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 
                   'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11',
                   'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6', 
                   'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 
                   'FpDensityMorgan2', 'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 
                   'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 
                   'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex', 'MaxPartialCharge', 
                   'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
                   'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 
                   'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 
                   'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 
                   'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 
                   'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1',
                   'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2',
                   'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 
                   'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 
                   'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
                   'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 
                   'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 
                   'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4', 'VSA_EState5', 'VSA_EState6',
                   'VSA_EState7', 'VSA_EState8', 'VSA_EState9']

def calc_descriptors_list(x,i):
        try:
            mol = Chem.MolFromSmiles(x)

            exec('from rdkit.Chem.Descriptors import %s as desc' % (i))
            desstr = i
            function = getattr(Descriptors, desstr)
            calc_desc = function(mol)
            return calc_desc 
        
        except:
    
            pass

        
def calc_rdmol_descriptors(x,i):
        try:
            mol = Chem.MolFromSmiles(x) 
            mol = Chem.AddHs(mol)
            
            AllChem.Compute2DCoords(mol)
            AllChem.EmbedMolecule(mol, useRandomCoords=True)
            AllChem.MMFFOptimizeMolecule(mol)
            
            mol = Chem.MolToMolBlock(mol)
            mol = Chem.MolFromMolBlock(mol)
    
            
            exec('from rdkit.Chem.rdMolDescriptors import %s' % (i))
            desstr = str(i)
            function = getattr(rdMolDescriptors, desstr)
            if (i == 'GetMorganFingerprintAsBitVect'):
                bitvector = function(mol, 3)
                calc_desc = bitvector.ToBitString()

            elif (i == 'GetHashedTopologicalTorsionFingerprintAsBitVect') or (i == 'GetMACCSKeysFingerprint'):
                bitvector = function(mol)
                calc_desc = bitvector.ToBitString()
            else:
                calc_desc = function(mol)
                
            if isinstance(calc_desc,float):
                return calc_desc
            else:
                return np.mean(calc_desc)
        except:
            return 'error'
    


# +
# FUNCTIONAL GROUP SEGEMENTATION ALGORITHIM #1
def merge(mol, marked, aset):
    bset = set()
    for idx in aset:
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            jdx = nbr.GetIdx()
            if jdx in marked:
                marked.remove(jdx)
                bset.add(jdx)
    if not bset:
        return
    merge(mol, marked, bset)
    aset.update(bset)

# atoms connected by non-aromatic double or triple bond to any heteroatom
# c=O should not match (see fig1, box 15).  I think using A instead of * should sort that out?
PATT_DOUBLE_TRIPLE = Chem.MolFromSmarts('A=,#[!#6]')
# atoms in non aromatic carbon-carbon double or triple bonds
PATT_CC_DOUBLE_TRIPLE = Chem.MolFromSmarts('[C,c]=,#C')
# acetal carbons, i.e. sp3 carbons connected to tow or more oxygens, nitrogens or sulfurs; these O, N or S atoms must have only single bonds
PATT_ACETAL = Chem.MolFromSmarts('[CX4](-[O,N,S])-[O,N,S]')
# all atoms in oxirane, aziridine and thiirane rings
PATT_OXIRANE_ETC = Chem.MolFromSmarts('[O,N,S]1CC1')

PATT_TUPLE = (PATT_DOUBLE_TRIPLE, PATT_CC_DOUBLE_TRIPLE, PATT_ACETAL, PATT_OXIRANE_ETC)

def identify_functional_groups(mol):
    marked = set()
#mark all heteroatoms in a molecule, including halogens
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in (6,1): # would we ever have hydrogen?
            marked.add(atom.GetIdx())

#mark the four specific types of carbon atom
    for patt in PATT_TUPLE:
        for path in mol.GetSubstructMatches(patt):
            for atomindex in path:
                marked.add(atomindex)

#merge all connected marked atoms to a single FG
    groups = []
    while marked:
        grp = set([marked.pop()])
        merge(mol, marked, grp)
        groups.append(grp)

#extract also connected unmarked carbon atoms
    ifg = namedtuple('IFG', ['atomIds', 'atoms', 'type'])
    ifgs = []
    for g in groups:
        uca = set()
        for atomidx in g:
            for n in mol.GetAtomWithIdx(atomidx).GetNeighbors():
                if n.GetAtomicNum() == 6:
                    uca.add(n.GetIdx())
        ifgs.append(ifg(atomIds=tuple(list(g)), atoms=Chem.MolFragmentToSmiles(mol, g, canonical=True), type=Chem.MolFragmentToSmiles(mol, g.union(uca),canonical=True)))
    return ifgs

def find_fg(smile):
    mol = Chem.MolFromSmiles(smile)
    ifgs = identify_functional_groups(mol)
    ifg_list = []
    for i in ifgs:
        ifg_list.append(i[2])
    return ifg_list


# -

# FUNCTIONAL GROUP SEGEMENTATION ALGORITHIM #2
from rdkit.Chem.BRICS import BRICSDecompose
def BRICS_implementation(x):
    mol = Chem.MolFromSmiles(x)
    res = list(BRICSDecompose(mol,keepNonLeafNodes=True,minFragmentSize=1, returnMols=True))
    smis = [Chem.MolToSmiles(x,True) for x in res]
    return smis


def create_all_fragments(data):
    
    # generating functional group descriptors - ALGORITHIM  #1
    before = data.columns.tolist()
    data['FGS1'] = data['SMILES'].parallel_apply(lambda x: find_fg(x))

    # generating functional group descriptors - ALGORITHIM  #2
    data['FGS2'] = data['SMILES'].parallel_apply(lambda x: BRICS_implementation(x))
    print('done')
    
    # generating functional group descriptors - ALGORITHIM  #3
    list_of_nodes, cluster_list = tree_search_main.initalize_nodes()
    data['FGS3'] = tree_search_main.generate_fg_hierarchy_descriptors(data, list_of_nodes, cluster_list)
    print('done')
    
    # combine all FGS results
    def extend_all(x1,x2):
        combined = []
        combined.extend(x1)
        combined.extend(x2)
        return list(set(combined))
    
    data['fgs'] = data.parallel_apply(lambda x: extend_all(x['FGS1'], x['FGS2']), axis=1)
    
    # one hot encode fgs
    data = data.assign(**pd.get_dummies(data['fgs'].apply(lambda x:pd.Series(x)).stack().reset_index(level=1,drop=True)).sum(level=0))
    data = data.drop(['FGS1','FGS2','fgs'],axis=1)
    after = data.columns.tolist()
    
    fg_cols = []
    for i in after:
        if i not in before:
            fg_cols.append(i)

    data[fg_cols] = data[fg_cols].apply(lambda x: np.int8(x))
    
    return data, fg_cols


# # GENERATE ALL VARIABLES FOR MODELING

def generate_all_descriptors(data, fingerprint_check, functional_group_check):
    
    # calculate MORDRED descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mordred_columns = list(calc.descriptors)
    cols=[]
    for i in mordred_columns :
          cols.append(str(i))  
            
    print('here')
    data[cols] = data['SMILES'].parallel_apply(lambda x: calculate_mordred(x))
    
    # rdmol / rdkit descriptors
    for i in descriptorslist:
        data[i] = data['SMILES'].parallel_apply(lambda x: calc_descriptors_list(x,i))

    for i in rdmoldescriptors:
        data[i] = data['SMILES'].parallel_apply(lambda x: calc_rdmol_descriptors(x,i))
    print('done')

    # QRD variables
    data[['MW','ALOGP','HBA','HBD',
    'PSA','ROTB','AROM','KI','QRD_uw','QRD_w']] = data['SMILES'].apply(lambda x:scoring.calculate_QRD(x))


    data['LogD'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateLogD(x))
    data['LogP'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateLogP(x))
    data['Acid'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateAcid(x))
    data['pKa'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculatepKa(x))
    data['Fsp3'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateFsp3(x))
    data['HetCarbonRatio'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateHetCarbonRatio(x))
    data['pKa'] = data['pKa'].replace([np.inf, -np.inf], np.nan, inplace=True)
    data['pKa'] = data['pKa'].fillna(0)
    data['pKa'] = abs(data['pKa'])
    print('done')
    
    # Amine
    data[['V1','V2','angle_avg','desc_avg']] = data['SMILES'].apply(lambda x: Amine.amine_score(x))
    
    # Shape
    #data['Shapes'] =data['SMILES'].apply(lambda x: scoring.shape_score([x]))
    
    # pybiomed descriptors
    pybio_desc = pybiomed_get_keys()
    data[pybio_desc] = data['SMILES'].parallel_apply(lambda x: pybiomed_descriptors(x))
    

    # create list of descriptor names / column names
    extra = ['MW','ALOGP','HBA','HBD','PSA','ROTB','AROM','KI','QRD_uw','QRD_w','LogD',
            'LogP','Acid','pKa','Fsp3','HetCarbonRatio','V1','V2','angle_avg','desc_avg']
    descriptorlist_full = descriptorslist + rdmoldescriptors
    descriptorlist_full.extend(extra)
    descriptorlist_full.extend(cols)
    descriptorlist_full.extend(pybio_desc)
    
    if fingerprint_check == 'yes':
            # fingerprints
        data, fps_og = calc_fingerprints(data)
        fps = []
        for i in fps_og:
            fps.extend(i)
        descriptorlist_full.extend(fps)
        
        # functional groups
    if functional_group_check == 'yes':
        data, fg_cols = create_all_fragments(data)
        descriptorlist_full.extend(fg_cols)

        
    descriptorlist_full = list(set(descriptorlist_full))
    errors = data.columns[(data == 'error').any()]
    data = data.drop(columns=data.columns[(data == 'error').any()])

    for x in errors:
          descriptorlist_full.remove(x)
            
    data = data.loc[:,~data.columns.duplicated()]
    
    for i in exclude:
        descriptorlist_full.remove(i)
    
    data, descriptorlist_full = clean_descriptors(data, descriptorlist_full)
  
    return  data, descriptorlist_full


def clean_descriptors(smiles, descriptorlist_full):
    # Reduce intercorrelated descriptors
    def reduce_correlated_descriptors(data2):
            cor_matrix = data2.corr().abs()
            upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
            return to_drop

    to_drop = reduce_correlated_descriptors(smiles[descriptorlist_full])

    for i in to_drop:
        if i in descriptorlist_full:
            descriptorlist_full.remove(i)
        else:
            print(i)

    # convert columns to numeric or remove from descriptors
    for i in descriptorlist_full:
        try: 
            smiles[i] = smiles[i].apply(pd.to_numeric)
        except:
            descriptorlist_full.remove(i)

    smiles.replace([np.inf, -np.inf], 0, inplace=True)

    #only keep columns that have no nan or inf values
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    keep_list = clean_dataset(smiles[descriptorlist_full]).columns.tolist()
    descriptorlist_full = keep_list.copy()

    # get rid of columns with method errors
    smiles = smiles.set_index(['SMILES'])
    before = smiles.columns.tolist()
    smiles = smiles.select_dtypes(exclude=['object'])
    after = smiles.columns.tolist()

    for i in before:
        if i not in after:
            try:
                descriptorlist_full.remove(i)
            except :
                'nothing'
                
    smiles= smiles.reset_index()
    return smiles, descriptorlist_full


# # GENERATE LIST OF 3D DESCRIPTORS TO EXCLUDE USING CONFORMERS

#from smi2sdf3d import smi2sdf
#results = smi2sdf.main_return()
def identify_3d_Descriptors(results):
    check_all = []
    for i in results:
        temp = pd.DataFrame(i)
        temp.columns = ['SMILES']

        for i in descriptorslist:
            temp[i] = temp['SMILES'].apply(lambda x: calc_descriptors_list(x,i))
        for i in rdmoldescriptors:
            temp[i] = temp['SMILES'].apply(lambda x: calc_rdmol_descriptors(x,i))

        check = []
        for j in temp.columns.tolist():
            if temp[j].nunique() != 1:
                check.append(j)
                
        check_all.append(check)
        
    return check_all


exclude = ['CalcAUTOCORR3D',
 'CalcAsphericity',
 'CalcEccentricity',
 'CalcGETAWAY',
 'CalcInertialShapeFactor',
 'CalcMORSE',
 'CalcNPR1',
 'CalcNPR2',
 'CalcPBF',
 'CalcPMI1',
 'CalcPMI2',
 'CalcPMI3',
 'CalcRDF',
 'CalcRadiusOfGyration',
 'CalcSpherocityIndex',
 'CalcWHIM',
 'GetUSR',
 'GetUSRCAT']


# # GENERATE SPECIFIC SUBSET OF DESCRIPTORS - FOR MODELING OR OTHERWISE 

# +
def check_desc(x, desc):
        for i in x:
            if x not in desc:
                i.remove(x)
                
        return i 
    
def extend(a,b,c):
    total = []
    total.extend(a)
    total.extend(b)
    total.extend(c)
    return total 

def generate_descriptor_subset(data, desc):
        desc_copy = desc.copy()
        rdmol = []
        reg = []
        col_names = []
        mor_desc = []
        for i in desc:
                if i in rdmoldescriptors:
                    rdmol.append(i)

                if i in descriptorslist:
                    reg.append(i)

                if i in mordred_dict:
                    val = mordred_dict.get(i)
                    mor_desc.append(val)
                    col_names.append(i)

            # initialize a calulcator with subset of mordred descriptors
        calc2 = Calculator(mor_desc, ignore_3D=False)
        def calculate_mordred(x):
                mol = Chem.MolFromSmiles(x)
                return pd.Series(calc2(mol))

        data[col_names] = data['SMILES'].parallel_apply(lambda x: calculate_mordred(x))

        for i in reg:
                data[i] = data['SMILES'].parallel_apply(lambda x: calc_descriptors_list(x,i))

        for i in rdmol:
                data[i] = data['SMILES'].parallel_apply(lambda x: calc_rdmol_descriptors(x,i))

        data[['MW','ALOGP','HBA','HBD',
            'PSA','ROTB','AROM','KI','QRD_uw','QRD_w']] = data['SMILES'].parallel_apply(lambda x:scoring.calculate_QRD(x))
        data['LogD'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateLogD(x))
        data['LogP'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateLogP(x))
        data['Acid'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateAcid(x))
        data['pKa'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculatepKa(x))
        data['Fsp3'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateFsp3(x))
        data['HetCarbonRatio'] = data['SMILES'].parallel_apply(lambda x: scoring.CalculateHetCarbonRatio(x))
        data['pKa'] = data['pKa'].replace([np.inf, -np.inf], np.nan, inplace=True)
        data['pKa'] = data['pKa'].fillna(0)
        data['pKa'] = abs(data['pKa'])
        data[['V1','V2','angle_avg','desc_avg']] = data['SMILES'].apply(lambda x: Amine.amine_score(x))

        if 'Shapes' in desc:
                data['Shapes'] =data['SMILES'].apply(lambda x: scoring.shape_score([x]))

            # these functions were defined by hand so please get associated PyBioMed file located in this directory
            # and make install with other site packages
        pybio_desc = pybiomed_needed_keys(desc)
        data[pybio_desc] = data['data'].apply(lambda x: pybiomed_needed_descriptors(x))

        current_descriptors = data.columns.tolist()
        for i in desc:
                if i in current_descriptors:
                    desc.remove(i)
            
        return data, desc
            
            
def generate_descriptor_subset_FGS(data desc): 
        # the remaining descriptors are functional group descriptors
        # this is for a mixed type of functional groups, using multiple algorithims 
        # remove Tanimoto, Wildcard and Coulomb Matrix column headers
            
        for i in range(len(0,desc)):
                    try:
                        desc[i] = desc[i].replace('_TANIMOTO','')
                        desc[i] = desc[i].replace('_WILDCARD','')
                        desc[i] = desc[i].replace('_COULOMB','')
                    except:
                        'no fgs'

                # # check Algorithim # 3 - check functional group hierarchy first 
        fgs, remaining = tree_search_main.generate_needed_fg_hierarchy_descriptors(data, desc)
        data['FGS'] = fgs

        if remaining: 
                    # check Algorithim # 2
                    data['FGS2'] = data['data'].parallel_apply(lambda x: BRICS_implementation(x))
                    data['FGS2'] = data['FGS2'].parallel_apply(lambda x: check_desc(x, desc))

                    # check Algorithim # 1
                    data['FGS1'] = data['SMILES'].parallel_apply(lambda x: find_fg(x))
                    data['FGS1'] = data['FGS1'].parallel_apply(lambda x: check_desc(x, desc))

                    data['FGS'] = data.parallel_apply(lambda x: extend(x['FGS'], x['FGS1'], x['FGS2']))

                    data = data.assign(**pd.get_dummies(data['fgs'].apply(lambda x:pd.Series(x)).stack().reset_index(level=1,drop=True)).sum(level=0))


        # Create column of 0s for functional groups necessary for QSAR but not present in input dataset  
        data_cols = data.columns.tolist()
        for i in desc:
                if i in data_cols:
                    desc.remove(i)

        if desc:
            for i in desc:
                data[i] = 0

        return data
