import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors
from io import StringIO
import re
import numpy as np
from numpy.linalg import norm
from numpy import arccos
from numpy import pi
import statistics
from itertools import combinations
from pandarallel import pandarallel

cpus = 20


def mol_with_atom_index(mol):
    atoms = mol.GetNumAtoms()
    for idx in range(atoms):
        mol.GetAtomWithIdx(idx).SetProp('molAtomMapNumber', str(mol.GetAtomWithIdx(idx).GetIdx()))
    return mol

def generate_conformers(mol, num):
    m = Chem.AddHs(mol)
    ids=AllChem.EmbedMultipleConfs(m, numConfs=num)
    for id in ids:
        AllChem.UFFOptimizeMolecule(m, confId=id)
    confIds = [x.GetId() for x in m.GetConformers()]
    confs_mol = []
    confs_xyz = []
    for id in confIds:
        confs_mol.append(Chem.MolToMolBlock(m, confId=id))
        confs_xyz.append(Chem.MolToXYZBlock(m, confId=id))
    return confs_mol, confs_xyz


def identify_ring_amine(conformer):
    mol = Chem.MolFromMolBlock(conformer)
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

    if a == 1:
        amine_patt = NH1_patt
    elif b == 1:
        amine_patt = NH0_patt
    elif f == 1:
        amine_patt = NH2_patt
    else:
        amine_patt = NH2_patt2

    amine_atom = mol.GetSubstructMatch(amine_patt)
    aromatic_6ring = Chem.MolFromSmarts('a1aaaaa1')
    aromatic_5ring = Chem.MolFromSmarts('a1aaaa1')
    six_rings = mol.GetSubstructMatches(aromatic_6ring)
    five_rings = mol.GetSubstructMatches(aromatic_5ring)

    ring_atoms = six_rings + five_rings

    return amine_atom, ring_atoms

def calculate_centroid(ring_atoms, xyz_df):
    ring_mean = []
    for ring in ring_atoms:
        x = []
        y = []
        z = []
        for ring_atom in ring:
            ring_point_xyz1 = xyz_df.loc[ring_atom]
            ring_pointx = ring_point_xyz1[1]
            ring_pointy = ring_point_xyz1[2]
            ring_pointz = ring_point_xyz1[3]
            x.append(ring_pointx)
            y.append(ring_pointy)
            z.append(ring_pointz)
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        z_mean = statistics.mean(z)
        ring_mean.append([x_mean,y_mean,z_mean])
    return ring_mean


def calc_max_descriptor(amine, point1, point2):
    # calculate distance to first point
    squared_dist1 = np.sum((amine - point1) ** 2, axis=0)
    dist1 = np.sqrt(squared_dist1)

    # calculate distance to second point
    squared_dist2 = np.sum((amine - point2) ** 2, axis=0)
    dist2 = np.sqrt(squared_dist2)

    f = point1 - amine
    e = point2 - amine
    abVec = norm(f[0:3])
    bcVec = norm(e[0:3])
    abNorm = f / abVec
    bcNorm = e / bcVec
    res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] + abNorm[2] * bcNorm[2]
    angle = arccos(res) * 180.0 / pi

    ####
    ####
    ####
    ####
    ####CHANGE THIS MATH TO CHANGE WHAT IS IMPORTANT!!!!!
    total_desc = dist1 + dist2 + angle / 10
    #####CHANGE HERE!!!!!
    ####
    ####
    ####

    return dist1, dist2, angle, total_desc


def calc_value_for_conf(conf_xyz, conf_mol):
    # FIND THE AMINE ATOM AND RING ATOMS FROM MOL FILE
    amine_atom_list, ring_atoms = identify_ring_amine(conf_mol)
    amine_atom = amine_atom_list[0]

    # CREATE XYZ DF WITH ALL COORDINATES
    xyz_file = conf_xyz
    xyz_file = re.sub("\s\s+", " ", xyz_file)
    xyz_df = pd.read_csv(StringIO(xyz_file), sep=' ', skiprows=2, names=['element', 'X', 'Y', 'Z'])

    # GET ARRAY OF RING CENTROIDS FROM RING ATOMS AND XYZ DATAFRAME
    ring_centroid = calculate_centroid(ring_atoms, xyz_df)
    ring_num = len(ring_centroid)

    # FIND ALL POSSIBLE COMBINATIONS OF AMINE VECTORS
    possible_combinations = list(combinations(ring_centroid, 2))

    # CALCULATE ALL POSSIBLE LINEAR COMBINATIONS AND FIND MAX VALUE CONFORMER
    amine_vectors = []
    max_list = []
    amine_list = xyz_df.loc[amine_atom]
    amine_coordinates = amine_list[1:4]

    for combination in possible_combinations:
        dist_angle = []
        dist_angle = calc_max_descriptor(amine_coordinates, combination[0], combination[1])
        amine_vectors.append(dist_angle)
        max_list.append(dist_angle[3])

    # THIS IS THE DESCRIPTOR FOR THE CONFORMER
    max_value_conf = amine_vectors[max_list.index(max(max_list))]
    return max_value_conf


def amine_position_description(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        confs_mol, confs_xyz = generate_conformers(mol, 10)
        num_of_confs = len(confs_mol)
        conf_amine = []
        total_desc = []
        dists1 = []
        dists2 = []
        angles = []
        for conf in range(0, num_of_confs):

            v1, v2, angle1, combined1 = calc_value_for_conf(confs_xyz[conf], confs_mol[conf])
            x = []
            if v1 > v2:
                x = v1, v2, angle1, combined1
            else:
                x = v2, v1, angle1, combined1

            conf_amine.append(x)
            total_desc.append(x[3])
            dists1.append(x[0])
            dists2.append(x[1])
            angles.append(x[2])

        dists1_avg = np.average(dists1)
        dists2_avg = np.average(dists2)
        angles_avg = np.average(angles)
        # max_value_conf = conf_amine[distance.index(max(distance))]
        # a, b, c, d = max_value_conf

        a = dists1_avg
        b = dists2_avg
        c = angles_avg
        d = dists1_avg + dists2_avg + angles_avg / 10

    except:
        a = ''
        b = ''
        c = ''
        d = ''

    return pd.Series([a, b, c, d])

def amine_score(x):
    return amine_position_description(x)





