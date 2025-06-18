import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy.spatial.distance import pdist, squareform
import pandas as pd
from torch.utils.data import Dataset

def sim_graph(omics_data, protein_number):
    sim_matrix = np.zeros((protein_number, protein_number), dtype=float)
    adj_matrix = np.zeros((protein_number, protein_number), dtype=float)

    for i in range(protein_number):
        for j in range(i + 1):
            sim_matrix[i, j] = np.dot(omics_data[i], omics_data[j]) / (
                        np.linalg.norm(omics_data[i]) * np.linalg.norm(omics_data[j]))
            sim_matrix[j, i] = sim_matrix[i, j]

    for i in range(protein_number):
        topindex = np.argsort(sim_matrix[i])[-10:]
        for j in topindex:
            adj_matrix[i, j] = 1
    return adj_matrix

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}
CHARISOSMILEN = 64
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}
CHARPROTLEN = 25
def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X
class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

def load_data(DATASET):
    w2v = np.genfromtxt("protein_Word2Vec.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    protein = w2v[:, 0]
    protein_number = len(protein)
    w2v = np.array(w2v)
    w2v = scale(np.array(w2v[:, 2:], dtype=float))
    pca = PCA(n_components=128)
    w2v = pca.fit_transform(w2v)

    DC = np.genfromtxt("protein_DC.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    protein = DC[:, 0]
    protein_number = len(protein)
    DC = np.array(DC)
    DC = scale(np.array(DC[:, 2:], dtype=float))
    pca = PCA(n_components=128)
    DC = pca.fit_transform(DC)

    fusion_protein_fea = np.concatenate((DC,w2v), axis=1)
    fusion_protein_adj = sim_graph(fusion_protein_fea,protein_number).astype(int)

    pc = pd.read_csv('drug_phy_chem.csv', header=0)
    drug = pc.iloc[:, 0]
    drug_number = len(drug)
    pc_data = np.array(pc.iloc[:, 1:], dtype=float)
    pc_data = np.nan_to_num(pc_data, nan=0.0)
    pc_data = scale(pc_data)
    pca = PCA(n_components=128)
    pc_data = pca.fit_transform(pc_data)

    MACCS = np.genfromtxt("drug_ecfp_features.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    drug = MACCS[:, 0]
    drug_number = len(drug)
    MACCS = np.array(MACCS)
    MACCS = scale(np.array(MACCS[:, 1:], dtype=float))
    pca = PCA(n_components=128)
    MACCS = pca.fit_transform(MACCS)

    fusion_drug_fea = np.concatenate((pc_data, MACCS), axis=1)
    fusion_drug_adj = sim_graph(fusion_drug_fea, drug_number).astype(int)

    protein_feat, protein_adj = torch.FloatTensor(fusion_protein_fea), torch.FloatTensor(fusion_protein_adj)
    drug_feat, drug_adj = torch.FloatTensor(fusion_drug_fea), torch.FloatTensor(fusion_drug_adj)

    labellist = []
    with open('mapped_data.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()
        elements = line.split(" ")
        processed_elements = [int(elements[0]), int(elements[1]), int(elements[2])]
        labellist.append(processed_elements)
    labellist = torch.Tensor(labellist).long()

    drug_input = ('mapped_drug_smiles.txt')
    with open(drug_input, "r") as f:
        drug_data_list = f.read().strip().split('\n')
    drug_dataset = CustomDataSet(drug_data_list)
    drug_N = len(drug_dataset)
    drug_ids = []
    compound_max = 100
    compound_new = torch.zeros((drug_N, compound_max), dtype=torch.long)
    for i, pair in enumerate(drug_dataset):
        pair = pair.strip().split()
        drug_index, compoundstr = pair[-2], pair[-1]
        drug_ids.append(drug_index)
        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint

    protein_input = ('mapped_protein_sequences.txt')
    with open(protein_input, "r") as f:
        protien_data_list = f.read().strip().split('\n')
    protein_dataset = CustomDataSet(protien_data_list)

    protein_N = len(protein_dataset)
    protein_ids = []
    protein_max = 1000
    protein_new = torch.zeros((protein_N, protein_max), dtype=torch.long)
    for i, pair in enumerate(protein_dataset):
        pair = pair.strip().split()
        protein_index, proteinstr = pair[-2], pair[-1]
        protein_ids.append(protein_index)
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint


    return protein_feat, protein_adj, drug_feat, drug_adj, labellist, compound_new, protein_new
