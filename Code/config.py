import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from utils import *
from torch.utils.data import Dataset
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features, concat=True):
        super(GraphConvolutionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)
        h_prime = torch.mm(adj, Wh)
        return F.relu(h_prime)

class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-4
        self.Epoch = 30
        self.Batch_size = 128
        self.Patience = 8
        self.weight_decay = 1e-4
        self.embed_dim = 256
        self.drug_Length = 100
        self.protein_Length = 1000
        self.conv = 40

def shuffle_dataset(dataset):

    shuffled_indices = torch.randperm(dataset.size(0))
    shuffled_dataset = dataset[shuffled_indices]
    return shuffled_dataset

class selfattention(nn.Module):
    def __init__(self, sample_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.query = nn.Linear(sample_size, d_k)
        self.key = nn.Linear(sample_size, d_k)
        self.value = nn.Linear(sample_size, d_v)
    def forward(self, x):
        x =x.T
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.d_k)
        att = torch.softmax(att, dim=1)
        output = torch.matmul(att, v)
        return output.T

class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE
    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)

def get_kfold_data(i, datasets, k=5):

    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        validset = datasets[val_start:val_end]
        trainset = np.concatenate((datasets[0:val_start], datasets[val_end:]), axis=0)
    elif i == 0:
        val_end = fold_size
        validset = datasets[val_start:val_end]
        trainset = datasets[val_end:]
    else:
        validset = datasets[val_start:]
        trainset = datasets[0:val_start]

    return trainset, validset

class EarlyStopping:
    def __init__(self, savepath=None, patience=8, verbose=False, delta=0, num_n_fold=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = -np.inf
        self.early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath

    def __call__(self, score, model, num_epoch):

        if self.best_score == -np.inf:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(score, model, num_epoch)
            self.best_score = score
            self.counter = 0

    def save_checkpoint(self, score, model, num_epoch):
        if self.verbose:
            print(
                f'Have a new best checkpoint: ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.savepath +
                   '/valid_best_checkpoint.pth')
def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
    Accuracy_mean = np.mean(Accuracy_List)
    Precision_mean = np.mean(Precision_List)
    Recall_mean = np.mean(Recall_List)
    AUC_mean = np.mean(AUC_List)
    AUPR_mean = np.mean(AUPR_List)

    print("The model's results:")
    filepath = "./{}/results.txt".format(DATASET)
    with open(filepath, 'w') as f:
        f.write('Accuracy:{:.4f}'.format(
            Accuracy_mean) + '\n')
        f.write('Precision:{:.4f}'.format(
            Precision_mean, ) + '\n')
        f.write('Recall:{:.4f}'.format(
            Recall_mean) + '\n')
        f.write('AUC:{:.4f}'.format(AUC_mean) + '\n')
        f.write('AUPR:{:.4f}'.format(AUPR_mean) + '\n')
    print('Accuracy:{:.4f}'.format(Accuracy_mean))
    print('Precision:{:.4f}'.format(Precision_mean))
    print('Recall:{:.4f}'.format(Recall_mean))
    print('AUC:{:.4f}'.format(AUC_mean))
    print('AUPR:{:.4f}'.format(AUPR_mean))

