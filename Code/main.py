import os
import time
import argparse
import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from config import *
from utils import load_data
from model import DCNNMS
import torch.utils.data as Dataset
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

parser = argparse.ArgumentParser(prog='DCNNMS-DTI',)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--hidden', type=int, default=16, help='Number of hidden units.')
parser.add_argument('--heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--K_Fold', type=int, default=5, help='Number of the K_Fold')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    torch.cuda.manual_seed(args.seed)
    used_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()
    print(f"GPU success，服务器GPU已分配：{used_memory / 1024 ** 3:.2f} GB，已缓存：{cached_memory / 1024 ** 3:.2f} GB")
else:
    DEVICE = torch.device("cpu")

hp = hyperparameter()

DATASET = "Human"
print("Train in " + DATASET)

protein_features, protein_adj, drug_features, drug_adj, sample_set, compound_new, protein_new = load_data(DATASET)
protein_features, protein_adj = protein_features.to(DEVICE), protein_adj.to(DEVICE)
drug_features, drug_adj = drug_features.to(DEVICE), drug_adj.to(DEVICE)
compound_new, protein_new = compound_new.to(DEVICE), protein_new.to(DEVICE)

Accuracy_List, AUC_List, AUPR_List, Recall_List, Precision_List= [], [], [], [], []

used_memory = torch.cuda.memory_allocated()
cached_memory = torch.cuda.memory_reserved()
print(f"数据上传成功，服务器GPU已分配：{used_memory / 1024**3:.2f} GB，已缓存：{cached_memory / 1024**3:.2f} GB")

'''shuffle data'''
sample_set = shuffle_dataset(sample_set)

# 划分数据集
train_set, test_set = train_test_split(np.arange(sample_set.shape[0]), test_size=0.2,
                                       random_state=np.random.randint(0, 1000))
index_test, y_test = sample_set[test_set[:], :2], sample_set[test_set[:], 2]
time_begin = time.time()
print("------------开始训练------------")

def train(epoch, index_tra, y_tra, index_val, y_val):
    time_begin = time.time()
    tra_dataset = Dataset.TensorDataset(index_tra, y_tra)
    train_dataset = Dataset.DataLoader(tra_dataset, batch_size=hp.Batch_size, shuffle=True,drop_last=True,num_workers=0)
    train_losses_in_epoch = []
    model.train()
    for index_trian, y_train in train_dataset:
        y_train = y_train.to(DEVICE)
        y_tpred = model(compound_new, protein_new, index_trian.numpy().astype(int), DEVICE,
                        drug_features, drug_adj, protein_features, protein_adj)
        y_train = y_train.to(torch.int64)
        loss_train = loss_func(y_tpred, y_train)
        train_losses_in_epoch.append(loss_train.item())
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        scheduler.step()

    train_loss_a_epoch = np.average(train_losses_in_epoch)
    valid_losses_in_epoch = []
    model.eval()
    Y, P, S = [], [], []
    val_dataset = Dataset.TensorDataset(index_val, y_val)
    valid_dataset = Dataset.DataLoader(val_dataset, batch_size=hp.Batch_size, shuffle=False,drop_last=True,num_workers=0)
    for index_valid, y_valid in valid_dataset:
        y_valid = y_valid.to(DEVICE)
        y_vpred = model(compound_new, protein_new,index_valid.numpy().astype(int), DEVICE,
                         drug_features, drug_adj, protein_features, protein_adj)
        y_valid = y_valid.to(torch.int64)
        loss_valid = loss_func(y_vpred, y_valid)
        valid_losses_in_epoch.append(loss_valid.item())

        valid_labels = y_valid.to('cpu').data.numpy()
        y_vpred = F.softmax(y_vpred, 1).to('cpu').data.numpy()

        valid_predictions = np.argmax(y_vpred, axis=1)
        valid_scores = y_vpred[:, 1]

        Y.extend(valid_labels)
        P.extend(valid_predictions)
        S.extend(valid_scores)
    time_over = time.time()

    valid_loss_a_epoch = np.average(valid_losses_in_epoch)
    Precision_dev = precision_score(Y, P)
    Reacll_dev = recall_score(Y, P)
    Accuracy_dev = accuracy_score(Y, P)
    AUC_valid = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    AUPR_valid = auc(fpr, tpr)

    print_msg = (f'Epoch: {epoch+1:04d}  ' +
                 f'train_loss: {train_loss_a_epoch:.4f} ' +
                 f'valid_loss: {valid_loss_a_epoch:.4f} ' +
                 f'valid_Accuracy: {Accuracy_dev:.4f} ' +
                 f'valid_Precision: {Precision_dev:.4f} ' +
                 f'valid_Reacll: {Reacll_dev:.4f} ' +
                 f'valid_AUC: {AUC_valid:.4f} ' +
                 f'valid_AUPR: {AUPR_valid:.4f} ')
    print(print_msg)
    early_stopping(AUC_valid, model, epoch)

def test(index_test, y_test, dataset_class):

    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    dataset = Dataset.TensorDataset(index_test, y_test)
    test_dataset = Dataset.DataLoader(dataset, batch_size=hp.Batch_size, shuffle=False, drop_last=True,num_workers=0)
    for index_test, y_test in test_dataset:

        y_test = y_test.to(DEVICE)
        predicted_scores = model(compound_new, protein_new,index_test.numpy().astype(int), DEVICE,
                                 drug_features, drug_adj, protein_features, protein_adj)
        y_test = y_test.to(torch.int64)
        loss = loss_func(predicted_scores, y_test)
        correct_labels = y_test.to('cpu').data.numpy()
        predicted_scores = F.softmax(predicted_scores, 1).to('cpu').data.numpy()
        predicted_labels = np.argmax(predicted_scores, axis=1)
        predicted_scores = predicted_scores[:, 1]

        Y.extend(correct_labels)
        P.extend(predicted_labels)
        S.extend(predicted_scores)
        test_losses.append(loss.item())

    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    AUPR = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)

    save = True
    if save:
        filepath = save_path + "/{}_{}_prediction.txt".format(DATASET, dataset_class)
        with open(filepath, 'a') as f:
            for i in range(len(Y)):
                f.write(str(Y[i]) + " " + str(P[i]) + '\n')
    results = '{}: Loss:{:.4f};Accuracy:{:.4f};Precision:{:.4f};Recall:{:.4f};AUC:{:.4f};AUPR:{:.4f}.'\
        .format(dataset_class, test_loss, Accuracy, Precision, Recall, AUC, AUPR)
    print(results)
    return results, Accuracy, Precision, Recall, AUC, AUPR


for i_fold in range(args.K_Fold):
    print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)
    train_dataset, valid_dataset = get_kfold_data(i_fold, train_set, k=args.K_Fold)

    index_train, y_train = sample_set[train_dataset[:], :2], sample_set[train_dataset[:], 2]
    train_size = len(index_train)
    index_valid, y_valid = sample_set[valid_dataset[:], :2], sample_set[valid_dataset[:], 2]
    model =DCNNMS(hp, nprotein=protein_features.shape[0], ndrug=drug_features.shape[0],
                    nproteinfeat=protein_features.shape[1], ndrugfeat=drug_features.shape[1],
                    nhid=args.hidden, nheads=args.nb_heads, alpha=args.alpha).to(DEVICE)

    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}],
        lr=hp.Learning_rate)
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
                                            cycle_momentum=False, step_size_up=train_size // hp.Batch_size)

    loss_func = CELoss(DEVICE=DEVICE)
    used_memory = torch.cuda.memory_allocated()
    cached_memory = torch.cuda.memory_reserved()
    print(f"loss函数上传成功，服务器GPU已分配：{used_memory / 1024**3:.2f} GB，已缓存：{cached_memory / 1024**3:.2f} GB")

    save_path = "./" + DATASET + "/{}".format(i_fold + 1)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'
    early_stopping = EarlyStopping(
        savepath=save_path, patience=hp.Patience, verbose=True, delta=0)
    print("------------设置早停机制------------")

    for epoch in range(hp.Epoch):
        if early_stopping.early_stop == True:
            break
        train(epoch, index_train, y_train, index_valid, y_valid)

    model.load_state_dict(torch.load(early_stopping.savepath + '/valid_best_checkpoint.pth'))
    trainset_test_results, _, _, _, _, _, _ = test(index_train, y_train, dataset_class="Train")
    validset_test_results, _, _, _, _, _, _ = test(index_valid, y_valid, dataset_class="Valid")
    testset_test_results, Accuracy_test, Precision_test, Recall_test, AUC_test, AUPR_test = test(index_test, y_test,
                                                                dataset_class="Test")
    AUC_List.append(AUC_test)
    Accuracy_List.append(Accuracy_test)
    AUPR_List.append(AUPR_test)
    Recall_List.append(Recall_test)
    Precision_List.append(Precision_test)
    with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
        f.write("Test the model" + '\n')
        f.write(trainset_test_results + '\n')
        f.write(validset_test_results + '\n')
        f.write(testset_test_results + '\n')

show_result(DATASET, Accuracy_List, Precision_List,Recall_List, AUC_List, AUPR_List)

time_total = time.time() - time_begin
print("Total train-time: {:.4f}s".format(time_total))







