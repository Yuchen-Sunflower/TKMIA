import numpy as np
import torch
from sklearn.metrics import ndcg_score
from sklearn.metrics import precision_score

def hamming_loss(y_GT, predict):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    temp = 0

    for i in range(GT_size.shape[0]):
        index=sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index]=1
        temp = temp + np.sum(y_GT[i] ^ predict_label[i])

    hmloss = temp/(y_GT.shape[0]*y_GT.shape[1])
    return hmloss

def FR(y_GT, predict):
    GT_size = np.sum(y_GT, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    temp = 0

    for i in range(GT_size.shape[0]):
        index = sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index] = 1
        if np.sum(y_GT[i] ^ predict_label[i])>0:
            temp = temp + 1

    fr = temp / y_GT.shape[0]
    return fr

def TP_index(y_targets, predict):
    GT_size = np.sum(y_targets, axis=1)
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    tp_flag = False

    for i in range(GT_size.shape[0]):
        index = sorted[i][-GT_size[i]:][::-1]
        predict_label[i][index] = 1
        if np.sum(y_targets[i] ^ predict_label[i])==0:
            tp_flag = True
    return tp_flag, predict_label


def nontargeted_TP_index(y_GT, predict, kvalue):
    predict_label = np.zeros(predict.shape, dtype=int)
    sorted = predict.argsort()
    flag = True

    for i in range(predict.shape[0]):
        index = sorted[i][-kvalue:][::-1]
        # index = torch.flip(index, [0])
        predict_label[i][index] = 1
        for j in index:
            if y_GT[i][j] == 1:
                flag = False
    return flag, predict_label
    
def specific_TP_index(y_GT, specific_GT, predict, GT_index_at_k, GT_sorted_at_k, GT_sort_origin_at_k, origin_GT_num):
    predict_label = np.zeros(predict.shape, dtype=int)
    # sorted = predict.argsort()
    flag = True
    dif = y_GT - specific_GT

    for i in range(predict.shape[0]):
        index = GT_index_at_k[i]
        # index = sorted[i][-kvalue:][::-1]
        if index.size < origin_GT_num or (GT_sorted_at_k < GT_sort_origin_at_k).any():  #not (GT_sorted_at_k == GT_sort_origin_at_k).all():
            flag = False
            break
        predict_label[i][index] = 1
        for j in index:
            if specific_GT[i][j] == 1 or dif[i][j] == 0:
                flag = False
                break
    return flag, predict_label
    
def delta_n(specific_GT, predict, predict_ori, k_value):
    index_topk = torch.argsort(predict, descending=True)[:, :k_value]
    index_topk_ori = torch.argsort(predict_ori, descending=True)[:, :k_value]

    GT_num_topk = torch.sum(specific_GT[torch.arange(specific_GT.shape[0]).unsqueeze(1), index_topk])
    GT_num_topk_ori = torch.sum(specific_GT[torch.arange(specific_GT.shape[0]).unsqueeze(1), index_topk_ori])

    return GT_num_topk_ori - GT_num_topk

def label_match(y_GT, predict,k_value):
    GT_size = np.sum(y_GT, axis=1)
    sorted = predict.argsort()

    for i in range(GT_size.shape[0]):
        index = sorted[i][-k_value:][::-1]
        for j in index:
            if y_GT[j]==1:
                return False
        return True
        
def predict_top_k_labels(predict_values, kvalue):
    labels = []
    for i in range(predict_values.shape[0]):
        a = predict_values[i].argsort()[-kvalue:][::-1]
        labels.append(np.asarray(a))
    return np.asarray(labels)

def topk_acc_metric(y_GT, predict, kvalue):
    count = 0
    GT_label_index_list = []
    for i in range(y_GT.shape[0]):
        GT_label_index_list.append(np.where(y_GT[i] == 1)[0])
    top_k_predict_labels_list = predict_top_k_labels(predict, kvalue)
    for i in range(top_k_predict_labels_list.shape[0]):
        if kvalue > GT_label_index_list[i].shape[0]:
            count = count + int(set(top_k_predict_labels_list[i]).issuperset(GT_label_index_list[i]))
        else:
            count = count + int(set(GT_label_index_list[i]).issuperset(top_k_predict_labels_list[i]))
    return count/top_k_predict_labels_list.shape[0]

def precision_at_k(y_GT, predict, k):
    y_GT = y_GT.clone().cpu().detach().numpy()
    predict = predict.clone().cpu().detach().numpy()
    top_k_idx = np.argpartition(predict, kth=-k)[:, -k:]
    top_k_targets = y_GT[np.arange(y_GT.shape[0])[:, np.newaxis], top_k_idx]
    top_k_scores = np.ones((y_GT.shape[0], k))
    return precision_score(top_k_targets, top_k_scores, average='micro')

def precision_at_k_instance(targets, scores, top_k_scores, k):
    top_k_idx = np.argpartition(scores, kth=-k)[-k:]
    top_k_targets = targets[top_k_idx]
    return precision_score(top_k_targets, top_k_scores[:k], average='micro')

def mAP_at_k(y_GT, predict, K):
    y_GT = y_GT.clone().cpu().detach().numpy()
    predict = predict.clone().cpu().detach().numpy()

    top_k_idx = np.argsort(predict)[:, ::-1][:, :K]
    top_k_scores = np.ones(K)
    top_k_targets = y_GT[np.arange(y_GT.shape[0])[:, np.newaxis], top_k_idx]
    p_at_k = np.array([[precision_at_k_instance(t, p, top_k_scores, k) for k in range(1, K + 1)] for t, p in zip(y_GT, predict)])

    n_k = np.sum(y_GT, axis=1)
    tmp_k = np.ones(y_GT.shape[0]) * K
    tag = n_k <= tmp_k
    n_k[~tag] = K
    return np.average(np.sum(p_at_k * top_k_targets, axis=1) / n_k)

def NDCG_at_k(y_GT, predict, k):
    y_GT = y_GT.clone().cpu().detach().numpy()
    predict = predict.clone().cpu().detach().numpy()
    return ndcg_score(y_true=y_GT, y_score=predict, k=k)
