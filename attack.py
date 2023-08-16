# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 10:12:49 2021
@author: Lipeng Ke
"""

from tqdm import tqdm
import torch
import gc
import os
import numpy as np
import torch
import numpy as np
from evaluate_metrics import delta_n, topk_acc_metric
from evaluate_metrics import precision_at_k, mAP_at_k, NDCG_at_k
from torch.autograd import Variable
from numpy import random as nr

def l2_topk_non_targeted_attack_global(model, inputs, label, specific_label, k_value, maxiter, boxmax, boxmin, device, args, lr=1e-2, weight_decay=1e-4):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = label
    shape = inputs.shape

    tkacc, tkacc_ori = 0, 0
    p_at_k, p_at_k_ori = 0, 0
    map_at_k, map_at_k_ori = 0, 0
    ndcg_at_k, ndcg_at_k_ori = 0, 0
    del_n = torch.tensor([[0.0]])
    spec_n = torch.tensor([[0.0]])

    modifier = Variable(torch.zeros(*shape).to(device), requires_grad=True)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr, weight_decay=weight_decay, momentum=0.9)

    purtabed_img = torch.zeros(*shape)
    attack_success = False
    flag = True
    Flag = False

    for iteration in range(maxiter):
        optimizer.zero_grad()
        purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        purtabed_out = model(purtabed_img)

        # loss
        sorted = torch.argsort(purtabed_out, descending=True)
        GT_sort = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted][:, :k_value]
        GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(GT_sort, as_tuple=True)[1].unsqueeze(0)]
        
        if iteration == 0:
            purtabed_out_ori = purtabed_out
            sorted_ori = sorted
            mixed = tlab * specific_label
            if torch.sum(tlab) < k_value + torch.sum(mixed):
                flag = False
                break
            sorted = torch.argsort(purtabed_out, dim=-1, descending=True)
            mixed_sort = mixed[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori][:, :k_value]
            if torch.sum(mixed_sort) == 0:
                flag = False
                break
            GT_sort_origin = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori]
            origin_GT_num = GT_index_at_k.shape[1]

            tkacc_ori = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
            p_at_k_ori = precision_at_k(tlab, purtabed_out, k_value)
            map_at_k_ori = mAP_at_k(tlab, purtabed_out, k_value)
            ndcg_at_k_ori = NDCG_at_k(tlab, purtabed_out, k_value)
            print('Original Top-{} Acc:{:.5f}'.format(k_value, tkacc_ori), \
                  'Original P@{}: {:.4f}'.format(k_value, p_at_k_ori), \
                  'Original mAP@{}: {:.4f}'.format(k_value, map_at_k_ori), \
                  'Original NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k_ori))
            
        real = torch.max(tlab * purtabed_out - ((1-tlab) * 10000))  # 求取一个batch最大的loss

        t_value = real - purtabed_out
        lambda_l, _ = torch.topk(t_value, label.shape[1] - k_value)
        loss = lambda_l[:, -1] + (1/(label.shape[1] - k_value)) \
            * torch.sum(torch.max(torch.zeros(t_value.shape).to(device), t_value - lambda_l[:, -1]))
        loss = torch.sum(loss)

        # Flag, predict_label = nontargeted_TP_index(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
        # Flag, predict_label = specific_TP_index(tlab.cpu().detach().numpy(), mixed.cpu().detach().numpy(), \
        #                         purtabed_out.cpu().detach().numpy(), GT_index_at_k.cpu().detach().numpy(), \
        #                         GT_sort.cpu().detach().numpy(), GT_sort_origin[:, :k_value].cpu().detach().numpy(), origin_GT_num)
        del_n = delta_n(mixed.cpu().detach(), purtabed_out.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        predict_label = tlab.cpu().detach().numpy()
        if del_n >= torch.sum(mixed_sort):
            Flag = True
            break
        # if Flag:
        #     break
        # Calculate gradient
        loss.backward()
        optimizer.step()

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    if flag:
        attack_success = Flag
        tkacc = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
        p_at_k = precision_at_k(tlab, purtabed_out, k_value)
        map_at_k = mAP_at_k(tlab, purtabed_out, k_value)
        ndcg_at_k = NDCG_at_k(tlab, purtabed_out, k_value)
        del_n = delta_n(mixed.cpu().detach(), purtabed_out.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        spec_n = torch.sum(mixed_sort).cpu().detach().numpy()
        print('iter:', iteration, 'loss= ', "{}".format(loss), \
                'attacked: ', Flag, \
                'GT:', label.cpu().detach().numpy(), \
                'sorted GT prime:', GT_sort_origin.cpu().detach().numpy(), \
                'sorted GT index prime:', sorted_ori.cpu().detach().numpy(), \
                'perturb GT:', predict_label, \
                'sorted GT:', GT_sort.cpu().detach().numpy(), \
                'sorted GT index:', sorted.cpu().detach().numpy(), \
                'Top-{} Acc:{:.5f}'.format(k_value, tkacc), \
                'P@{}: {:.4f}'.format(k_value, p_at_k), \
                'mAP@{}: {:.4f}'.format(k_value, map_at_k), \
                'NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k), \
                'Detla N: {}'.format(del_n), \
                'Spec N: {}'.format(spec_n))
        print('size:', "{:.5f}".format(np.linalg.norm(modifier_out)))

    measures = [tkacc - tkacc_ori, p_at_k - p_at_k_ori, map_at_k - map_at_k_ori, ndcg_at_k - ndcg_at_k_ori]
    return purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n


def l2_topk_non_targeted_attack_random(model, inputs, label, n_pert, k_value, maxiter, boxmax, boxmin, device, args, lr=1e-2, weight_decay=1e-4):
    # trick from CW, normalize to [boxin, boxmin]
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = label
    shape = inputs.shape

    modifier = Variable(torch.zeros(*shape).to(device), requires_grad=True)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr, weight_decay=weight_decay, momentum=0.9)

    purtabed_img = torch.zeros(*shape)
    attack_success = False
    flag = True
    Flag = False

    tkacc, tkacc_ori = 0, 0
    p_at_k, p_at_k_ori = 0, 0
    map_at_k, map_at_k_ori = 0, 0
    ndcg_at_k, ndcg_at_k_ori = 0, 0
    del_n = torch.tensor([[0.0]])
    spec_n = torch.tensor([[0.0]])

    for iteration in range(maxiter):
        optimizer.zero_grad()
        purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        purtabed_out = model(purtabed_img)

        # loss
        sorted = torch.argsort(purtabed_out, descending=True)
        GT_sort = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted][:, :k_value]
        GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(GT_sort, as_tuple=True)[1].unsqueeze(0)]
        
        if iteration == 0:
            purtabed_out_ori = purtabed_out
            sorted_ori = sorted
            if n_pert <= 0 or n_pert > k_value or GT_index_at_k.shape[1] < n_pert:
                flag = False
                break
            choose = nr.choice(GT_index_at_k.cpu().detach().numpy()[0, :], size=n_pert, replace=False)
            specific_label = torch.zeros_like(tlab).to(device)
            specific_label[:, choose] = 1
            mixed = tlab * specific_label

            if torch.sum(tlab) < k_value + torch.sum(mixed):
                flag = False
                break
            GT_sort_origin = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori]

            tkacc_ori = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
            p_at_k_ori = precision_at_k(tlab, purtabed_out, k_value)
            map_at_k_ori = mAP_at_k(tlab, purtabed_out, k_value)
            ndcg_at_k_ori = NDCG_at_k(tlab, purtabed_out, k_value)
            print('Original Top-{} Acc:{:.5f}'.format(k_value, tkacc_ori), \
                  'Original P@{}: {:.4f}'.format(k_value, p_at_k_ori), \
                  'Original mAP@{}: {:.4f}'.format(k_value, map_at_k_ori), \
                  'Original NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k_ori))
            
        real = torch.max(tlab * purtabed_out - ((1-tlab) * 10000))  # 求取一个batch最大的loss

        t_value = real - purtabed_out
        lambda_l, _ = torch.topk(t_value, label.shape[1] - k_value)
        loss = lambda_l[:, -1] + (1/(label.shape[1] - k_value)) \
            * torch.sum(torch.max(torch.zeros(t_value.shape).to(device), t_value - lambda_l[:, -1]))
        loss = torch.sum(loss)

        # Flag, predict_label = specific_TP_index(tlab.cpu().detach().numpy(), mixed.cpu().detach().numpy(), \
        #                         purtabed_out.cpu().detach().numpy(), GT_index_at_k.cpu().detach().numpy(), \
        #                         GT_sort.cpu().detach().numpy(), GT_sort_origin[:, :k_value].cpu().detach().numpy(), origin_GT_num)
        del_n = delta_n(mixed.cpu().detach(), purtabed_out.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        if del_n >= args.del_n:
            Flag = True
            break

        # Calculate gradient
        loss.backward()
        optimizer.step()

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    if flag:
        attack_success = Flag
        tkacc = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
        p_at_k = precision_at_k(tlab, purtabed_out, k_value)
        map_at_k = mAP_at_k(tlab, purtabed_out, k_value)
        ndcg_at_k = NDCG_at_k(tlab, purtabed_out, k_value)
        spec_n = torch.sum(mixed).cpu().detach().numpy()
        print('iter:', iteration, 'loss= ', "{}".format(loss), \
                'attacked: ', Flag, \
                'GT:', label.cpu().detach().numpy(), \
                'sorted GT prime:', GT_sort_origin.cpu().detach().numpy(), \
                'sorted GT index prime:', sorted_ori.cpu().detach().numpy(), \
                'sorted GT:', GT_sort.cpu().detach().numpy(), \
                'sorted GT index:', sorted.cpu().detach().numpy(), \
                'Top-{} Acc:{:.5f}'.format(k_value, tkacc), \
                'P@{}: {:.4f}'.format(k_value, p_at_k), \
                'mAP@{}: {:.4f}'.format(k_value, map_at_k), \
                'NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k), \
                'Detla N: {}'.format(del_n), \
                'Spec N: {}'.format(spec_n))
        print('size:', "{:.5f}".format(np.linalg.norm(modifier_out)))

    measures = [tkacc - tkacc_ori, p_at_k - p_at_k_ori, map_at_k - map_at_k_ori, ndcg_at_k - ndcg_at_k_ori]
    return purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n


def l2_global_specific_targeted_attack(model, inputs, label, specific_label, k_value, maxiter, boxmax, boxmin, device, args, lr=1e-2, weight_decay=1e-4):
    # trick from CW, normalize to [boxin, boxmin]
    assert k_value % 1 == 0
    assert label.shape[0] == 1

    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = label
    shape = inputs.shape

    modifier = Variable(torch.zeros(*shape).to(device), requires_grad=True)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr, weight_decay=weight_decay, momentum=0.9)

    purtabed_img = torch.zeros(*shape)
    attack_success = False
    flag = True
    Flag = False

    tkacc, tkacc_ori = 0, 0
    p_at_k, p_at_k_ori = 0, 0
    map_at_k, map_at_k_ori = 0, 0
    ndcg_at_k, ndcg_at_k_ori = 0, 0
    del_n = torch.tensor([[0.0]])
    spec_n = torch.tensor([[0.0]])
    
    for iteration in range(maxiter):
        optimizer.zero_grad()
        purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        purtabed_out = model(purtabed_img)

        # loss
        sorted = torch.argsort(purtabed_out, descending=True)
        GT_sort = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted][:, :k_value]
        GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(GT_sort, as_tuple=True)[1].unsqueeze(0)]
        # non_GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(1 - GT_sort, as_tuple=True)[1].unsqueeze(0)]
        
        if iteration == 0:
            purtabed_out_ori = purtabed_out
            mixed = tlab * specific_label
            sorted_ori = sorted
            mixed_sort = mixed[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori][:, :k_value]
            if torch.sum(tlab) < k_value + torch.sum(mixed) or torch.sum(mixed_sort) == 0:
                flag = False
                break
            GT_sort_origin = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori]
            origin_GT_num = GT_index_at_k.shape[1]
            # origin_GT_pos = GT_index_at_k
            tkacc_ori = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
            p_at_k_ori = precision_at_k(tlab, purtabed_out, k_value)
            map_at_k_ori = mAP_at_k(tlab, purtabed_out, k_value)
            ndcg_at_k_ori = NDCG_at_k(tlab, purtabed_out, k_value)
            print('Original Top-{} Acc:{:.5f}'.format(k_value, tkacc_ori), \
                  'Original P@{}: {:.4f}'.format(k_value, p_at_k_ori), \
                  'Original mAP@{}: {:.4f}'.format(k_value, map_at_k_ori), \
                  'Original NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k_ori))

        real_1 = torch.max(mixed * purtabed_out - (1 - mixed) * 10000)               # \max_{s \in S} f_s
        real_2 = torch.min((tlab - mixed) * purtabed_out + (1 - (tlab - mixed)) * 10000)      # \min_{y \in Y\S} f_y

        # extract three main terms
        t_value_1 = real_1 - purtabed_out
        t_value_2 = purtabed_out - real_2

        lambda_l_1, _ = torch.topk(t_value_1, label.shape[1] - k_value)  # obtain the top-coef values of each row n*1
        lambda_l_2, _ = torch.topk(t_value_2, k_value)

        loss = lambda_l_1[:, -1] + lambda_l_2[:, -1] \
            + (1 / (label.shape[1] - k_value)) * torch.sum(torch.max(torch.zeros(t_value_1.shape).to(device), t_value_1 - lambda_l_1[:, -1])) \
                + (1 / k_value) * torch.sum(torch.max(torch.zeros(t_value_2.shape).to(device), t_value_2 - lambda_l_2[:, -1]))  # Eq.8 objective
        loss = torch.sum(loss)

        # Flag, predict_label = specific_TP_index(tlab.cpu().detach().numpy(), mixed.cpu().detach().numpy(), \
        #                         purtabed_out.cpu().detach().numpy(), GT_index_at_k.cpu().detach().numpy(), \
        #                         GT_sort.cpu().detach().numpy(), GT_sort_origin[:, :k_value].cpu().detach().numpy(), origin_GT_num)
        del_n = delta_n(mixed.cpu().detach(), purtabed_out.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        predict_label = tlab.cpu().detach().numpy()
        if del_n >= torch.sum(mixed_sort):
            Flag = True
            break

        # If attack success terminate and return
        # if Flag:
        #     break

        # Calculate gradient
        loss.backward()
        optimizer.step()

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    if flag:
        attack_success = Flag
        tkacc = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
        p_at_k = precision_at_k(tlab, purtabed_out, k_value)
        map_at_k = mAP_at_k(tlab, purtabed_out, k_value)
        ndcg_at_k = NDCG_at_k(tlab, purtabed_out, k_value)
        del_n = delta_n(mixed.cpu().detach(), purtabed_out.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        spec_n = torch.sum(mixed_sort).cpu().detach().numpy()
        print('iter:', iteration, 'loss= ', "{}".format(loss), \
                'attacked: ', Flag, \
                'GT:', label.cpu().detach().numpy(), \
                'sorted GT prime:', GT_sort_origin.cpu().detach().numpy(), \
                'sorted GT index prime:', sorted_ori.cpu().detach().numpy(), \
                'perturb GT:', predict_label, \
                'sorted GT:', GT_sort.cpu().detach().numpy(), \
                'sorted GT index:', sorted.cpu().detach().numpy(), \
                'Top-{} Acc:{:.5f}'.format(k_value, tkacc), \
                'P@{}: {:.4f}'.format(k_value, p_at_k), \
                'mAP@{}: {:.4f}'.format(k_value, map_at_k), \
                'NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k), \
                'Detla N: {}'.format(del_n), \
                'Spec N: {}'.format(spec_n))
        print('size:', "{:.5f}".format(np.linalg.norm(modifier_out)))

    measures = [tkacc - tkacc_ori, p_at_k - p_at_k_ori, map_at_k - map_at_k_ori, ndcg_at_k - ndcg_at_k_ori]
    return purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n


def l2_random_specific_targeted_attack(model, inputs, label, n_pert, k_value, maxiter, boxmax, boxmin, device, args, lr=1e-2, weight_decay=1e-4):
    # trick from CW, normalize to [boxin, boxmin]
    assert k_value % 1 == 0
    assert label.shape[0] == 1

    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = label
    shape = inputs.shape

    tkacc, tkacc_ori = 0, 0
    p_at_k, p_at_k_ori = 0, 0
    map_at_k, map_at_k_ori = 0, 0
    ndcg_at_k, ndcg_at_k_ori = 0, 0
    del_n = torch.tensor([[0.0]])
    spec_n = torch.tensor([[0.0]])

    modifier = Variable(torch.zeros(*shape).to(device), requires_grad=True)
    model.eval()

    optimizer = torch.optim.SGD([{'params': modifier}], lr=lr, weight_decay=weight_decay, momentum=0.9)

    purtabed_img = torch.zeros(*shape)
    attack_success = False
    flag = True
    Flag = False
    
    for iteration in range(maxiter):
        optimizer.zero_grad()
        purtabed_img = torch.tanh(modifier + timg) * boxmul + boxplus
        purtabed_out = model(purtabed_img)

        # loss
        sorted = torch.argsort(purtabed_out, descending=True)
        GT_sort = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted][:, :k_value]
        GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(GT_sort, as_tuple=True)[1].unsqueeze(0)]
        non_GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(1 - GT_sort, as_tuple=True)[1].unsqueeze(0)]
        
        if iteration == 0:
            purtabed_out_ori = purtabed_out
            sorted_ori = sorted
            if n_pert <= 0 or n_pert > k_value or GT_index_at_k.shape[1] < n_pert:
                flag = False
                break
            choose = nr.choice(GT_index_at_k.cpu().detach().numpy()[0, :], size=n_pert, replace=False)
            specific_label = torch.zeros_like(tlab).to(device)
            specific_label[:, choose] = 1
            mixed = tlab * specific_label

            if torch.sum(tlab) < k_value + torch.sum(mixed):
                flag = False
                break
            GT_sort_origin = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori]
            origin_GT_num = GT_index_at_k.shape[1]
            origin_GT_pos = GT_index_at_k
            tkacc_ori = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
            p_at_k_ori = precision_at_k(tlab, purtabed_out, k_value)
            map_at_k_ori = mAP_at_k(tlab, purtabed_out, k_value)
            ndcg_at_k_ori = NDCG_at_k(tlab, purtabed_out, k_value)
            print('Original Top-{} Acc:{:.5f}'.format(k_value, tkacc_ori), \
                  'Original P@{}: {:.4f}'.format(k_value, p_at_k_ori), \
                  'Original mAP@{}: {:.4f}'.format(k_value, map_at_k_ori), \
                  'Original NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k_ori))

        real_1 = torch.max(mixed * purtabed_out - (1 - mixed) * 10000)               # \max_{s \in S} f_s
        real_2 = torch.min((tlab - mixed) * purtabed_out + (1 - (tlab - mixed)) * 10000)      # \min_{y \in Y\S} f_y

        # extract three main terms
        t_value_1 = real_1 - purtabed_out
        t_value_2 = purtabed_out - real_2

        lambda_l_1, _ = torch.topk(t_value_1, label.shape[1] - k_value)  # obtain the top-coef values of each row n*1
        lambda_l_2, _ = torch.topk(t_value_2, k_value)

        loss = lambda_l_1[:, -1] + lambda_l_2[:, -1] \
            + (1 / (label.shape[1] - k_value)) * torch.sum(torch.max(torch.zeros(t_value_1.shape).to(device), t_value_1 - lambda_l_1[:, -1])) \
                + (1 / k_value) * torch.sum(torch.max(torch.zeros(t_value_2.shape).to(device), t_value_2 - lambda_l_2[:, -1]))  # Eq.8 objective
        loss = torch.sum(loss)

        # Flag, predict_label = specific_TP_index(tlab.cpu().detach().numpy(), mixed.cpu().detach().numpy(), \
        #                         purtabed_out.cpu().detach().numpy(), GT_index_at_k.cpu().detach().numpy(), \
        #                         GT_sort.cpu().detach().numpy(), GT_sort_origin[:, :k_value].cpu().detach().numpy(), origin_GT_num)
        del_n = delta_n(mixed.cpu().detach(), purtabed_out.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        predict_label = tlab.cpu().detach().numpy()
        if del_n >= args.del_n:
            Flag = True
            break

        # If attack success terminate and return
        # if Flag:
        #     break

        # Calculate gradient
        loss.backward()
        optimizer.step()

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    if flag:
        attack_success = Flag
        tkacc = topk_acc_metric(tlab.cpu().detach().numpy(), purtabed_out.cpu().detach().numpy(), k_value)
        p_at_k = precision_at_k(tlab, purtabed_out, k_value)
        map_at_k = mAP_at_k(tlab, purtabed_out, k_value)
        ndcg_at_k = NDCG_at_k(tlab, purtabed_out, k_value)
        del_n = delta_n(mixed.cpu().detach(), purtabed_out.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        spec_n = torch.sum(mixed).cpu().detach().numpy()
        print('iter:', iteration, 'loss= ', "{}".format(loss), \
                'attacked: ', Flag, \
                'GT:', label.cpu().detach().numpy(), \
                'sorted GT prime:', GT_sort_origin.cpu().detach().numpy(), \
                'sorted GT index prime:', sorted_ori.cpu().detach().numpy(), \
                'perturb GT:', predict_label, \
                'sorted GT:', GT_sort.cpu().detach().numpy(), \
                'sorted GT index:', sorted.cpu().detach().numpy(), \
                'Top-{} Acc:{:.5f}'.format(k_value, tkacc), \
                'P@{}: {:.4f}'.format(k_value, p_at_k), \
                'mAP@{}: {:.4f}'.format(k_value, map_at_k), \
                'NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k), \
                'Detla N: {}'.format(del_n), \
                'Spec N: {}'.format(spec_n))
        print('size:', "{:.5f}".format(np.linalg.norm(modifier_out)))

    measures = [tkacc - tkacc_ori, p_at_k - p_at_k_ori, map_at_k - map_at_k_ori, ndcg_at_k - ndcg_at_k_ori]
    return purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n


def tkmlap(args, model, device, val_loader):
    """
    Evaluate a deep neural network model

    Args:
        model: pytorch model object
        device: cuda or cpu
        val_dataloader: validation images dataloader

    """
    print('kvalue: ', args.k_value, 'app_type:', args.app, 'specific_index:', args.specific_index)
    success_count = 0
    index = 0
    img = []
    label = []
    img_out = []
    perturb_out = []
    perturb_norm = []
    sorted_set_ori = []
    sorted_set_pert = []
    overall_tkacc = []
    overall_p = []
    overall_map = []
    overall_ndcg = []
    overall_del_n = []
    overall_spec_n = []
    model.eval()

    pos = list(map(int, args.specific_index.split(',')))
    specific_label = torch.zeros((args.batch_size, args.num_classes))
    specific_label[:, pos] = 1

    if args.app == 'global_specific_attack':
        # specific target attack
        size = torch.sum(specific_label)
        for ith, (data, GT) in enumerate(val_loader):
            # if torch.sum(GT) >= args.k_value + size:
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()

            if index < 1000:
                # print('\n')
                data, GT = data.to(device), GT.to(device)
                specific_label = specific_label.to(device)
                purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n = \
                    l2_global_specific_targeted_attack(model, data, GT, specific_label, args.k_value, args.maxiter, args.boxmax, args.boxmin, \
                                            device, args, lr=args.lr_attack, weight_decay=args.weight_decay)
                if flag:
                    if attack_success:
                        print('pixel perturb: {:.5f}'.format(np.linalg.norm(modifier_out) / args.image_size / args.image_size))
                        img.append(data.cpu().numpy())
                        label.append(GT.cpu().numpy())
                        img_out.append(purtabed_img_out)
                        perturb_out.append(modifier_out)
                        perturb_norm.append(np.linalg.norm(modifier_out))
                        sorted_set_ori.append(sorted_ori.cpu().detach().numpy())
                        sorted_set_pert.append(sorted.cpu().detach().numpy())
                        success_count = success_count + 1

                    overall_tkacc.append(measures[0])
                    overall_p.append(measures[1])
                    overall_map.append(measures[2])
                    overall_ndcg.append(measures[3])
                    overall_del_n.append(del_n)
                    overall_spec_n.append(spec_n)
                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == 1000:
                break

        save_result(args, success_count, index, img, label, img_out, perturb_out, perturb_norm, sorted_set_ori, sorted_set_pert, \
                overall_tkacc, overall_p, overall_map, overall_ndcg, overall_del_n, overall_spec_n)

    if args.app == 'random_specific_attack':
        # specific target attack
        for ith, (data, GT) in enumerate(val_loader):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()

            if index < 1000:
                # print('\n')
                data, GT = data.to(device), GT.to(device)
                purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n = \
                    l2_random_specific_targeted_attack(model, data, GT, args.n_pert, args.k_value, args.maxiter, args.boxmax, args.boxmin, \
                                            device, args, lr=args.lr_attack, weight_decay=args.weight_decay)
                if flag:
                    if attack_success:
                        print('pixel perturb: {:.5f}'.format(np.linalg.norm(modifier_out) / args.image_size / args.image_size))
                        img.append(data.cpu().numpy())
                        label.append(GT.cpu().numpy())
                        img_out.append(purtabed_img_out)
                        perturb_out.append(modifier_out)
                        perturb_norm.append(np.linalg.norm(modifier_out))
                        sorted_set_ori.append(sorted_ori.cpu().detach().numpy())
                        sorted_set_pert.append(sorted.cpu().detach().numpy())
                        success_count = success_count + 1

                    overall_tkacc.append(measures[0])
                    overall_p.append(measures[1])
                    overall_map.append(measures[2])
                    overall_ndcg.append(measures[3])
                    overall_del_n.append(del_n)
                    overall_spec_n.append(spec_n)
                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == 1000:
                break

        save_result(args, success_count, index, img, label, img_out, perturb_out, perturb_norm, sorted_set_ori, sorted_set_pert, \
                overall_tkacc, overall_p, overall_map, overall_ndcg, overall_del_n, overall_spec_n)

    if args.app == 'none_target_attack_global':
        # none target attack
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            # if ith in sample_list[:1000]:
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()

            if index < 1000:
                data, GT = data.to(device), GT.to(device)
                specific_label = specific_label.to(device)
                purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n = \
                    l2_topk_non_targeted_attack_global(model, data, GT, specific_label, args.k_value, args.maxiter, args.boxmax, args.boxmin, \
                                            device, args, lr=args.lr_attack, weight_decay=args.weight_decay)

                if flag:
                    if attack_success:
                        print('pixel perturb: {:.5f}'.format(np.linalg.norm(modifier_out) / args.image_size / args.image_size))
                        img.append(data.cpu().numpy())
                        label.append(GT.cpu().numpy())
                        img_out.append(purtabed_img_out)
                        perturb_out.append(modifier_out)
                        perturb_norm.append(np.linalg.norm(modifier_out))
                        sorted_set_ori.append(sorted_ori.cpu().detach().numpy())
                        sorted_set_pert.append(sorted.cpu().detach().numpy())
                        success_count = success_count + 1

                    overall_tkacc.append(measures[0])
                    overall_p.append(measures[1])
                    overall_map.append(measures[2])
                    overall_ndcg.append(measures[3])
                    overall_del_n.append(del_n)
                    overall_spec_n.append(spec_n)
                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == 1000:
                break

        save_result(args, success_count, index, img, label, img_out, perturb_out, perturb_norm, sorted_set_ori, sorted_set_pert, \
                overall_tkacc, overall_p, overall_map, overall_ndcg, overall_del_n, overall_spec_n)

    if args.app == 'none_target_attack_random':
        # none target attack
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            # if ith in sample_list[:1000]:
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()

            if index < 1000:
                data, GT = data.to(device), GT.to(device)
                specific_label = specific_label.to(device)
                purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n = \
                    l2_topk_non_targeted_attack_random(model, data, GT, args.n_pert, args.k_value, args.maxiter, args.boxmax, args.boxmin, \
                                            device, args, lr=args.lr_attack, weight_decay=args.weight_decay)

                if flag:
                    if attack_success:
                        print('pixel perturb: {:.5f}'.format(np.linalg.norm(modifier_out) / args.image_size / args.image_size))
                        img.append(data.cpu().numpy())
                        label.append(GT.cpu().numpy())
                        img_out.append(purtabed_img_out)
                        perturb_out.append(modifier_out)
                        perturb_norm.append(np.linalg.norm(modifier_out))
                        sorted_set_ori.append(sorted_ori.cpu().detach().numpy())
                        sorted_set_pert.append(sorted.cpu().detach().numpy())
                        success_count = success_count + 1

                    overall_tkacc.append(measures[0])
                    overall_p.append(measures[1])
                    overall_map.append(measures[2])
                    overall_ndcg.append(measures[3])
                    overall_del_n.append(del_n)
                    overall_spec_n.append(spec_n)
                    print('success:{}/{}'.format(success_count, index + 1))
                    index = index + 1
            if index == 1000:
                break

        save_result(args, success_count, index, img, label, img_out, perturb_out, perturb_norm, sorted_set_ori, sorted_set_pert, \
                overall_tkacc, overall_p, overall_map, overall_ndcg, overall_del_n, overall_spec_n)

    torch.cuda.empty_cache()


def save_result(args, success_count, index, img, label, img_out, perturb_out, perturb_norm, sorted_set_ori, sorted_set_pert, \
                overall_tkacc, overall_p, overall_map, overall_ndcg, overall_del_n, overall_spec_n):
    print("IASR: {}".format(success_count / index))
    if 'random' in args.app:
        path = f'/data1/sunyuchen/result/{args.dataset}/{args.app}/{args.k_value}_{args.lr_attack}_{args.arch}_{args.n_pert}_{args.maxiter}_{args.del_n}/'
    else:
        path = f'/data1/sunyuchen/result/{args.dataset}/{args.app}/{args.k_value}_{args.lr_attack}_{args.arch}_{args.specific_index}_{args.maxiter}/'
    path_txt = path + 'log.txt'
    path_img = path + 'perturb_img'
    path_label = path + 'label'
    path_perturb_img = path + 'img'
    path_perturb = path + 'perturb'
    path_sort_ori = path + 'sort_ori'
    path_sort_pert = path + 'sort_pert'

    path_tkacc = path + 'tkacc'
    path_p = path + 'p'
    path_map = path + 'map'
    path_ndcg = path + 'ndcg'
    if not os.path.exists(path):
        os.makedirs(path)
    img = np.array(img)
    img_out = np.array(img_out)
    perturb_out = np.array(perturb_out)
    perturb_norm = np.array(perturb_norm)
    sorted_set_ori = np.array(sorted_set_ori)
    sorted_set_pert = np.array(sorted_set_pert)
    average_tkacc = np.mean(np.array(overall_tkacc))
    average_p = np.mean(np.array(overall_p))
    average_map = np.mean(np.array(overall_map))
    average_ndcg = np.mean(np.array(overall_ndcg))
    average_del_n = np.mean(np.array(overall_del_n))
    average_spec_n = np.mean(np.array(overall_spec_n))
    print('Average Top-k Acc: {}'.format(average_tkacc))
    print('Average P@k: {}'.format(average_p))
    print('Average mAP@k: {}'.format(average_map))
    print('Average NDCG@k: {}'.format(average_ndcg))
    print('Average Delta N: {}'.format(average_del_n))
    print('Average Specific N: {}'.format(average_spec_n))
    print('Overall Perturbation: {}'.format(np.mean(perturb_norm)))
    np.save(path_img + '.npy', img)
    np.save(path_label + '.npy', label)
    np.save(path_perturb_img + '.npy', img_out)
    np.save(path_perturb + '.npy', perturb_out)
    np.save(path_sort_ori + '.npy', sorted_set_ori)
    np.save(path_sort_pert + '.npy', sorted_set_pert)

    np.save(path_tkacc + '.npy', overall_tkacc)
    np.save(path_p + '.npy', overall_p)
    np.save(path_map + '.npy', overall_map)
    np.save(path_ndcg + '.npy', overall_ndcg)
    with open(path_txt, "w") as f:
        f.write(str(f"Specific Index: {args.specific_index}\n"))
        f.write(str(f"IASR = {success_count / index}, Overall Perturbation = {np.mean(perturb_norm)}, number = {index}\n"))
        f.write(str(f"Average Top-k Acc = {average_tkacc}, Average P@k = {average_p}, \
                    Average mAP@k = {average_map}, Average NDCG@k = {average_ndcg}, \
                    Average Delta N = {average_del_n}, Average Specific N = {average_spec_n}\n"))
        f.close()