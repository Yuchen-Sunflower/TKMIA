from tqdm import tqdm
import torch
import torch
import numpy as np
from evaluate_metrics import delta_n, topk_acc_metric
from evaluate_metrics import precision_at_k, mAP_at_k, NDCG_at_k
from attack import save_result
from numpy import random as nr

def jacobian(predictions, x, nb_classes):
    list_derivatives = []

    for class_ind in range(nb_classes):
        outputs = predictions[:, class_ind]
        derivatives, = torch.autograd.grad(outputs, x, grad_outputs=torch.ones_like(outputs), retain_graph=True)
        list_derivatives.append(derivatives)

    return list_derivatives

def kfool_global(model, inputs, GT, specific_label, k_value, boxmax, boxmin, maxiter, device, args, lr=1e-2):
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = GT
    shape = inputs.shape
    purtabed_img_out = np.zeros(shape)
    modifier_out = np.zeros(shape)
    del_n = torch.tensor([[0.0]])
    spec_n = torch.tensor([[0.0]])
    all_list = set(list(range(0, tlab.cpu().detach().numpy().shape[1])))

    with torch.no_grad():
        F = model(torch.tanh(timg) * boxmul + boxplus)

    purtabed_img = (torch.tanh(timg) * boxmul + boxplus).clone().requires_grad_()

    loop_i = 0
    attack_success = False
    flag = True
    Flag = False
    measures = [0] * 4
    F = model(purtabed_img)
    max_label = torch.argmax(tlab * F - (1 - tlab) * 10000)
    p = torch.argsort(F, dim=1, descending=True)
    tlab_all = ((tlab == 1).nonzero(as_tuple=True)[1]).cpu().detach().numpy()
    complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
    predict_label = GT

    r_tot = torch.zeros(timg.size()).to(device)

    # loss
    sorted = torch.argsort(F, descending=True)
    GT_sort = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted][:, :k_value]
    GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(GT_sort, as_tuple=True)[1].unsqueeze(0)]
    origin_GT_num = GT_index_at_k.shape[1]
    
    purtabed_out_ori = F
    mixed = tlab * specific_label
    sorted_ori = sorted
    mixed_sort = mixed[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori][:, :k_value]
    if torch.sum(tlab) < k_value + torch.sum(mixed) or torch.sum(mixed_sort) == 0:
        flag = False
        return purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n
    GT_sort_origin = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori]

    tkacc_ori = topk_acc_metric(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
    p_at_k_ori = precision_at_k(tlab, F, k_value)
    map_at_k_ori = mAP_at_k(tlab, F, k_value)
    ndcg_at_k_ori = NDCG_at_k(tlab, F, k_value)
    print('Original Top-{} Acc:{:.5f}'.format(k_value, tkacc_ori), \
            'Original P@{}: {:.4f}'.format(k_value, p_at_k_ori), \
            'Original mAP@{}: {:.4f}'.format(k_value, map_at_k_ori), \
            'Original NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k_ori))

    while (complement_set.issuperset(set(tlab_all))) == False and loop_i < maxiter:
        w = torch.squeeze(torch.zeros(timg.size()[1:])).to(device)
        f = torch.tensor([[0.0]]).to(device)
        top_F = F.topk(k_value + 1)[0]
        max_F = F[0][max_label].reshape((1,1))
        gradients_top = torch.stack(jacobian(top_F, purtabed_img, k_value + 1), dim=1)
        gradients_max = torch.stack(jacobian(max_F, purtabed_img, 1), dim=1)
        # gradients = torch.stack(jacobian(F, purtabed_img, nb_classes), dim=1)
        with torch.no_grad():
            for idx in range(inputs.size(0)):
                for k in range(k_value + 1):
                    if torch.all(torch.eq(gradients_top[idx, k, ...], gradients_max[idx,0,...]))==False and p[0][k]!=max_label:
                        norm = torch.div(1, torch.norm(gradients_top[idx, k, ...] - gradients_max[idx,0,...]))
                        w = w + (gradients_top[idx, k, ...] - gradients_max[idx,0,...]) * norm
                        f = f + (F[idx, p[0][k]] - F[idx, max_label]) * norm
                r_tot[idx, ...] = r_tot[idx, ...] + torch.abs(f) * w / torch.norm(w)
        purtabed_img = (torch.tanh(r_tot + timg) * boxmul + boxplus).requires_grad_()
        F = model(purtabed_img)
        p = torch.argsort(F, dim=1, descending=True)

        complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
        sorted = torch.argsort(F, descending=True)
        # if complement_set.issuperset(set(tlab_all)):
        # Flag, predict_label = specific_TP_index(tlab.cpu().detach().numpy(), mixed.cpu().detach().numpy(), \
        #                     F.cpu().detach().numpy(), GT_index_at_k.cpu().detach().numpy(), \
        #                     GT_sort.cpu().detach().numpy(), GT_sort_origin[:, :k_value].cpu().detach().numpy(), origin_GT_num)
        del_n = delta_n(mixed.cpu().detach(), F.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        if del_n >= torch.sum(mixed_sort):
            Flag = True
            break
        # if Flag:
        #     break
        loop_i = loop_i + 1

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    if flag:
        attack_success = Flag
        # print(F.cpu().detach().numpy())
        # print(tlab.cpu().detach().numpy())
        tkacc = topk_acc_metric(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
        p_at_k = precision_at_k(tlab, F, k_value)
        map_at_k = mAP_at_k(tlab, F, k_value)
        ndcg_at_k = NDCG_at_k(tlab, F, k_value)
        del_n = delta_n(mixed.cpu().detach(), F.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        spec_n = torch.sum(mixed_sort).cpu().detach().numpy()
        print('iter:', loop_i + 1, \
                'attacked: ', Flag, \
                'GT:', tlab.cpu().detach().numpy(), \
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


def kfool_random(model, inputs, GT, n_pert, k_value, boxmax, boxmin, maxiter, device, args, lr=1e-2):
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmax + boxmin) / 2.0

    timg = inputs
    tlab = GT
    shape = inputs.shape
    purtabed_img_out = np.zeros(shape)
    modifier_out = np.zeros(shape)
    del_n = torch.tensor([[0.0]])
    spec_n = torch.tensor([[0.0]])
    all_list = set(list(range(0, tlab.cpu().detach().numpy().shape[1])))

    with torch.no_grad():
        F = model(torch.tanh(timg) * boxmul + boxplus)

    purtabed_img = (torch.tanh(timg) * boxmul + boxplus).clone().requires_grad_()

    loop_i = 0
    attack_success = False
    flag = True
    Flag = False
    measures = [0] * 4
    F = model(purtabed_img)
    max_label = torch.argmax(tlab * F - (1 - tlab) * 10000)
    p = torch.argsort(F, dim=1, descending=True)
    tlab_all = ((tlab == 1).nonzero(as_tuple=True)[1]).cpu().detach().numpy()
    complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
    predict_label = GT

    r_tot = torch.zeros(timg.size()).to(device)

    # loss
    sorted = torch.argsort(F, descending=True)
    GT_sort = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted][:, :k_value]
    GT_index_at_k = sorted[torch.arange(tlab.shape[0]).unsqueeze(1), torch.nonzero(GT_sort, as_tuple=True)[1].unsqueeze(0)]
    
    purtabed_out_ori = F
    sorted_ori = sorted
    if n_pert <= 0 or n_pert > k_value or GT_index_at_k.shape[1] < n_pert:
        flag = False
        return purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n
    choose = nr.choice(GT_index_at_k.cpu().detach().numpy()[0, :], size=n_pert, replace=False)
    specific_label = torch.zeros_like(tlab).to(device)
    specific_label[:, choose] = 1
    mixed = tlab * specific_label

    if torch.sum(tlab) < k_value + torch.sum(mixed):
        flag = False
        return purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n
    GT_sort_origin = tlab[torch.arange(tlab.shape[0]).unsqueeze(1), sorted_ori]

    tkacc_ori = topk_acc_metric(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
    p_at_k_ori = precision_at_k(tlab, F, k_value)
    map_at_k_ori = mAP_at_k(tlab, F, k_value)
    ndcg_at_k_ori = NDCG_at_k(tlab, F, k_value)
    print('Original Top-{} Acc:{:.5f}'.format(k_value, tkacc_ori), \
            'Original P@{}: {:.4f}'.format(k_value, p_at_k_ori), \
            'Original mAP@{}: {:.4f}'.format(k_value, map_at_k_ori), \
            'Original NDCG@{}: {:.4f}'.format(k_value, ndcg_at_k_ori))

    while (complement_set.issuperset(set(tlab_all))) == False and loop_i < maxiter:
        w = torch.squeeze(torch.zeros(timg.size()[1:])).to(device)
        f = torch.tensor([[0.0]]).to(device)
        top_F = F.topk(k_value + 1)[0]
        max_F = F[0][max_label].reshape((1,1))
        gradients_top = torch.stack(jacobian(top_F, purtabed_img, k_value + 1), dim=1)
        gradients_max = torch.stack(jacobian(max_F, purtabed_img, 1), dim=1)
        # gradients = torch.stack(jacobian(F, purtabed_img, nb_classes), dim=1)
        with torch.no_grad():
            for idx in range(inputs.size(0)):
                for k in range(k_value + 1):
                    if torch.all(torch.eq(gradients_top[idx, k, ...], gradients_max[idx,0,...]))==False and p[0][k]!=max_label:
                        norm = torch.div(1, torch.norm(gradients_top[idx, k, ...] - gradients_max[idx,0,...]))
                        w = w + (gradients_top[idx, k, ...] - gradients_max[idx,0,...]) * norm
                        f = f + (F[idx, p[0][k]] - F[idx, max_label]) * norm
                r_tot[idx, ...] = r_tot[idx, ...] + torch.abs(f) * w / torch.norm(w)
        purtabed_img = (torch.tanh(r_tot + timg) * boxmul + boxplus).requires_grad_()
        F = model(purtabed_img)
        p = torch.argsort(F, dim=1, descending=True)

        complement_set = all_list - set((p[0][:k_value]).cpu().detach().numpy())
        sorted = torch.argsort(F, descending=True)
        # if complement_set.issuperset(set(tlab_all)):
        del_n = delta_n(mixed.cpu().detach(), F.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        if del_n >= args.del_n:
            Flag = True
            break
        loop_i = loop_i + 1

    purtabed_img_out = np.arctanh(((purtabed_img - boxplus) / boxmul * 0.999999).cpu().detach().numpy())
    modifier_out = purtabed_img_out - timg.cpu().detach().numpy()
    if flag:
        attack_success = Flag
        tkacc = topk_acc_metric(tlab.cpu().detach().numpy(), F.cpu().detach().numpy(), k_value)
        p_at_k = precision_at_k(tlab, F, k_value)
        map_at_k = mAP_at_k(tlab, F, k_value)
        ndcg_at_k = NDCG_at_k(tlab, F, k_value)
        del_n = delta_n(mixed.cpu().detach(), F.cpu().detach(), purtabed_out_ori.cpu().detach(), k_value)
        spec_n = torch.sum(mixed).cpu().detach().numpy()
        print('iter:', loop_i + 1, \
                'attacked: ', Flag, \
                'GT:', tlab.cpu().detach().numpy(), \
                'sorted GT prime:', GT_sort_origin.cpu().detach().numpy(), \
                'sorted GT index prime:', sorted_ori.cpu().detach().numpy(), \
                'perturb GT:', predict_label.cpu().detach().numpy(), \
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

def baselineap(args, model, device, val_loader):
    print('kvalue: ', args.k_value, 'app_type:', args.app, 'uap_norm:', args.uap_norm, 'uap_eps:', args.uap_eps)
    model.eval()
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

    if args.app == 'baseline_kfool_global':
        pos = list(map(int, args.specific_index.split(',')))
        specific_label = torch.zeros((args.batch_size, args.num_classes))
        specific_label[:, pos] = 1
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()
            if index < 1000:
                data, GT = data.to(device), GT.to(device)
                specific_label = specific_label.to(device)
                purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n =\
                    kfool_global(model, data, GT, specific_label, args.k_value, args.boxmax, args.boxmin, args.maxiter, device=device, args=args, lr=args.lr_attack)
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
        
    if args.app == 'baseline_kfool_random':
        pos = list(map(int, args.specific_index.split(',')))
        specific_label = torch.zeros((args.batch_size, args.num_classes))
        specific_label[:, pos] = 1
        for ith, (data, GT) in tqdm(enumerate(val_loader)):
            if len(GT.shape) == 3:
                GT = GT.max(dim=1)[0]
            else:
                pass
            GT = GT.int()
            if index < 1000:
                data, GT = data.to(device), GT.to(device)
                purtabed_img_out, modifier_out, sorted_ori, sorted, attack_success, flag, measures, del_n, spec_n =\
                    kfool_random(model, data, GT, args.n_pert, args.k_value, args.boxmax, args.boxmin, args.maxiter, device=device, args=args, lr=args.lr_attack)
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