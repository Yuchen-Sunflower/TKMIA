
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:50:25 2019
@author: Keshik
"""
import argparse
import numpy as np
import os
from matplotlib import pyplot as plt


def plot(args):

    mean = 0.5
    std = 0.5

    if 'random' in args.app:
        file_path = f'/data1/sunyuchen/result/{args.dataset}/{args.app}/{args.k_value}_{args.lr_attack}_{args.arch}_{args.n_pert}_{args.maxiter}_{args.del_n}/'
        save_path = f'/data1/sunyuchen/plot_fig/{args.dataset}/{args.app}/k={args.k_value},pert={args.n_pert},iter={args.maxiter},del_n={args.del_n}/'
    else:
        file_path = f'/data1/sunyuchen/result/{args.dataset}/{args.app}/{args.k_value}_{args.lr_attack}_{args.arch}_{args.specific_index}_{args.maxiter}/'
        save_path = f'/data1/sunyuchen/plot_fig/{args.dataset}/{args.app}/k={args.k_value},sl={args.specific_index},iter={args.maxiter}/'
    print(file_path)
    if not os.path.exists(file_path):
        print('not existing files')
        return 

    img = np.load(file_path + "img.npy")
    label = np.load(file_path + "label.npy")
    img_pert = np.load(file_path + "perturb_img.npy")
    pert = np.load(file_path + "perturb.npy")
    sort_ori = np.load(file_path + "sort_ori.npy")
    sort_pert = np.load(file_path + "sort_pert.npy")

    tkacc = np.load(file_path + "tkacc.npy")
    p = np.load(file_path + "p.npy")
    map = np.load(file_path + "map.npy")
    ndcg = np.load(file_path + "ndcg.npy")

    index = 0

    path_img = save_path + 'img/'
    path_txt = path_img + 'log.txt'
    if not os.path.exists(path_img):
        os.makedirs(path_img)
    with open(path_txt, "w") as f:
        for ith in range(img.shape[0]):
            fig = plt.figure(constrained_layout=True)
            plt.imshow(img[ith][0].transpose((1, 2, 0)) * std + mean)
            plt.axis('off')
            plt.tight_layout()
            # plt.show()
            fig.savefig(path_img + f'{ith}.jpg', bbox_inches='tight')
            plt.close()
            # print('GT:{}'.format(np.arange(args.num_classes)[label[ith][0] == 1]))
            f.write(str(f"Img{ith}: {np.arange(args.num_classes)[label[ith][0] == 1]}\n"))
            f.write(str(f"Perturbation: {np.linalg.norm(pert[ith][0])}\n"))
            f.write(str(f"Original sort: {sort_ori[ith][0]}\n"))
            f.write(str(f"Perturbed sort: {sort_pert[ith][0]}\n"))
            f.write(str(f"TkAcc: {tkacc[ith]}, p@k: {p[ith]}, mAP@k: {map[ith]}, NDCG@k: {ndcg[ith]}\n"))
            # print(sort_ori[ith][0])
            index += 1
            if index == 100:
                break
        f.close()
    
    index = 0
    path_pert = save_path + 'pert/'
    if not os.path.exists(path_pert):
        os.makedirs(path_pert)
    for ith in range(pert.shape[0]):
        fig = plt.figure(constrained_layout=True)
        plt.imshow(pert[ith][0].transpose((1, 2, 0)) * std * 100)
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        fig.savefig(path_pert + f'{ith}.jpg', bbox_inches='tight')
        plt.close()
        index += 1
        if index == 100:
            break

    index = 0
    path_pert_img = save_path + 'pert_img/'
    if not os.path.exists(path_pert_img):
        os.makedirs(path_pert_img)
    for ith in range(img_pert.shape[0]):
        fig = plt.figure(constrained_layout=True)
        plt.imshow(img_pert[ith][0].transpose((1, 2, 0)) * std + mean)
        plt.axis('off')
        plt.tight_layout()
        # plt.show()
        fig.savefig(path_pert_img + f'{ith}.jpg', bbox_inches='tight')
        plt.close()
        index += 1
        if index == 100:
            break


# Execute main function here.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TKIA')
    parser.add_argument('--dataset', default='VOC2012', type=str, choices={'VOC2012', 'COCO2014', 'NUSWIDE'}, help='path to dataset')
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
    parser.add_argument('--arch',  default='resnet50', help='model architecture:' + '(default: resnet18)')
    parser.add_argument('--k_value', default=10, type=int, help='k-value')
    parser.add_argument('--lr_attack', default=1e-2, type=float, help='learning rate of attacks')
    parser.add_argument('--maxiter', default=300, type=int, help='max iteration to attack')
    parser.add_argument('--del_n', default=1, type=int, help='the number of perturbed labels')
    parser.add_argument('--specific_index', default='0', type=str, help='the index of specific labels')
    parser.add_argument('--n_pert', default='1', type=int, help='the number of specific labels for global untargeted')
    parser.add_argument('--app', default='none_target_attack', type=str,
                        choices={'none_target_attack_global', 'none_target_attack_random', 'random_specific_attack', 'global_specific_attack', \
                                 'baseline_rank', 'baseline_kfool_global', 'baseline_kfool_random', 'test', 'train'}, help='attack types')
    args = parser.parse_args()
    plot(args)
