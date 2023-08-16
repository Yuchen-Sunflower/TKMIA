
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 10:50:25 2019
@author: Keshik
"""
import argparse
import torch
import numpy as np
import torchvision.models as models
import torch.optim as optim
from train import train_model, test
from attack import tkmlap
from baseline_attacks import baselineap
from utils import encode_labels, plot_history
import os
import utils
import random
from models.inception import Inception3
from datetime import datetime
from dataset import create_dataset

os.environ['TORCH_HOME'] = '.'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True

def main(args):
    """
    Main function

    Args:
        data_dir: directory to download Pascal VOC data
        model_name: resnet18, resnet34 or resnet50
        num: model_num for file management purposes (can be any postive integer. Your results stored will have this number as suffix)
        lr: initial learning rate list [lr for resnet_backbone, lr for resnet_fc]
        epochs: number of training epochs
        batch_size: batch size. Default=16
        download_data: Boolean. If true will download the entire 2012 pascal VOC data as tar to the specified data_dir.
        Set this to True only the first time you run it, and then set to False. Default False
        save_results: Store results (boolean). Default False

    Returns:
        test-time loss and average precision

    Example way of running this function:
        if __name__ == '__main__':
            main('../data/', "resnet34", num=1, lr = [1.5e-4, 5e-2], epochs = 15, batch_size=16, download_data=False, save_results=True)
    """

    data_dir = args.data + args.dataset
    model_name = args.arch
    num = args.num
    lr = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    download_data = args.download_data
    save_results = args.save_results

    model_add = {
        'resnet18': '/data1/sunyuchen/pretrained/resnet18-5c106cde.pth',
        'resnet34': '/data1/sunyuchen/pretrained/resnet34-333f7ec4.pth',
        'resnet50': '/data1/sunyuchen/pretrained/resnet50-19c8e357.pth',
        'resnet101': '/data1/sunyuchen/pretrained/resnet101-63fe2227.pth',
        'inception_v3': '/data1/sunyuchen/pretrained/inception_v3_google-1a9a5a14.pth'
    }

    model_collections_dict = {
            "resnet18": models.resnet18(),
            "resnet34": models.resnet34(),
            "resnet50": models.resnet50(),
            "resnet101": models.resnet101(),
            "inception_v3": models.inception_v3()
            }

    # Initialize cuda parameters
    setup_seed(2023)
    # device = torch.device("cuda" if use_cuda else "cpu")
    device_ids = list(map(int, args.device_id.split(',')))
    device = torch.device("cuda:{}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Available device = ", device)

    if model_name in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'inception_v3']:
        model = model_collections_dict[model_name]
        model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        state = torch.load(model_add[model_name], map_location='cpu')
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, args.num_classes)
    else:
        model = Inception3(num_classes=args.num_classes)

    model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    if args.opt == 'SGD':
        optimizer = optim.SGD([
            {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
            {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
        ])
    elif args.opt == 'Adam':
        optimizer = optim.Adam([
            {'params': list(model.parameters())[:-1], 'lr': lr[0], 'momentum': 0.9},
            {'params': list(model.parameters())[-1], 'lr': lr[1], 'momentum': 0.9}
        ])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 12, eta_min=0, last_epoch=-1)

    train_loader, valid_loader = create_dataset(args)



    #---------------Test your model here---------------------------------------
    # Load the best weights before testing

    if args.app == 'train':
        time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_dir = f"./{args.results}/{args.arch}_{args.image_size}_{args.batch_size}_{args.lr[0]}_{args.opt}_{args.normalize}_{time_stamp}/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log_file = open(os.path.join(model_dir, "log-{}.txt".format(num)), "w+")
        log_file.write("----------Experiment {} - {}-----------\n".format(num, model_name))
        # log_file.write("transformations == {}\n".format(transformations.__str__()))
        trn_hist, val_hist = train_model(model, device, optimizer, scheduler, train_loader, valid_loader, model_dir, num, epochs, log_file, args)
        torch.cuda.empty_cache()

        plot_history(trn_hist[0], val_hist[0], "Loss", os.path.join(model_dir, "loss-{}".format(num)))
        plot_history(trn_hist[1], val_hist[1], "Accuracy", os.path.join(model_dir, "accuracy-{}".format(num)))
        log_file.close()

    elif 'attack' in args.app:
        weights_file_path = f"./{args.results}/{args.model_path}/model-{num}.pth"
        print(weights_file_path)
        if os.path.isfile(weights_file_path):
            print("Loading best weights")
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weights_file_path).items()})
            tkmlap(args, model, device, valid_loader)
        else:
            print("Failed to load model")
            return 

    elif 'baseline' in args.app:
        weights_file_path = f"./{args.results}/{args.model_path}/model-{num}.pth"
        print(weights_file_path)
        if os.path.isfile(weights_file_path):
            print("Loading best weights")
            model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(weights_file_path).items()})
            baselineap(args, model, device, valid_loader)
        else:
            print("Failed to load model")
            return 



# Execute main function here.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TKMLAP')
    parser.add_argument('--data', default='/data1/zitai/dataset/', help='path to dataset') ## ./data or ./data/COCO_2014
    parser.add_argument('--dataset', default='VOC2012', type=str, choices={'VOC2007', 'VOC2012', 'COCO2014', 'NUSWIDE'}, help='path to dataset')
    parser.add_argument('--results', default='models_VOC2012', help='path to dataset')
    parser.add_argument('--model_path', default='resnet101_448_32_0.0001_SGD_2023-03-13_10-42-56', help='path to pre-trained model')
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes') ## 20 or 80 or 81
    parser.add_argument('--arch',  default='resnet101', help='model architecture: '+' (default: inception_v3, resnet50)')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N')
    parser.add_argument('--image_size', default=300, type=int, metavar='N', help='input image size (default: 300, 224)')
    parser.add_argument('--lr', '--learning-rate', default=[1e-4, 1e-3], type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='M', help='weight_decay')
    parser.add_argument('--device-id', type=str, default='0', help='the index of used gpu')
    parser.add_argument('--normalize', default='boxmaxmin', type=str, choices={'mean_std', 'boxmaxmin'}, help='optimizer for training')
    parser.add_argument('--opt', default='Adam', type=str, choices={'Adam', 'SGD'}, help='optimizer for training')
    parser.add_argument('--num', default=1, type=int, help='num to resume')
    parser.add_argument('--download_data', default=False, type=bool, help='download data')
    parser.add_argument('--save_results', default=True, type=bool, help='save results')
    parser.add_argument('--k_value', default=3, type=int, help='k-value')
    parser.add_argument('--eps', default=10, type=int, help='eps')
    parser.add_argument('--maxiter', default=1000, type=int, help='max iteration to attack')
    parser.add_argument('--boxmax', default=1, type=float, help='max value of input')
    parser.add_argument('--boxmin', default=-1, type=float, help='min value of input')
    parser.add_argument('--lr_attack', default=1e-2, type=float, help='learning rate of attacks')
    parser.add_argument('--del_n', default=1, type=int, help='the number of perturbed labels')
    parser.add_argument('--specific_index', default='0', type=str, help='the index of specific labels')
    parser.add_argument('--n_pert', default='1', type=int, help='the number of specific labels for global untargeted')
    parser.add_argument('--app', default='none_target_attack_global', type=str, \
                        choices={'none_target_attack_global', 'none_target_attack_random', 'random_specific_attack', 'global_specific_attack', \
                                 'baseline_kfool_global', 'baseline_kfool_random', 'train'}, help='attack types')

    args = parser.parse_args()
    main(args)
