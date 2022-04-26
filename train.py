#!/usr/bin/python

import os
import sys
import argparse
import random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# custom functions defined by user
from model import DeepCNN
from datasets import EPIDataSet
from trainer import Trainer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample


def randomparams():
    space = {
        'lr': hp.choice('lr', (0.001, 0.0001)),
        'beta1': hp.choice('beta1', (0.9, 0.9)),
        'beta2': hp.choice('beta2', (0.99, 0.999)),
        'weight': hp.choice('weight', (0, 0.0001))
    }
    params = sample(space)

    return params


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="CNN for predicting CSSBS")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")

    parser.add_argument("-g", dest="gpu", type=str, default='1',
                        help="choose gpu device.")
    parser.add_argument("-s", dest="seed", type=int, default=5,
                        help="Random seed to have reproducible results.")

    parser.add_argument("-b", dest="batch_size", type=int, default=1,
                        help="Number of sequences sent to the network in one step.")
    parser.add_argument("-e", dest="max_epoch", type=int, default=30,
                        help="Number of training steps.")

    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        device = torch.device("cuda:" + args.gpu)
    else:
        device = torch.device("cpu")
        torch.manual_seed(args.seed)
    motifLen = 20
    Data = np.load(osp.join(args.data_dir, '%s_train.npz' % args.name))
    data_tr, label_tr = Data['data'], Data['label']
    Data = np.load(osp.join(args.data_dir, '%s_val.npz' % args.name))
    data_val, label_val = Data['data'], Data['label']
    dim = data_val.shape[1]
    f = open(osp.join(args.checkpoint, 'score.txt'), 'w')
    f.write('auc\tprauc\n')
    # build training data generator
    train_data = EPIDataSet(data_tr, label_tr)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    # build test data generator
    val_data = EPIDataSet(data_val, label_val)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0)
    # implement
    auc_best = 0
    for _ in range(10):
        params = randomparams()
        model = DeepCNN(in_channels=dim, motiflen=motifLen)
        optimizer = optim.Adam(model.parameters(), lr=params['lr'],
                               betas=(params['beta1'], params['beta2']), weight_decay=params['weight'])
        criterion = nn.BCELoss()
        executor = Trainer(model=model,
                           optimizer=optimizer,
                           criterion=criterion,
                           device=device,
                           checkpoint=args.checkpoint,
                           start_epoch=0,
                           max_epoch=8,
                           train_loader=train_loader,
                           test_loader=val_loader,
                           lr_policy=None)

        auc, prauc, state_dict = executor.train()
        if auc_best < auc:
            print("Store the weights of the model in the current run.\n")
            auc_best = auc
            lr = params['lr']
            betas = (params['beta1'], params['beta2'])
            weight_decay = params['weight']
            checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
            torch.save({
                'model_state_dict': state_dict
            }, checkpoint_file)
    # loading the best model
    checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
    checkpoint = torch.load(checkpoint_file)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    criterion = nn.BCELoss()
    executor = Trainer(model=model,
                       optimizer=optimizer,
                       criterion=criterion,
                       device=device,
                       checkpoint=args.checkpoint,
                       start_epoch=0,
                       max_epoch=args.max_epoch,
                       train_loader=train_loader,
                       test_loader=val_loader,
                       lr_policy=scheduler)
    auc, prauc, state_dict = executor.train()
    torch.save({'model_state_dict': state_dict}, checkpoint_file)
    f.write("{:.3f}\t{:.3f}\n".format(auc, prauc))
    f.close()


if __name__ == "__main__":
    main()

