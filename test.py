#!/usr/bin/python

import os
import sys
import argparse
import random
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

# custom functions defined by user
from model import DeepCNN
from datasets import EPIDataSet
from trainer import Trainer
from sklearn.metrics import average_precision_score, roc_auc_score


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
    parser.add_argument("-g", dest="gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("-c", dest="checkpoint", type=str, default='./models/',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    if torch.cuda.is_available():
        device = torch.device("cuda:" + args.gpu)
    else:
        device = torch.device("cpu")
    motifLen = 20
    f = open(osp.join(args.checkpoint, 'score.txt'), 'a')
    Data = np.load(osp.join(args.data_dir, '%s_test.npz' % args.name))
    data_te, label_te = Data['data'], Data['label']
    dim = data_te.shape[1]
    test_data = EPIDataSet(data_te, label_te)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)
    # Load weights
    checkpoint_file = osp.join(args.checkpoint, 'model_best.pth')
    chk = torch.load(checkpoint_file)
    state_dict = chk['model_state_dict']
    model = DeepCNN(in_channels=dim, motiflen=motifLen)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    label_p_all = []
    label_t_all = []
    for i_batch, sample_batch in enumerate(test_loader):
        X_data = sample_batch["data"].float().to(device)
        label = sample_batch["label"]
        with torch.no_grad():
            label_p = model(X_data)
        label_p_all.append(label_p.view(-1).data.cpu().numpy()[0])
        label_t_all.append(label.view(-1).data.cpu().numpy()[0])

    auc = roc_auc_score(label_t_all, label_p_all)
    prauc = average_precision_score(label_t_all, label_p_all)
    f.write("{:.3f}\t{:.3f}\n".format(auc, prauc))
    print("{}\t{:.3f}\t{:.3f}\n".format(args.name, auc, prauc))
    f.close()


if __name__ == "__main__":
    main()

