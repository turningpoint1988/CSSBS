#!/usr/bin/python

import os
import sys
import argparse
import random
import numpy as np
import os.path as osp
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


def plotting(bst, outdir, num):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 15)
    xgb.plot_importance(bst, ax=ax, grid=False, max_num_features=50, show_values=False, height=0.5)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.ylabel("Features", fontsize=20)
    plt.xlabel("F score", fontsize=20)
    plt.title("Feature importance", fontsize=25)
    fig.savefig(osp.join(outdir, 'feature.jpg'), format='jpg', bbox_inches='tight', dpi=300)
    plt.close(fig)
    #
    fig, ax = plt.subplots()
    fig.set_size_inches(40, 10)
    xgb.plot_tree(bst, num_trees=num, ax=ax, **{'condition_node_params': {'shape': 'box', 'style': 'filled,rounded', 'fillcolor': '#78bceb'},
                                                'leaf_node_params': {'shape': 'box', 'style': 'filled', 'fillcolor': '#e48038'}})
    fig.savefig(osp.join(outdir, 'tree.jpg'), format='jpg', bbox_inches='tight', dpi=300)
    plt.close(fig)


def get_args():
    """Parse all the arguments.

        Returns:
          A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="xgboost")

    parser.add_argument("-d", dest="data_dir", type=str, default=None,
                        help="A directory containing the training data.")
    parser.add_argument("-n", dest="name", type=str, default=None,
                        help="The name of a specified data.")
    parser.add_argument("-o", dest="outdir", type=str, default='',
                        help="Where to save snapshots of the model.")

    return parser.parse_args()


def main():
    """Create the model and start the training."""
    args = get_args()
    # loading data
    data_tr = np.load(osp.join(args.data_dir, '%s_feature_train.npz' % args.name))
    data_va = np.load(osp.join(args.data_dir, '%s_feature_val.npz' % args.name))
    data_te = np.load(osp.join(args.data_dir, '%s_feature_test.npz' % args.name))
    feature_names = []
    with open(osp.join(args.data_dir, '%s_feature_names.txt' % args.name)) as f:
        for line in f:
            line_split = line.strip().split()
            feature_names.append(line_split[1])
    # training data
    feature_tr, label_tr = data_tr['data'], data_tr['label']
    dtrain = xgb.DMatrix(feature_tr, label=label_tr)
    # validation data
    feature_va, label_va = data_va['data'], data_va['label']
    dvalidation = xgb.DMatrix(feature_va, label=label_va)
    # test data
    feature_te, label_te = data_te['data'], data_te['label']
    dtest = xgb.DMatrix(feature_te)
    # Hyper-parameter selection
    hyperparams = {
        'max_depth': [5, 6, 7, 8],
        'learning_rate': [0.01, 0.1, 0.3, 0.5, 1],
        'subsample': [0.8, 1],
        'colsample_bytree': [0.6, 0.8, 1]
    }
    params = list(ParameterGrid(hyperparams))
    # fixed parameters
    num_round = 1000
    early_stopping = 50
    evallist = [(dtrain, 'train'), (dvalidation, 'eval')]
    # selecting the best parameters
    best_score = 0
    best_param = {}
    for param in params:
        param['booster'] = 'gbtree'
        param['objective'] = 'binary:logistic'
        param['nthread'] = 4
        param['eval_metric'] = ['error', 'aucpr', 'auc']
        # train
        bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=evallist, early_stopping_rounds=early_stopping)
        # prediction: bst.best_score, bst.best_iteration
        score = bst.best_score
        if best_score < score:
            best_score = score
            best_param = param
    print("the best parameter is {}".format(best_param))
    #
    bst = xgb.train(best_param, dtrain, num_boost_round=num_round, evals=evallist, early_stopping_rounds=early_stopping)
    best_iteration = bst.best_iteration
    ypred = bst.predict(dtest, iteration_range=(0, best_iteration + 1), pred_contribs=False, pred_interactions=False)
    auroc = roc_auc_score(label_te, ypred)
    auprc = average_precision_score(label_te, ypred)
    print("{}: {:.3f}\t{:.3f}\n".format(args.name, auroc, auprc))
    # plotting
    plotting(bst, args.outdir, best_iteration)
    # save
    bst.save_model(osp.join(args.outdir, 'model.json'))
    f = open(osp.join(args.outdir, 'score.txt'), 'w')
    f.write("{}\n".format(best_param))
    f.write("{:.3f}\t{:.3f}\n".format(auroc, auprc))
    f.close()
    #
    feature_importance = bst.get_score()
    f = open(osp.join(args.outdir, 'feature_importance.txt'), 'w')
    keys = list(feature_importance.keys())
    values = list(feature_importance.values())
    index = np.argsort(np.asarray(values))
    index = index[::-1]
    for i in index:
        key = keys[i]
        value = values[i]
        name = feature_names[int(key[1:])]
        f.write("{}\t{}\t{}\n".format(key, value, name))
    f.close()
    # load trained models to predict test data
    # bst = xgb.Booster()
    # bst.load_model(osp.join(args.outdir, 'model.json'))
    # best_iteration = bst.best_iteration
    # ypred = bst.predict(dtest, iteration_range=(0, best_iteration + 1), pred_contribs=False, pred_interactions=False)
    # auroc = roc_auc_score(label_te, ypred)
    # auprc = average_precision_score(label_te, ypred)
    # f1 = f1_score(label_te, [int(x > 0.5) for x in ypred])
    # print("{}: {:.3f}\t{:.3f}\t{:.3f}\n".format(args.name, f1, auroc, auprc))
    # save feature_importance


if __name__ == "__main__":
    main()

