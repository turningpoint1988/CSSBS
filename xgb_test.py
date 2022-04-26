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


def boxplot(x, y, out_f, name):
    plt.figure(figsize=(6, 10))
    sns.set_theme(style="white")
    position = list(range(len(x), 0, -1))
    plt.barh(position, width=y, height=0.6, tick_label=x, facecolor='tan', edgecolor='r', alpha=0.6)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel("Features", fontsize=18)
    plt.xlabel("SHAP Value", fontsize=18)
    plt.title("Feature importance ({})".format(name), fontsize=22)
    plt.savefig(out_f, format='jpg', bbox_inches='tight', dpi=300)


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
    data_te = np.load(osp.join(args.data_dir, '%s_feature_test.npz' % args.name))
    feature_names = []
    with open(osp.join(args.data_dir, '%s_feature_names.txt' % args.name)) as f:
        for line in f:
            line_split = line.strip().split()
            feature_names.append(line_split[1])
    # test data
    feature_te, label_te = data_te['data'], data_te['label']
    dtest = xgb.DMatrix(feature_te)
    # load trained models to predict test data
    bst = xgb.Booster()
    bst.load_model(osp.join(args.outdir, 'model.json'))
    best_iteration = bst.best_iteration
    ypred = bst.predict(dtest, iteration_range=(0, best_iteration + 1), pred_contribs=False)
    auroc = roc_auc_score(label_te, ypred)
    auprc = average_precision_score(label_te, ypred)
    print("{}: {:.3f}\t{:.3f}\n".format(args.name, auroc, auprc))
    f = open(osp.join(args.outdir, 'score.txt'), 'w')
    f.write("{}\n".format(best_param))
    f.write("{:.3f}\t{:.3f}\n".format(auroc, auprc))
    f.close()
    # calculating SHAP values
    index_pos = (label_te == 1)
    index_neg = (label_te == 0)
    dtest_pos = xgb.DMatrix(feature_te[index_pos])
    dtest_neg = xgb.DMatrix(feature_te[index_neg])
    # load trained models to predict SHAP values
    SHAP_pos = bst.predict(dtest_pos, iteration_range=(0, best_iteration + 1), pred_contribs=True)
    SHAP_pos = SHAP_pos[:, :-1]
    feature_importance = np.sum(SHAP_pos, axis=0)
    index = np.argsort(np.asarray(feature_importance))
    index = index[::-1]
    values = []
    names = []
    for i in index:
        value = feature_importance[i]
        values.append(value)
        name = feature_names[i]
        names.append(name)
    top = 30
    # plotting
    boxplot(names[:top], values[:top], args.outdir + '/{}_SHAP_pos.jpg'.format(args.name), args.name)
    # for neg
    SHAP_neg = bst.predict(dtest_neg, iteration_range=(0, best_iteration + 1), pred_contribs=True)
    SHAP_neg = SHAP_neg[:, :-1]
    feature_importance = np.sum(SHAP_neg, axis=0)
    index = np.argsort(np.asarray(feature_importance))
    index = index[::-1]
    values = []
    names = []
    for i in index:
        value = feature_importance[i]
        values.append(value)
        name = feature_names[i]
        names.append(name)
    # plotting
    boxplot(names[:top], values[:top], args.outdir + '/{}_SHAP_neg.jpg'.format(args.name), args.name)


if __name__ == "__main__":
    main()

