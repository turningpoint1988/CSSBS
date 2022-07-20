# coding:utf-8
import os.path as osp
import os
import sys
import argparse
import random
import numpy as np
from Bio import SeqIO
import pyBigWig

seq_len = 601
INDEX = ['chr' + str(i + 1) for i in range(23)]
INDEX[22] = 'chrX'
CHROM = {}

with open(osp.dirname(__file__) + '/hg19/chromsize') as f:
    for i in f:
        line_split = i.strip().split()
        CHROM[line_split[0]] = int(line_split[1])


def one_hot(seq):
    seq_dict = {'A':[1, 0, 0, 0], 'G':[0, 0, 1, 0],
                'C':[0, 1, 0, 0], 'T':[0, 0, 0, 1],
                'a':[1, 0, 0, 0], 'g':[0, 0, 1, 0],
                'c':[0, 1, 0, 0], 't':[0, 0, 0, 1]}
    temp = []
    for c in seq:
        temp.append(seq_dict.get(c, [0, 0, 0, 0]))
    return temp


def relocation(chrom, start, end):
    original_len = end - start
    if original_len < seq_len:
        start_update = start - np.ceil((seq_len - original_len) / 2)
    elif original_len > seq_len:
        start_update = start + np.ceil((original_len - seq_len) / 2)
    else:
        start_update = start
    end_update = start_update + seq_len
    if end_update > CHROM[chrom]:
        end_update = CHROM[chrom]
        start_update = end_update - seq_len
    return int(start_update), int(end_update)


def encode_sequence(pos_diff1, pos_diff2, neg_file, sequence_dict, gm12878_chromatin, k562_chromatin):
    with open(pos_diff1) as f1, open(pos_diff2) as f2, open(neg_file) as f3:
        pos_bed1 = f1.readlines()
        pos_bed2 = f2.readlines()
        neg_bed = f3.readlines()
    data_pos = []
    gm12878_chromatin_pos = []
    k562_chromatin_pos = []
    chromatin_pos = []
    bed_pos = []
    for line in (pos_bed1 + pos_bed2):
        bed = line.strip().split()
        chr = bed[0]
        start = int(bed[1])
        end = int(bed[2])
        if chr not in INDEX:
            continue
        start_p, end_p = relocation(chr, start, end)
        seq = str(sequence_dict[chr].seq[start_p:end_p])
        data_pos.append(one_hot(seq))
        sample1 = np.array(gm12878_chromatin.values(chr, start_p, end_p))
        sample1[np.isnan(sample1)] = 0.
        sample1 = np.log10(1 + sample1)
        gm12878_chromatin_pos.append([sample1])
        sample2 = np.array(k562_chromatin.values(chr, start_p, end_p))
        sample2[np.isnan(sample2)] = 0.
        sample2 = np.log10(1 + sample2)
        k562_chromatin_pos.append([sample2])
        bed_pos.append("{}-{}-{}".format(chr, start_p, end_p))
        chromatin_pos.append([sample1 - sample2])

    data_pos = np.array(data_pos, dtype=np.float32)
    data_pos = data_pos.transpose((0, 2, 1))
    gm12878_chromatin_pos = np.array(gm12878_chromatin_pos, dtype=np.float32)
    k562_chromatin_pos = np.array(k562_chromatin_pos, dtype=np.float32)
    chromatin_pos = np.array(chromatin_pos, dtype=np.float32)
    data_pos = np.concatenate((data_pos, gm12878_chromatin_pos, k562_chromatin_pos, chromatin_pos), axis=1)
    # data_pos = np.concatenate((data_pos, chromatin_pos), axis=1)
    # negative set
    data_neg = []
    gm12878_chromatin_neg = []
    k562_chromatin_neg = []
    chromatin_neg = []
    bed_neg = []
    for line in neg_bed:
        bed = line.strip().split()
        chr = bed[0]
        start = int(bed[1])
        end = int(bed[2])
        if chr not in INDEX:
            continue
        start_p, end_p = relocation(chr, start, end)
        seq = str(sequence_dict[chr].seq[start_p:end_p])
        data_neg.append(one_hot(seq))
        sample1 = np.array(gm12878_chromatin.values(chr, start_p, end_p))
        sample1[np.isnan(sample1)] = 0.
        sample1 = np.log10(1 + sample1)
        gm12878_chromatin_neg.append([sample1])
        sample2 = np.array(k562_chromatin.values(chr, start_p, end_p))
        sample2[np.isnan(sample2)] = 0.
        sample2 = np.log10(1 + sample2)
        k562_chromatin_neg.append([sample2])
        bed_neg.append("{}-{}-{}".format(chr, start_p, end_p))
        chromatin_neg.append([sample1 - sample2])

    data_neg = np.array(data_neg, dtype=np.float32)
    data_neg = data_neg.transpose((0, 2, 1))
    gm12878_chromatin_neg = np.array(gm12878_chromatin_neg, dtype=np.float32)
    k562_chromatin_neg = np.array(k562_chromatin_neg, dtype=np.float32)
    chromatin_neg = np.array(chromatin_neg, dtype=np.float32)
    data_neg = np.concatenate((data_neg, gm12878_chromatin_neg, k562_chromatin_neg, chromatin_neg), axis=1)
    # data_neg = np.concatenate((data_neg, chromatin_neg), axis=1)

    return data_pos, bed_pos, data_neg, bed_neg


def getindex(bed, val_set, test_set):
    index = list(range(len(bed)))
    index_tr = []
    index_val = []
    index_te = []
    for i in index:
        chr = bed[i].split('-')[0]
        if chr in val_set:
            index_val.append(i)
        elif chr in test_set:
            index_te.append(i)
        else:
            index_tr.append(i)
    return index_tr, index_val, index_te


def get_args():
    parser = argparse.ArgumentParser(description="pre-process data.")
    parser.add_argument("-n", dest="name", type=str, default='')

    return parser.parse_args()


def main():
    params = get_args()
    name = params.name
    root = osp.join(osp.dirname(__file__), 'GK')
    data_dir = osp.join(root, name)
    chromatin_dir = osp.join(root, 'Chromatin')
    out_dir = osp.join(data_dir, 'encode')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    # motif
    # encode sequences with one-hot
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(osp.dirname(__file__) + '/hg19/hg19.fa'), 'fasta'))
    print('Experiment on %s dataset' % name)
    gm_specific = osp.join(data_dir, 'GM12878_specific.txt')
    k562_specific = osp.join(data_dir, 'K562_specific.txt')
    neg_file = osp.join(data_dir, 'shared.txt')
    gm12878_chromatin = pyBigWig.open(osp.join(chromatin_dir, 'GM12878', 'DNase.bigWig'))
    k562_chromatin = pyBigWig.open(osp.join(chromatin_dir, 'K562', 'DNase.bigWig'))
    data_pos, bed_pos, data_neg, bed_neg = encode_sequence(gm_specific, k562_specific, neg_file,
                                                           sequence_dict, gm12878_chromatin, k562_chromatin)
    # close
    gm12878_chromatin.close()
    k562_chromatin.close()

    # split data as the train, validation, and test sets
    val_set = ['chr18']
    test_set = ['chr8', 'chr16']
    index_tr, index_val, index_te = getindex(bed_pos, val_set, test_set)
    data_pos_te = data_pos[index_te]
    label_pos_te = np.ones(len(index_te))
    data_pos_tr = data_pos[index_tr]
    label_pos_tr = np.ones(len(index_tr))
    data_pos_val = data_pos[index_val]
    label_pos_val = np.ones(len(index_val))

    # neg
    index_tr, index_val, index_te = getindex(bed_neg, val_set, test_set)
    data_neg_te = data_neg[index_te]
    label_neg_te = np.zeros(len(index_te))
    data_neg_tr = data_neg[index_tr]
    label_neg_tr = np.zeros(len(index_tr))
    data_neg_val = data_neg[index_val]
    label_neg_val = np.zeros(len(index_val))

    data_te = np.concatenate((data_pos_te, data_neg_te), axis=0)
    label_te = np.concatenate((label_pos_te, label_neg_te))
    data_tr = np.concatenate((data_pos_tr, data_neg_tr), axis=0)
    label_tr = np.concatenate((label_pos_tr, label_neg_tr), axis=0)
    data_val = np.concatenate((data_pos_val, data_neg_val), axis=0)
    label_val = np.concatenate((label_pos_val, label_neg_val), axis=0)
    # store
    np.savez(out_dir + '/%s_test.npz' % name, data=data_te, label=label_te)
    np.savez(out_dir + '/%s_train.npz' % name, data=data_tr, label=label_tr)
    np.savez(out_dir + '/%s_val.npz' % name, data=data_val, label=label_val)
    print("{}: The train data are: {}x{}x{}".format(name, data_tr.shape[0], data_tr.shape[1], data_tr.shape[2]))
    print("{}: The val data are: {}x{}x{}".format(name, data_val.shape[0], data_val.shape[1], data_val.shape[2]))
    print("{}: The test data are: {}x{}x{}".format(name, data_te.shape[0], data_te.shape[1], data_te.shape[2]))


if __name__ == '__main__':  main()
