# coding:utf-8
import os.path as osp
import os
import sys
import argparse
import random
import numpy as np
from Bio import SeqIO
import pyBigWig
from hic_feature import *

seq_len = 601
THRESHOLD = 0.8
segment_len = 601
thres = 3
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


def jasparDict(jaspar):
    with open(jaspar) as f:
        lines = f.readlines()
    jaspar_dict = {}
    for i, line in enumerate(lines):
        line_split = line.strip().split()
        if len(line_split) == 0: continue
        if 'MOTIF' == line_split[0]:
            start = i + 2
            TF = line_split[-1].split('.')[-1]
        elif 'URL' == line_split[0]:
            end = i
            values = []
            for each in lines[start:end]:
                each_split = each.strip().split()
                values.append([float(x) for x in each_split])
            values = np.array(values, dtype=np.float32)
            # values = np.transpose(values)
            if TF not in jaspar_dict.keys():
                jaspar_dict[TF] = [values]
            else:
                jaspar_dict[TF].append(values)
    return jaspar_dict


def extractMotifs(ppi, jaspar_dict, target):
    candidates = [target]
    with open(ppi) as f1:
        f1.readline()
        for line in f1:
            line_split = line.strip().split()
            TF1 = line_split[0]
            TF2 = line_split[1]
            score = float(line_split[-1])
            if TF1 == target and score > THRESHOLD:
                candidates.append(TF2)

    candidates = list(set(candidates))
    candidates.sort()
    # search TF from jaspar
    candidates_filtered = []
    candidates_name = []
    for TF in candidates:
        if TF in jaspar_dict.keys():
            candidates_filtered.append(jaspar_dict[TF][0])
            candidates_name.append(TF)
    return candidates_filtered, candidates_name


def chromatin_feature(chrom, start, end, chromatin_files):
    num = int(seq_len / segment_len)
    features = []
    for bw in chromatin_files:
        try:
            sample = np.array(bw.values(chrom, start, end))
        except:
            print(chrom, start, end)
            sys.exit(0)
        sample[np.isnan(sample)] = 0.
        # sample = np.log10(1 + sample)
        for i in range(num):
            segment = sample[i*segment_len:(i+1)*segment_len]
            min = np.min(segment)
            mean = np.mean(segment)
            max = np.max(segment)
            features += [min, mean, max]
    return features


def get_chromatin_feature(line, chromatin_files):
    bed = line.split('-')
    chr = bed[0]
    start = int(bed[1])
    end = int(bed[2])
    features = chromatin_feature(chr, start, end, chromatin_files)
    
    return features


def motif_feature(seq, candidates):
    seq_dict = {'A': 'T', 'G': 'C',
                'C': 'G', 'T': 'A',
                'a': 'T', 'g': 'C',
                'c': 'G', 't': 'A'}
    num = int(seq_len / segment_len)
    features = []
    for pfm in candidates:
        pfm_len = len(pfm)
        for i in range(num):
            segment = seq[i*segment_len:(i+1)*segment_len]
            segment_onehot = np.array(one_hot(segment))
            scores_f = []
            for j in range(segment_len - pfm_len + 1):
                score = np.sum(pfm * segment_onehot[j:(j + pfm_len), :])
                scores_f.append(score)
            scores_f.sort(reverse=True)
            # reverse complement
            segment_rc = ''
            for c in segment[::-1]:
                segment_rc += seq_dict.get(c, 'N')
            segment_rc_onehot = np.array(one_hot(segment_rc))
            scores_b = []
            for j in range(segment_len - pfm_len + 1):
                score = np.sum(pfm * segment_rc_onehot[j:(j + pfm_len), :])
                scores_b.append(score)
            scores_b.sort(reverse=True)
            scores = scores_f[:thres] + scores_b[:thres]
            scores.sort(reverse=True)
            features += scores[:thres]
    return features


def encode_sequence(pos_file1, pos_file2, neg_file, sequence_dict, candidates):
    with open(pos_file1) as f1, open(pos_file2) as f2, open(neg_file) as f3:
        pos_bed1 = f1.readlines()
        pos_bed2 = f2.readlines()
        neg_bed = f3.readlines()

    bed_pos = []
    features_pos = []
    for line in (pos_bed1 + pos_bed2):
        bed = line.strip().split()
        chr = bed[0]
        start = int(bed[1])
        end = int(bed[2])
        if chr not in INDEX:
            continue
        start_p, end_p = relocation(chr, start, end)
        seq = str(sequence_dict[chr].seq[start_p:end_p])
        bed_pos.append("{}-{}-{}".format(chr, start_p, end_p))
        features_pos.append(motif_feature(seq, candidates))

    # negative set
    bed_neg = []
    features_neg = []
    for line in neg_bed:
        bed = line.strip().split()
        chr = bed[0]
        start = int(bed[1])
        end = int(bed[2])
        if chr not in INDEX:
            continue
        start_p, end_p = relocation(chr, start, end)
        seq = str(sequence_dict[chr].seq[start_p:end_p])
        bed_neg.append("{}-{}-{}".format(chr, start_p, end_p))
        features_neg.append(motif_feature(seq, candidates))

    return bed_pos, features_pos, bed_neg, features_neg


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
    root = osp.dirname(__file__) + '/GK'
    data_dir = osp.join(root, name)
    out_dir = osp.join(data_dir, 'data')
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
    # motif
    ppi = osp.join(data_dir, 'PPI/string_interactions.tsv')
    jaspar = osp.join(root, 'JASPAR2022_CORE_vertebrates_redundant.meme')
    jaspar_dict = jasparDict(jaspar)
    candidates, candidates_name = extractMotifs(ppi, jaspar_dict, name)
    # get the names of motif features
    motif_feature_name = []
    for x in candidates_name:
        motif_feature_name += [x] * int(seq_len / segment_len) * thres
    # encode sequences with one-hot
    sequence_dict = SeqIO.to_dict(SeqIO.parse(open(osp.dirname(__file__) + '/hg19/hg19.fa'), 'fasta'))
    print('Experiment on %s dataset' % name)
    pos_file1 = osp.join(data_dir, 'GM12878_specific.txt')
    pos_file2 = osp.join(data_dir, 'K562_specific.txt')
    neg_file = osp.join(data_dir, 'Shared.txt')
    bed_pos, motif_features_pos, bed_neg, motif_features_neg = \
        encode_sequence(pos_file1, pos_file2, neg_file, sequence_dict, candidates)
    # pre-load chromatin
    chromatin_dir = osp.join(root, 'Chromatin')
    chromatins = ["DNase", "H2A", "H3K4me1", "H3K4me2", "H3K4me3", "H3K9ac", "H3K9me3", "H3K27ac", "H3K27me3",
                  "H3K36me3", "H3K79me2", "H4K20me1", "MNase", "RRBS", "RNA"]
    gm12878_chromatin_files = [pyBigWig.open(osp.join(chromatin_dir, 'GM12878', x + '.bigWig')) for x in chromatins]
    k562_chromatin_files = [pyBigWig.open(osp.join(chromatin_dir, 'K562', x + '.bigWig')) for x in chromatins]
    # get the names of chromatin features
    chromatin_feature_name = []
    for x in chromatins:
        chromatin_feature_name += ['gm12878_' + x] * int(seq_len / segment_len) * 3
    for x in chromatins:
        chromatin_feature_name += ['k562_' + x] * int(seq_len / segment_len) * 3
    for x in chromatins:
        chromatin_feature_name += ['diff_' + x] * int(seq_len / segment_len) * 3
    # pre-load hi-c
    hic_dir = osp.join(root, 'Hi-C')
    hic = ['5kb']
    resolutions = [5000]
    number = 20
    gm12878_hic_dict = []
    k562_hic_dict = []
    gm12878_hic_files = [osp.join(hic_dir, 'GM12878', x + '.cool') for x in hic]
    for hic_file, resolution in zip(gm12878_hic_files, resolutions):
        gm12878_hic_dict.append(load_hic(hic_file, resolution))
    k562_hic_files = [osp.join(hic_dir, 'K562', x + '.cool') for x in hic]
    for hic_file, resolution in zip(k562_hic_files, resolutions):
        k562_hic_dict.append(load_hic(hic_file, resolution))
    # get the names of hic features
    hic_feature_name = []
    count = 3
    for x in hic:
        hic_feature_name += ['gm12878_hic_' + x] * count + ['gm12878_dis_' + x] * count
    for x in hic:
        hic_feature_name += ['k562_hic_' + x] * count + ['k562_dis_' + x] * count
    for x in hic:
        hic_feature_name += ['diff_hic_' + x] * count + ['diff_dis_' + x] * count
    # merge feature names
    feature_names = motif_feature_name + chromatin_feature_name + hic_feature_name
    # for positive samples
    gm12878_chromatin_pos = []
    k562_chromatin_pos = []
    gm12878_hic_pos = []
    k562_hic_pos = []
    for i, bed in enumerate(bed_pos):
        if (i + 1) % 100 == 0:
            print("the {}-th positive element is being processed.".format(i+1))
        # encode chromatin features for GM12878 and K562
        # GM12878 cell line
        gm12878_chromatin_pos.append(get_chromatin_feature(bed, gm12878_chromatin_files))
        # K562 cell line
        k562_chromatin_pos.append(get_chromatin_feature(bed, k562_chromatin_files))
        # encode hic features for GM12878 and K562
        # GM12878 cell line
        hic_features = []
        for hic_dict, resolution in zip(gm12878_hic_dict, resolutions):
            hic_features += get_hic(bed, hic_dict, resolution, number)
        gm12878_hic_pos.append(hic_features)
        # K562 cell line
        hic_features = []
        for hic_dict, resolution in zip(k562_hic_dict, resolutions):
            hic_features += get_hic(bed, hic_dict, resolution, number)
        k562_hic_pos.append(hic_features)

    # merge features
    motif_features_pos = np.array(motif_features_pos, dtype=np.float32)
    gm12878_chromatin_pos = np.array(gm12878_chromatin_pos, dtype=np.float32)
    k562_chromatin_pos = np.array(k562_chromatin_pos, dtype=np.float32)
    chromatin_pos = gm12878_chromatin_pos - k562_chromatin_pos
    gm12878_hic_pos = np.array(gm12878_hic_pos, dtype=np.float32)
    k562_hic_pos = np.array(k562_hic_pos, dtype=np.float32)
    hic_pos = gm12878_hic_pos - k562_hic_pos

    features_pos = np.concatenate((motif_features_pos, gm12878_chromatin_pos, k562_chromatin_pos, chromatin_pos,
                                   gm12878_hic_pos, k562_hic_pos, hic_pos), axis=1)
    # for negative samples
    gm12878_chromatin_neg = []
    k562_chromatin_neg = []
    gm12878_hic_neg = []
    k562_hic_neg = []
    for i, bed in enumerate(bed_neg):
        if (i + 1) % 100 == 0:
            print("the {}-th negative element is being processed.".format(i+1))
        # encode chromatin features for GM12878 and K562
        # GM12878 cell line
        gm12878_chromatin_neg.append(get_chromatin_feature(bed, gm12878_chromatin_files))
        # K562 cell line
        k562_chromatin_neg.append(get_chromatin_feature(bed, k562_chromatin_files))
        # encode hic features for GM12878 and K562
        # GM12878 cell line
        hic_features = []
        for hic_dict, resolution in zip(gm12878_hic_dict, resolutions):
            hic_features += get_hic(bed, hic_dict, resolution, number)
        gm12878_hic_neg.append(hic_features)
        # K562 cell line
        hic_features = []
        for hic_dict, resolution in zip(k562_hic_dict, resolutions):
            hic_features += get_hic(bed, hic_dict, resolution, number)
        k562_hic_neg.append(hic_features)
    # merge features
    motif_features_neg = np.array(motif_features_neg, dtype=np.float32)
    gm12878_chromatin_neg = np.array(gm12878_chromatin_neg, dtype=np.float32)
    k562_chromatin_neg = np.array(k562_chromatin_neg, dtype=np.float32)
    chromatin_neg = gm12878_chromatin_neg - k562_chromatin_neg
    gm12878_hic_neg = np.array(gm12878_hic_neg, dtype=np.float32)
    k562_hic_neg = np.array(k562_hic_neg, dtype=np.float32)
    hic_neg = gm12878_hic_neg - k562_hic_neg

    features_neg = np.concatenate((motif_features_neg, gm12878_chromatin_neg, k562_chromatin_neg, chromatin_neg,
                                   gm12878_hic_neg, k562_hic_neg, hic_neg), axis=1)
    # close
    for file in gm12878_chromatin_files:
        file.close()
    for file in k562_chromatin_files:
        file.close()

    # split data as the train, validation, and test sets
    val_set = ['chr18']
    test_set = ['chr8', 'chr16']
    index_tr, index_val, index_te = getindex(bed_pos, val_set, test_set)
    label_pos_te = np.ones(len(index_te))
    label_pos_tr = np.ones(len(index_tr))
    label_pos_val = np.ones(len(index_val))
    # feature
    features_pos_te = features_pos[index_te]
    features_pos_tr = features_pos[index_tr]
    features_pos_val = features_pos[index_val]

    # neg
    index_tr, index_val, index_te = getindex(bed_neg, val_set, test_set)
    label_neg_te = np.zeros(len(index_te))
    label_neg_tr = np.zeros(len(index_tr))
    label_neg_val = np.zeros(len(index_val))
    # feature
    features_neg_te = features_neg[index_te]
    features_neg_tr = features_neg[index_tr]
    features_neg_val = features_neg[index_val]

    label_te = np.concatenate((label_pos_te, label_neg_te), axis=0)
    label_tr = np.concatenate((label_pos_tr, label_neg_tr), axis=0)
    label_val = np.concatenate((label_pos_val, label_neg_val), axis=0)
    # feature
    features_te = np.concatenate((features_pos_te, features_neg_te), axis=0)
    features_tr = np.concatenate((features_pos_tr, features_neg_tr), axis=0)
    features_val = np.concatenate((features_pos_val, features_neg_val), axis=0)
    # store features
    np.savez(out_dir + '/%s_feature_test.npz' % name, data=features_te, label=label_te)
    np.savez(out_dir + '/%s_feature_train.npz' % name, data=features_tr, label=label_tr)
    np.savez(out_dir + '/%s_feature_val.npz' % name, data=features_val, label=label_val)
    print("{}: The train features are: {}x{}".format(name, features_tr.shape[0], features_tr.shape[1]))
    print("{}: The val features are: {}x{}".format(name, features_val.shape[0], features_val.shape[1]))
    print("{}: The test features are: {}x{}".format(name, features_te.shape[0], features_te.shape[1]))
    # store feature names
    f1 = open(out_dir + '/%s_feature_names.txt' % name, 'w')
    i = 0
    for feat in feature_names:
        f1.write('{}\t{}\n'.format(i, feat))
        i = i + 1
    f1.close()
    print("{}: The feature names are: {}".format(name, len(feature_names)))


if __name__ == '__main__':  main()
