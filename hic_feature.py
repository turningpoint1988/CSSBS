#!/usr/bin/env python
import numpy as np
import sys

import struct
import os
import straw
import cooler
# functions for processing hi-c files


def csr_contact_matrix(norm, hicfile, chr1loc, chr2loc, unit,
                       binsize, is_synapse=False):
    '''
    Extract the contact matrix from .hic in CSR sparse format
    '''

    from scipy.sparse import csr_matrix
    import numpy as np
    import straw

    tri_list = straw.straw(norm, hicfile, chr1loc,
                           chr2loc, unit, binsize, is_synapse)
    row = [r//binsize for r in tri_list[0]]
    col = [c//binsize for c in tri_list[1]]
    value = tri_list[2]
    N = max(col) + 1

    # re-scale KR matrix to ICE-matrix range
    M = csr_matrix((value, (row, col)), shape=(N, N))
    margs = np.array(M.sum(axis=0)).ravel() + \
        np.array(M.sum(axis=1)).ravel() - M.diagonal(0)
    margs[np.isnan(margs)] = 0
    scale = margs[margs != 0].mean()
    row, col = M.nonzero()
    value = M.data / scale
    M = csr_matrix((value, (row, col)), shape=(N, N))

    return M


def get_hic_chromosomes(hicfile, res):

    hic_info = read_hic_header(hicfile)
    chromosomes = []
    # handle with inconsistency between .hic header and matrix data
    for c, Len in hic_info['chromsizes'].items():
        try:
            loc = '{0}:{1}:{2}'.format(c, 0, min(Len, 100000))
            _ = straw.straw('NONE', hicfile, loc, loc, 'BP', res)
            chromosomes.append(c)
        except:
            pass

    return chromosomes


def find_chrom_pre(chromlabels):

    ini = chromlabels[0]
    if ini.startswith('chr'):
        return 'chr'

    else:
        return ''


def readcstr(f):
    buf = ""
    while True:
        b = f.read(1)
        b = b.decode('utf-8', 'backslashreplace')
        if b is None or b == '\0':
            return str(buf)
        else:
            buf = buf + b


def read_hic_header(hicfile):

    if not os.path.exists(hicfile):
        return None  # probably a cool URI

    req = open(hicfile, 'rb')
    # print(req.read(3))
    magic_string = struct.unpack('<3s', req.read(3))[0]
    # print(magic_string)
    req.read(1)
    if magic_string != b"HIC":
        return None  # this is not a valid .hic file

    info = {}
    version = struct.unpack('<i', req.read(4))[0]
    info['version'] = str(version)

    masterindex = struct.unpack('<q', req.read(8))[0]
    info['Master index'] = str(masterindex)

    genome = ""
    c = req.read(1).decode("utf-8")
    while (c != '\0'):
        genome += c
        c = req.read(1).decode("utf-8")
    info['Genome ID'] = str(genome)

    nattributes = struct.unpack('<i', req.read(4))[0]
    attrs = {}
    for i in range(nattributes):
        key = readcstr(req)
        value = readcstr(req)
        attrs[key] = value
    info['Attributes'] = attrs

    nChrs = struct.unpack('<i', req.read(4))[0]
    chromsizes = {}
    for i in range(nChrs):
        name = readcstr(req)
        length = struct.unpack('<i', req.read(4))[0]
        if name != 'ALL':
            chromsizes[name] = length

    info['chromsizes'] = chromsizes

    info['Base pair-delimited resolutions'] = []
    nBpRes = struct.unpack('<i', req.read(4))[0]
    for i in range(nBpRes):
        res = struct.unpack('<i', req.read(4))[0]
        info['Base pair-delimited resolutions'].append(res)

    info['Fragment-delimited resolutions'] = []
    nFrag = struct.unpack('<i', req.read(4))[0]
    for i in range(nFrag):
        res = struct.unpack('<i', req.read(4))[0]
        info['Fragment-delimited resolutions'].append(res)

    return info
# end
# functions for extracting interacting features


def buildmatrix(Matrix, start, resolution, num):
    distance = []
    features = []
    x = start
    try:
        vector = Matrix[x, :].toarray()[0]
        vector[np.isnan(vector)] = 0.
        vector[x] = 0.
        index = np.argsort(vector)
        index = index[::-1]
        for i in index[:num]:
            distance.append(abs(i - x)*resolution)
            features.append(vector[i])
    except:
        print("retrieval is wrong.")
        sys.exit(0)
    features = np.asarray(features, dtype=np.float32)
    distance = np.asarray(distance, dtype=np.float32)
    features_out = [np.min(features), np.mean(features), np.max(features),
                    np.min(distance), np.mean(distance), np.max(distance)]
    # features_out = features + distance
    return features_out


def parsebed(bed, res=5000):
    coords = []
    s = bed.split('-')
    chrom = s[0]
    start = int(s[1])
    end = int(s[2])
    start /= res
    end /= res
    if int(start) == int(end):
        coords.append((chrom, int(start)))
    else:
        if int(end) - start > end - int(end):
            coords.append((chrom, int(start)))
        else:
            coords.append((chrom, int(end)))

    return coords
# end


def get_hic_feature(bed, path, resolution, number, balance=True):
    # more robust to check if a file is .hic
    hic_info = read_hic_header(path)

    if hic_info is None:
        hic = False
    else:
        hic = True
    #

    coords = parsebed(bed, res=resolution)

    if not hic:
        Lib = cooler.Cooler(path)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = get_hic_chromosomes(path, resolution)

    # pos
    for coord in coords:
        chrom, start = coord
        if not hic:
            X = Lib.matrix(balance=balance, sparse=True).fetch(chrom).tocsr()
        else:
            if balance:
                X = csr_contact_matrix('KR', path, chrom, chrom, 'BP', resolution)
            else:
                X = csr_contact_matrix('NONE', path, chrom, chrom, 'BP', resolution)

        try:
            features = buildmatrix(X, start, resolution, num=number)
        except:
            print(' failed to gather fts')

    return features


def load_hic(hic_file, resolution, balance=True):
    # more robust to check if a file is .hic
    hic_info = read_hic_header(hic_file)

    if hic_info is None:
        hic = False
    else:
        hic = True
    #

    if not hic:
        Lib = cooler.Cooler(hic_file)
        chromosomes = Lib.chromnames[:]
    else:
        chromosomes = get_hic_chromosomes(hic_file, resolution)

    dict = {}
    for chrom in chromosomes:
        if chrom == "chrY" or chrom == "chrM":
            continue
        if not hic:
            X = Lib.matrix(balance=balance, sparse=True).fetch(chrom).tocsr()
        else:
            if balance:
                X = csr_contact_matrix('KR', path, chrom, chrom, 'BP', resolution)
            else:
                X = csr_contact_matrix('NONE', path, chrom, chrom, 'BP', resolution)
        dict[chrom] = X

    return dict


def get_hic(bed, hic_dict, resolution, number=5):
    coords = parsebed(bed, res=resolution)
    for coord in coords:
        chrom, start = coord
        X = hic_dict[chrom]
        try:
            features = buildmatrix(X, start, resolution, num=number)
        except:
            print(' failed to gather fts')

    return features
