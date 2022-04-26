#!/usr/bin/env python

import os
import os.path as osp


def download(inputfile, outdir):
    with open(inputfile) as f:
        lines = f.readlines()
    
    if not osp.exists(outdir):
            os.mkdir(outdir)
    for line in lines:
        line_split = line.strip().split()
        if len(line_split) == 1:
            accession = line_split[0]
            name = accession.split('-')[1]
            name = name.split('.')[0]
            url = "https://egg2.wustl.edu/roadmap/data/byFileType/signal/consolidated/macs2signal/pval/{}".format(accession)
            print("downloading p-value bigWig for {}...".format(name))
            outfile = outdir + '/{}.bigWig'.format(name)
            os.system('curl -o {} -J -L {}'.format(outfile, url))
        else:
            name = line_split[0]
            accession = line_split[1]
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.bigWig'.format(accession, accession)
            print("downloading p-value bigWig for {}...".format(name))
            outfile = outdir + '/{}.bigWig'.format(name)
            os.system('curl -o {} -J -L {}'.format(outfile, url))   


root = osp.dirname(__file__)
# GM12878
inputfile = root + '/Chromatin/GM12878/GM12878_chromatin.list'
outdir = root + '/Chromatin/GM12878'
download(inputfile, outdir)
# K562
inputfile = root + '/Chromatin/K562/K562_chromatin.list'
outdir = root + '/Chromatin/K562'
download(inputfile, outdir)


