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
        name = line_split[0]
        accession = line_split[1]
        suffix = line_split[2]
        if 'bed' in suffix:
            print("downloading peak files for {}...".format(name))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, suffix)
            outfile = outdir + '/{}.bed.gz'.format(name)
            os.system('curl -o {} -J -L {}'.format(outfile, url))
            os.system('gunzip {}'.format(outfile))
        else:
            print("downloading bam files for {}...".format(name))
            url = 'https://www.encodeproject.org/files/{}/@@download/{}.{}'.format(accession, accession, suffix)  
            outfile = outdir + '/{}.bam'.format(name)
            os.system('curl -o {} -J -L {}'.format(outfile, url))   

root = osp.dirname(__file__)
inputfile = root + '/data.list'
outdir = root + '/Download'
args = get_args()
download(inputfile, outdir)




