# CSSBSs
Computational prediction and characterization of cell-type-specific and shared binding sites (CSSBSs). 

<p align="center"> 
<img src=https://github.com/turningpoint1988/CSSBSs/blob/main/Picture1.jpg>
</p>

<p align="center"> 
<img src=https://github.com/turningpoint1988/CSSBSs/blob/main/Picture1.jpg>
</p>

## Prerequisites and Dependencies

- XGBoost [[Install]](https://xgboost.readthedocs.io/en/latest/install.html)
- Pytorch 1.1 [[Install]](https://pytorch.org/)
- CUDA 9.0
- Python 3.6
- Python packages: biopython, scikit-learn, pyBigWig, scipy, pandas, matplotlib, seaborn
- Download [hg19.fa](https://hgdownload.soe.ucsc.edu/downloads.html#human) then unzip them and put them into `hg19/`

## Other Tools

- [MEME Suite](https://meme-suite.org/meme/doc/download.html): The tool integrates several methods used by this paper, including MEME, TOMTOM and FIMO.
- [Bedtools](https://bedtools.readthedocs.io/en/latest/content/installation.html): The tool is used for data preparation.
- [DESeq2](http://www.bioconductor.org/packages/release/bioc/html/DESeq2.html): The tool is used for data preparation.


## Downloading Datasets

We have offered the python scripts for downloading ChIP-seq datasets and chromatin landscapes by just runing: <br>

```
cd ./GK/
python get_data.py & get_chromatin.py 
```

For those TFs which lack binding peaks, we will use the peak calling software SPP [1] to generate corresponding binding peaks.


## Differential Binding sites Preparation
We used DESeq2[2] to generate all cell-type-specific and shared binding peaks, which can be found in the directory 'GK'. If you want to generate them from new TFs and cell types, we also provided the R script 'DESeq2.R' in the directory 'GK'. However, before doing this, you should calculate the number of reads from each cell line falling into the merged peaks by Bedtools, which are separately denoted by 'GM12878_count.bed' and 'K562_count.bed' in the R script. 


## Model Training

We constructed two models, in which one is based on XGBoost and another is based on CNN.


### 1. Training the XGBoost-based model:

```
python xgb_classifier.py -d <> -n <> -c <>
```

| Arguments  | Description                                                                |
| ---------- | ---------------------------------------------------------------------------|
| -d         | The path of a specified dataset, e.g. /your_path/CSSBSs/GK                 |
| -n         | The name of the specified dataset, e.g. CTCF                               |
| -c         | The path for storing models, e.g. /your_path/CSSBSs/models_xgb/CTCF   |

### Output

A trained model for the XGBoost-based model on the specified dataset. For example, A trained model is saved as `/your_path/CSSBSs/models_xgb/CTCF/model.json`. 


### 2. Training the CNN-based model:

```
python train.py -d <> -n <> -g <> -s <> -b <> -e <> -c <>
```

| Arguments  | Description                                                                      |
| ---------- | -------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data     |
| -n         | The name of the specified dataset, e.g. CTCF                                     |
| -g         | The GPU device id (default is 0)                                                 |
| -s         | Random seed                                                                      |
| -b         | The number of sequences in a batch size (default is 300)                         |
| -e         | The epoch of training steps (default is 50)                                      |
| -c         | The path for storing models, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF       |

### Output

A trained model for the CNN-based model on the specified dataset. For example, A trained model is saved as `/your_path/CSSBSs/models_cnn/CTCF/model_best.pth`.

## Testing 

### Testing the XGBoost-based model:

```
python xgb_test.py -d <> -n <> -c <>
```

| Arguments  | Description                                                                |
| ---------- | ---------------------------------------------------------------------------|
| -d         | The path of a specified dataset, e.g. /your_path/CSSBSs/GK                 |
| -n         | The name of the specified dataset, e.g. CTCF                               |
| -c         | The path of the trained model, e.g. /your_path/CSSBSs/models_xgb/CTCF   |

### Output

Generating `score.txt` recording the area under the receiver operating characteristic curve (AUC) and the area under the precision-recall curve (PRAUC).

### Testing the CNN-based model:

```
python test.py -d <> -n <> -c <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/CSSBSs/GK                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -c         | The trained model path of a specified dataset, e.g. /your_path/CSSBSs/models_cnn/CTCF |

### Output

Generating `score.txt` recording the area under the receiver operating characteristic curve (AUC) and the area under the precision-recall curve (PRAUC).

## References

[1] Kharchenko, Peter V., Michael Y. Tolstorukov, and Peter J. Park. "Design and analysis of ChIP-seq experiments for DNA-binding proteins." Nature biotechnology 26.12 (2008): 1351-1359. </br>
[2] Love, Michael I., Wolfgang Huber, and Simon Anders. "Moderated estimation of fold change and dispersion for RNA-seq data with DESeq2." Genome biology 15.12 (2014): 1-21.
