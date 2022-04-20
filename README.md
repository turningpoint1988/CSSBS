# CSSBSs
Computational prediction and characterization of cell-type-specific and shared binding sites (CSSBSs). 

<p align="center"> 
<img src=https://github.com/turningpoint1988/CSSBSs/blob/main/flowchart.jpg>
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


## Differential Binding sites Preparation
We All cell-type-specific and shared binding peaks have been prepared well which can be found in the directory 'GK'

```
python bed2signal.py -d <> -n <> -s <>
```

| Arguments   | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| -d          | The path of datasets, e.g. /your_path/FCNsignal/HeLa-S3/CTCF   |
| -n          | The name of the specified dataset, e.g. CTCF                   |
| -s          | Random seed (default is 666)                                   |


## Model Training

Train FCNsignal models on specified datasets:

```
python run_signal.py -d <> -n <> -g <> -s <> -b <> -e <> -c <>
```

| Arguments  | Description                                                                      |
| ---------- | -------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data     |
| -n         | The name of the specified dataset, e.g. CTCF                                     |
| -g         | The GPU device id (default is 0)                                                 |
| -s         | Random seed                                                                      |
| -b         | The number of sequences in a batch size (default is 500)                         |
| -e         | The epoch of training steps (default is 50)                                      |
| -c         | The path for storing models, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF       |

### Output

Trained models for FCNsignal on the specified datasets. For example, A trained model can be found at `/your_path/FCNsignal/models/HeLa-S3/CTCF/model_best.pth`.

## Model Classification

Test FCNsignal on the specified test data:

```
python test_signal.py -d <> -n <> -g <> -c <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|

### Output

Generate `record.txt` indicating the mean squared error (MSE), the pearson correlation coefficient (Pearsonr), the area under the receiver operating characteristic curve (AUC) and the area under the precision-recall curve (PRAUC) of the trained model in predicting binding signals on the test data.

## Motif Prediction

Motif prediction on the specified test data:

```
python motif_prediction.py -d <> -n <> -g <> -t <> -c <> -o <>
```

| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -d         | The path of a specified dataset, e.g. /your_path/FCNsignal/HeLa-S3/CTCF/data                |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -t         | The threshold value (default is 0.3)                                                        |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|
| -o         | The path of storing motif files, e.g. /your_path/FCNsignal/motifs/HeLa-S3/CTCF              |

### Output

Generate motif files in MEME format, which are subsequently used by TOMTOM.


## Locating TFBSs

Locating potential binding regions on inputs of arbitrary length:

```
python TFBS_locating.py -i <> -n <> -g <> -t <> -w <> -c <>
```
| Arguments  | Description                                                                                 |
| ---------- | ------------------------------------------------------------------------------------------- |
| -i         | The input file in bed format, e.g. /your_path/FCNsignal/input.bed                           |
| -n         | The name of the specified dataset, e.g. CTCF                                                |
| -g         | The GPU device id (default is 0)                                                            |
| -t         | The threshold value to determine the binding regions (default is 1.5)                       |
| -w         | The length of the binding regions (default is 60)                                           |
| -c         | The trained model path of a specified dataset, e.g. /your_path/FCNsignal/models/HeLa-S3/CTCF|

### Output

The outputs include the base-resolution prediction of inputs and the position of potential binding regions in the genome (bed format). <br/>
We also provide the line plots of the above base-resolutiion prediction. For example:

<p align="center"> 
<img src=https://github.com/turningpoint1988/FCNsignal/blob/main/output.jpg>
</p>
