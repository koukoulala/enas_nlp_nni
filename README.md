# NNI-ENAS-NLP-Example
This code is for running enas_nlp (Neural Architecture Search for Text Classiﬁcation and Representation) on nni.

## Overview

This is an example of running ENAS on [NNI](https://github.com/microsoft/nni) using NNI's NAS interface. The documents about NNI's NAS interface can be found [here](https://github.com/microsoft/nni/blob/master/docs/en_US/GeneralNasInterfaces.md).

This project includes code for 10 text classification tasks task. We conduct neural architecture search experiments in TensorFlow which redevelop the open source code of ENAS for our experiments. 

## Step

> RUN NAS

- `pip install -r requirements.txt`
- `export CUDA_VISIBLE_DEVICES=0`
- `cd NAS && nnictl create --config config.yml`

## Datasets

These datasets include:

    SST
    AG’s News
    Sogou News
    DBPedia
    Yelp Review Polarity
    Yelp Review Full
    Yahoo! Answers
    Amazon Review Full
    Amazon Review Polarity

For text classification task, we have provided SST dataset, and the remain datasets can be downloaded from https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M

SST is short for Stanford Sentiment Treebank which is a commonly used dataset for sentiment classification. 
There are about 12 thousands reviews in it and each review is labeled to one of the five sentiment classes. 
There is another version of the dataset, SST-Binary, which has only two classes representing positive/negative while the neutral samples are discarded.

## Experiments

In our experiments, we perform 12-layer neural architecture search on SST dataset and evaluate the derived architectures on both SST and SST-Binary datasets. 
We follow the pre-defined train/validation/test split of the datasets\footnote{https://nlp.stanford.edu/sentiment/code.html}. 
The word embedding vectors are initialized by pre-trained GloVe (glove.840B.300d\footnote{https://nlp.stanford.edu/projects/glove/}) and fine-tuned during training. 
We set batch size as 128, hidden unit dimension for each layer as 32.

During search process (in NNI environment), the test result of SST is about 0.44.

model | SST | SST-B |
:----: | :----: |:----: |
ARC-I | 51.77 | 89.94 |
ARC-II | 52.51 | 88.92 |
ARC-III | 52.79 | 89.27 |
JOINT ARC | 53.44 | 90.23 |

