# NNI-ENAS-NLP-Example
This code is for running enas_nlp (Neural Architecture Search for Text Classiﬁcation and Representation) on nni.

## Overview

This is an example of running ENAS on [NNI](https://github.com/microsoft/nni) using NNI's NAS interface. The documents about NNI's NAS interface can be found [here](https://github.com/microsoft/nni/blob/master/docs/en_US/GeneralNasInterfaces.md).

This project includes code for 10 text classification tasks task. We conduct neural architecture search experiments in TensorFlow which redevelop the open source code of ENAS for our experiments. 

## Step

> 1. INSTALL NNI

- ```python3 -m pip install --upgrade nni```

[More information](https://github.com/microsoft/nni/blob/master/docs/en_US/Installation.md)

> 2. RUN NAS

- `pip install -r requirements.txt`
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

