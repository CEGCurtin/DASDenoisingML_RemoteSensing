![Repository Avatar](./ML_denoising_logo.webp)

# DASDenoisingML_RemoteSensing
*Tools for DAS Denoising using Machine Learning*

# Overview
This folder contains Jupyter notebooks and Python utilities for denoising of DAS data using Machine Learning.

## Jupyter Notebooks
In the repository, there are several example notebooks that can be run using field and synthetic data.

## Data
The *Data* folder contains the data (field and synthetic) used in the notebooks.

## Models
The *Models* folder contains pre-trained denoising neural networks:
- *N2N_LowPowerSourceTest_30ep_v10_patch128x96_dgtarget_fliplr.h5* was trained using the Noise2Noise data on weight drop data acquired with Silixa iDAS v2 at Curtin NGL well.
- *Supervised_Otway_100Shots_250Epochs0.8_3_Nnet4.h5* was trained on semi-synthetic data.

## Utils
The *Utils* folder contains loading and processing utilities used in the Jupyter notebooks.
