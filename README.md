# Sign Language Recognition with CorrNet

## Quick Start
A Jupyter notebook `train_corrnet.ipynb` is provided that automates all the setup and training steps described below. This is the recommended way to get started quickly. You will still have to manually download the data, which is not included in this repository. 

NOTE: The dataset is a 53GB download, so it is recommended to use Google Colab or similar.

## Manual Setup Instructions

### Prerequisites
1. Python 3.11
2. GPU access (Google Colab or similar)

### Installation
Install required dependencies: pip install -r requirements.txt
    
### Data Preparation
Create a folder called 'data' if it is not in the root directory, and download and unzip RWTH-PHOENIX Weather 2014, which can be found https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/

### Preprocessing
Run the preprocessing script: python preprocessing_pipeline.py

### Training 
Run the training script: python train_corrnet.py


