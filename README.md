# VRDL-Homework2

## Requirements
The following packages are used in this homework
```
torch==1.9.0+cu111
torchvision==0.10.0+cu111
numpy==1.19.3
scipy==1.6.0
matplotlib==3.3.3
Pillow==7.2.0
```

## Folder structure
    .
    ├──dataset
       ├──DataInfo
          ├──answer.zip
          ├──digitStruct.mat  # put digitStruct here
          ├──Train_GT.json
          ├──Valid_GT.json
          ├──Valid_PD.json
       ├──Testing
          ├──117.png          # put all testing images here
       ├──Training
          ├──1.png            # put all training images here
    ├──weights
       ├──29.pth
    ├──function
    ├──models
    ├──split_training_validation.py
    ├──anchor_kmeans.py
    ├──config.py
    ├──train.py
    ├──validation.py
    ├──inferece.py

## Training code
There are a few steps for training
1. 
