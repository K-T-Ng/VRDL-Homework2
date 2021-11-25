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
1. In order to split the training dataset and validation dataset, run
   ```
   split_training_validation.py
   ```
   Modify line 37 if you want more or less validation data.\
   This program will produce ```Train_GT.json``` and ```Valid_GT.json```.
2. For the anchor boxes prior, run
    ```
    anchor_kmeans.py
    ```
    The program will output 9 anchor boxes sizes.\
    Then do the following steps by hand.\
    Dividing them by 8, 16 and 32, then fill the value into ```config.py```\
    See the comment in line 92 ~ 100 in ```anchor_kmeans.py``` and line 19 ~ 21 in ```config.py``` 
3. (Optional) Modify the hyper-parameters in ```config.py``` before training.
4. run ```train.py```.

