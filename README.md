# RI-Loss: A Learnable Residual-Informed Loss for Time Series Forecasting(RI-Loss)
This is the official open-source repository for our paper.

## Required Packages
* pytorch==2.1.0  
* tqdm==4.66.1  
* python==3.8.18  
* numpy==1.24.3  
* matplotlib==3.7.5

## Usage
1.Create ./data directory and place dataset files in ./data directory.

2.Train the model and evaluate. We provide the experiment scripts of all backbones under the folder ```./scripts/```. You can reproduce the results using the following commands.

```
cd Dlinear

# Use only MSE loss to train the model
bash ./scripts/EXP-LongForecasting/Linear/etth1.sh  

# Use only RI-Loss to train the model
# In the exp_main.py file, comment out the MSE loss (lines 87 and 139) and uncomment the RI-Loss (lines 88, 89, and 140).
bash ./scripts/EXP-LongForecasting/Linear/etth1.sh
```

## Acknowledgements
We appreciate the following GitHub repos a lot for their valuable code and efforts.

* Dlinear: https://github.com/cure-lab/LTSF-Linear  
* Informer: https://github.com/zhouhaoyi/Informer2020  
* Autoformer: https://github.com/thuml/Autoformer  
* iTransformer: https://github.com/thuml/iTransformer  
* RAFT: https://github.com/archon159/RAFT  



