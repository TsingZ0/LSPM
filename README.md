# LSPM
This is the implementation for the paper: Online Personalized Next-Item Recommendation via Long Short Term Preference Learning

## Environments
- Python=3.5
- Tensorflow=1.8.0
- numpy=1.14.2
- pandas=0.24.1

## Datasets
The raw data are the same as [TLSAN](https://github.com/TsingZ0/TLSAN).

## How to run the codes
Build dataset:
```
python3 build_dataset.py
```
Train and evaluate the model:
```
python3 train.py
```
