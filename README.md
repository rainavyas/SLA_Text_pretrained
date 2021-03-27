# Objective
Use pretrained encoders to train a model for Spoken Language Assessment.

# Train

## Install with PyPI

pip install torch, torchvision, transformers

## Train a SLA model

An example command is given below

```console
python train.py model_name.th train_data_file.txt test_data_file.txt train_grades_file.txt test_grades_file.txt --epochs=5 --sch=3 --B=16 --seed=47 
```
