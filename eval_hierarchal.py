'''
Evaluation Approach (Always using model ensembles)
Use A1-C2 models to get grade predictions
Choose a threshold k
For all datapoints with predictions > k
Get new prediction using B2-C2 models
Caluclate MSE for all predictions
Repeat for all k to get a plot of MSE vs threshold k
'''

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from data_prep import get_data
import sys
import os
import argparse
from tools import AverageMeter, get_default_device, calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg
from models import BERTGrader
import statistics
import matplotlib.pyplot as plt
import numpy as np
from eval_ensemble import eval

def get_ensemble_preds(all_preds):
    y_sum = torch.zeros(len(all_preds[0]))
    for preds in all_preds:
        y_sum += torch.FloatTensor(preds)
    ensemble_preds = y_sum/len(all_preds)
    return ensemble_preds

def apply_hierarchal(preds_stage1, preds_stage2, thresh=4.0):
    preds = []
    for predA, predB in zip(preds_stage1, preds_stage2):
        if predA < thresh:
            preds.append(predA.item())
        else:
            preds.append(predB.item())
    return preds

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODELSA', type=str, help='trained .th models for first stage separated by space')
    commandLineParser.add_argument('MODELSB', type=str, help='trained .th models for second stage separated by space')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")

    args = commandLineParser.parse_args()
    model_pathsA = args.MODELSA
    model_pathsA = model_pathsA.split()
    model_pathsB = args.MODELSB
    model_pathsB = model_pathsB.split()
    test_data_file = args.TEST_DATA
    test_grades_files = args.TEST_GRADES
    batch_size = args.B

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_hierarchal.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load the data as tensors
    input_ids_test, mask_test, labels_test = get_data(test_data_file, test_grades_files)
    test_ds = TensorDataset(input_ids_test, mask_test, labels_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size)


    # Load the models
    modelsA = []
    for model_path in model_pathsA:
        model = BERTGrader()
        model.load_state_dict(torch.load(model_path))
        modelsA.append(model)

    modelsB = []
    for model_path in model_pathsB:
        model = BERTGrader()
        model.load_state_dict(torch.load(model_path))
        modelsB.append(model)

    targets = None
    all_predsA = []
    all_predsB = []

    for model in modelsA:
        preds, targets = eval(test_dl, model)
        all_predsA.append(preds)

    for model in modelsB:
        preds, targets = eval(test_dl, model)
        all_predsB.append(preds)

    predsA = get_ensemble_preds(all_predsA)
    predsB = get_ensemble_preds(all_predsB)

    ks = []
    rmses = []
    rmses_ref = []
    ref = calculate_mse(torch.FloatTensor(predsA), torch.FloatTensor(targets)).item()

    for k in np.linspace(0, 6, 60):
        preds = apply_hierarchal(predsA, predsB, thresh=k)
        mse = calculate_mse(torch.FloatTensor(preds), torch.FloatTensor(targets)).item()
        rmse = mse**0.5
        ks.append(k)
        rmses.append(rmse)
        rmses_ref.append(ref)

    # Plot
    filename = 'rmse_vs_k.png'
    plt.plot(ks, rmses)
    plt.plot(ks, rmses_ref)
    plt.xlabel("Threshold")
    plt.ylabel("RMSE")
    plt.savefig(filename)
    plt.clf()
