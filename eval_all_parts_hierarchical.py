import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import BERTGrader
from data_prep_all import get_data
import sys
import os
import argparse
from tools import calculate_mse
from eval_ensemble import eval
from eval_hierarchal import apply_hierarchal, apply_hierarchal_ref
import matplotlib.pyplot as plt
import numpy as np

def get_all_parts_avg_preds(part_data_list, models_dir, filter=False):
    part_preds = []
    for part in range(1,6):

        # Specify the data as tensors
        part_data = part_data_list[part-1]
        input_ids_test = part_data['input_ids']
        mask_test = part_data['mask']
        labels_test = part_data['labels']

        test_ds = TensorDataset(input_ids_test, mask_test, labels_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size)

        # Load the models
        models = []
        for seed in range(1,6):

            if not filter:
                model_path = models_dir + '/bert_part'+str(part)+'_seed'+str(seed)+'.th'
            else:
                model_path = models_dir + '/bert_part'+str(part)+'_seed'+str(seed)+'_filter4.0.th'
            model = BERTGrader()
            model.load_state_dict(torch.load(model_path))
            models.append(model)

        targets = None
        all_preds = []

        for model in models:
            preds, targets = eval(test_dl, model)
            all_preds.append(preds)


        y_sum = torch.zeros(len(all_preds[0]))
        for preds in all_preds:
            y_sum += torch.FloatTensor(preds)
        ensemble_preds = y_sum/len(all_preds)

        part_preds.append(ensemble_preds)

    # Get overall preds
    part_preds = torch.stack(part_preds, dim=0)
    avg_preds = torch.mean(part_preds, dim=0)

    return avg_preds

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODELS_DIR', type=str, help='trained models directory, saved as bert_partX_seedY.th or bert_partX_seedY_filter4.0.th')
    commandLineParser.add_argument('TEST_DATA_DIR', type=str, help='prepped test data file directory, saved as /useful_partX.txt')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")

    args = commandLineParser.parse_args()
    models_dir = args.MODELS_DIR
    test_data_dir = args.TEST_DATA_DIR
    test_grades_files = args.TEST_GRADES
    batch_size = args.B

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval_all_parts_hierarchical.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load all parts data aligned
    part_data_list, labels = get_data(test_data_dir, test_grades_files)

    # Get first and second stage predictions
    stage1_preds = get_all_parts_avg_preds(part_data_list, models_dir, filter=False)
    stage2_preds = get_all_parts_avg_preds(part_data_list, models_dir, filter=True)


    # Create hierarchical plot
    ks = []
    rmses = []
    rmses_ref = []
    rmses_baseline = []
    baseline = calculate_mse(stage1_preds, labels).item()
    baseline = ref ** 0.5

    for k in np.linspace(0, 6, 60):
        preds = apply_hierarchal(stage1_preds, stage2_preds, thresh=k)
        mse = calculate_mse(torch.FloatTensor(preds), labels).item()
        rmse = mse**0.5
        preds_ref = apply_hierarchal_ref(stage1_preds, stage2_preds, labels, thresh=k)
        mse_ref = calculate_mse(torch.FloatTensor(preds_ref), labels).item()
        rmse_ref = mse_ref**0.5
        ks.append(k)
        rmses.append(rmse)
        rmses_baseline.append(baseline)
        rmses_ref.append(rmse_ref)

    # Plot
    filename = 'all_parts_rmse_vs_k.png'
    plt.plot(ks, rmses_baseline, label="Baseline")
    plt.plot(ks, rmses, label="Hierarchical")
    plt.plot(ks, rmse_ref, label="Reference")
    plt.xlabel("Threshold")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(filename)
    plt.clf()
