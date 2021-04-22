import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from models import BERTGrader
from data_prep_all import get_data
import sys
import os
import argparse
from tools import AverageMeter, calculate_mse, calculate_pcc, calculate_less1, calculate_less05, calculate_avg
from eval_ensemble import get_ensemble_stats, get_single_stats, eval

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODELS_DIR', type=str, help='trained models directory, saved as bert_partX_seedY.th')
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
    with open('CMDs/eval_all_parts.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Load all parts data aligned
    part_data_list, labels = get_data(test_data_dir, test_grades_files)

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
            model_path = models_dir + '/bert_part'+str(part)+'_seed'+str(seed)+'.th'
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

        # Get single stats
        mse_mean, mse_std, pcc_mean, pcc_std, avg_mean, avg_std, less05_mean, less05_std, less1_mean, less1_std = get_single_stats(all_preds, targets)
        print("STATS FOR PART", part)
        print()
        print("SINGLE STATS\n")
        print("MSE: "+str(mse_mean)+" +- "+str(mse_std))
        print("PCC: "+str(pcc_mean)+" +- "+str(pcc_std))
        print("AVG: "+str(avg_mean)+" +- "+str(avg_std))
        print("LESS05: "+str(less05_mean)+" +- "+str(less05_std))
        print("LESS1: "+str(less1_mean)+" +- "+str(less1_std))

        # Get ensemble stats
        mse, pcc, avg, less05, less1 = get_ensemble_stats(all_preds, targets)
        print()
        print("ENSEMBLE STATS\n")
        print("MSE: ", mse)
        print("PCC: ", pcc)
        print("AVG: ", avg)
        print("LESS05: ", less05)
        print("LESS1: ", less1)
        print()
        print()

    # Get overall preds
    part_preds = torch.stack(part_preds, dim=0)
    avg_preds = torch.mean(part_preds, dim=0)

    mse = calculate_mse(avg_preds, labels).item()
    pcc = calculate_pcc(avg_preds, labels).item()
    avg = calculate_avg(avg_preds).item()
    less05 = calculate_less05(avg_preds, labels)
    less1 = calculate_less1(avg_preds, labels)

    print()
    print("OVERALL STATS")
    print()
    print("MSE: ", mse)
    print("PCC: ", pcc)
    print("AVG: ", avg)
    print("LESS05: ", less05)
    print("LESS1: ", less1)
