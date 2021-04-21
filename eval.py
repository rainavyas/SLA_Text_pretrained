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

def eval(val_loader, model, device):
    '''
    Run evaluation
    '''
    mses = AverageMeter()
    pccs = AverageMeter()
    less05s = AverageMeter()
    less1s = AverageMeter()
    avgs = AverageMeter()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (id, mask, target) in enumerate(val_loader):

            id = id.to(device)
            mask = mask.to(device)
            target = target.to(device)

            # Forward pass
            pred = model(id, mask)

            # update all stats
            mses.update(calculate_mse(pred, target).item(), id.size(0))
            pccs.update(calculate_pcc(pred, target).item(), id.size(0))
            less05s.update(calculate_less05(pred, target), id.size(0))
            less1s.update(calculate_less1(pred, target), id.size(0))
            avgs.update(calculate_avg(pred).item(), id.size(0))

    return mses.avg, pccs.avg, less05s.avg, less1s.avg, avgs.avg

if __name__ == "__main__":

    # Get command line arguments
    commandLineParser = argparse.ArgumentParser()
    commandLineParser.add_argument('MODEL', type=str, help='trained .th model')
    commandLineParser.add_argument('TEST_DATA', type=str, help='prepped test data file')
    commandLineParser.add_argument('TEST_GRADES', type=str, help='test data grades')
    commandLineParser.add_argument('--B', type=int, default=16, help="Specify batch size")
    commandLineParser.add_argument('--part', type=int, default=3, help="Specify part of exam")

    args = commandLineParser.parse_args()
    model_path = args.MODEL
    test_data_file = args.TEST_DATA
    test_grades_files = args.TEST_GRADES
    batch_size = args.B
    part = args.part

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/eval.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    device = get_default_device()

    # Load the data as tensors
    input_ids_test, mask_test, labels_test = get_data(test_data_file, test_grades_files, part=part)
    test_ds = TensorDataset(input_ids_test, mask_test, labels_test)
    test_dl = DataLoader(test_ds, batch_size=batch_size)

    # Load the model
    model = BERTGrader()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    # Get all stats
    mse, pcc, less05, less1, avg = eval(test_dl, model, device)

    print("STATS for "+model_path)
    print("MSE: ", mse)
    print("PCC: ", pcc)
    print("Less 0.5: ", less05)
    print("Less 1.0: ", less1)
    print("Avg: ", avg)
