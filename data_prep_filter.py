import torch
import torch.nn as nn
from transformers import BertTokenizer
import json
from data_prep import get_spk_to_utt, get_spk_to_grade, tokenize_text


def align(spk_to_utt, grade_dict, grade_lim):
    grades = []
    utts = []
    for id in spk_to_utt:
        try:
            if grade_dict[id]>=grade_lim:
                grades.append(grade_dict[id])
                utts.append(spk_to_utt[id])
        except:
            print("Falied for speaker " + str(id))
    return utts, grades


def get_data(data_file, grades_file, grade_lim, part=3):
    '''
    Prepare data as tensors
    part=6 gives overall grades
    '''
    spk_to_utt = get_spk_to_utt(data_file)
    grade_dict = get_spk_to_grade(grades_file, part)
    utts, grades = align(spk_to_utt, grade_dict, grade_lim)
    ids, mask = tokenize_text(utts)
    labels = torch.FloatTensor(grades)

    return ids, mask, labels
