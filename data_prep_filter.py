import torch
import torch.nn as nn
from transformers import BertTokenizer
import json

def get_prompts_dict(file_loc):
    with open(file_loc, 'r') as f:
            lines = f.readlines()
    lines = [line.rstrip('\n') for line in lines]

    key = "dummy"
    val = "dummy"
    prompts_dict = {}
    for line in lines[1:]:
        if line[0] == '"':
            prompts_dict[key]=val
            key = line[1:-12]
            val = ''
        else:
            val+= line + ' '
    return prompts_dict


def get_spk_to_utt(data_file):
    # Load the data
    with open(data_file, 'r') as f:
        utterances = json.loads(f.read())

    # Convert json output from unicode to string
    utterances = [[str(item[0]), str(item[1])] for item in utterances]

    # Concatentate utterances of a speaker
    spk_to_utt = {}
    for item in utterances:
        fileName = item[0]
        speakerid = fileName
        sentence = item[1]

        if speakerid not in spk_to_utt:
            spk_to_utt[speakerid] =  sentence
        else:
            spk_to_utt[speakerid] =spk_to_utt[speakerid] + ' ' + sentence
    return spk_to_utt

def get_spk_to_grade(grades_file, part=3):
    grade_dict = {}

    lines = [line.rstrip('\n') for line in open(grades_file)]
    for line in lines[1:]:
        speaker_id = line[:12]
        grade_overall = line[-3:]
        grade1 = line[-23:-20]
        grade2 = line[-19:-16]
        grade3 = line[-15:-12]
        grade4 = line[-11:-8]
        grade5 = line[-7:-4]
        grades = [grade1, grade2, grade3, grade4, grade5, grade_overall]

        grade = float(grades[part-1])
        grade_dict[speaker_id] = grade
    return grade_dict

def align(spk_to_utt, grade_dict, prompts_dict, grade_lim):
    grades = []
    utts = []
    prompts = []
    for id in spk_to_utt:
        grade_id = id[:12]
        prompt_id = str(id[:7]+id[22:24])
        try:
            if grade_dict[grade_id] >= grade_lim:
                grades.append(grade_dict[grade_id])
                prompts.append(prompts_dict[prompt_id])
                utts.append(spk_to_utt[id])
        except:
            print("Falied for speaker " + str(id))
    return utts, prompts, grades

def tokenize_text(utts, prompts):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encoded_inputs = tokenizer(utts, prompts, padding=True, truncation=True, return_tensors="pt")
    input_ids = encoded_inputs['input_ids']
    mask = encoded_inputs['attention_mask']
    token_ids = encoded_inputs['token_type_ids']
    return input_ids, mask, token_ids


def get_data(data_file, grades_file, prompts_mlf, grade_lim):
    '''
    Prepare data as tensors
    '''
    spk_to_utt = get_spk_to_utt(data_file)
    grade_dict = get_spk_to_grade(grades_file)
    prompts_dict = get_prompts_dict(prompts_mlf)
    utts, prompts, grades = align(spk_to_utt, grade_dict, prompts_dict, grade_lim)
    input_ids, mask, token_ids = tokenize_text(utts, prompts)
    labels = torch.FloatTensor(grades)

    return input_ids, mask, token_ids, labels
