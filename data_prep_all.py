import torch
from data_prep import tokenize_text, get_spk_to_utt, get_spk_to_grade

def align(spk_to_utt1, spk_to_utt2, spk_to_utt3, spk_to_utt4, spk_to_utt5, grade_dict1, grade_dict2, grade_dict3, grade_dict4, grade_dict5, grade_dict_overall):
    grades1 = []
    grades2 = []
    grades3 = []
    grades4 = []
    grades5 = []
    grades_overall = []

    utts1 = []
    utts2 = []
    utts3 = []
    utts4 = []
    utts5 = []

    for id in spk_to_utt1:
        try:
            grades1.append(grade_dict1[id])
            grades2.append(grade_dict2[id])
            grades3.append(grade_dict3[id])
            grades4.append(grade_dict4[id])
            grades5.append(grade_dict5[id])
            grades_overall.append(grade_dict_overall[id])

            utts1.append(spk_to_utt1[id])
            utts2.append(spk_to_utt2[id])
            utts3.append(spk_to_utt3[id])
            utts4.append(spk_to_utt4[id])
            utts5.append(spk_to_utt5[id])
        except:
            print("Failed for speaker " + str(id))
        all_utts = [utts1, utts2, utts3, utts4, utts5]
        all_grades = [grades1, grades2, grades3, grades4, grades5, grades_overall]
    return all_utts, all_grades

def get_data(data_dir, grades_file):
    utt_dicts = []
    grade_dicts = []

    for part in range(1, 6):
        data_file = data_dir+'/useful_part'+str(part)+'.txt'
        spk_to_utt = get_spk_to_utt(data_file)
        grade_dict = get_spk_to_grade(grades_file, part)
        utt_dicts.append(spk_to_utt)
        grade_dicts.append(grade_dict)
    overall_grade_dict = get_spk_to_grade(grades_file, part=6)

    all_utts, all_grades = align(utt_dicts[0], utt_dicts[1], utt_dicts[2], utt_dicts[3], utt_dicts[4], grade_dicts[0], grade_dicts[1], grade_dicts[2], grade_dicts[3], grade_dicts[4], overall_grade_dict)

    all_data = []
    for j in range(5):
        ids, mask = tokenize_text(all_utts[j])
        labels = torch.FloatTensor(all_grades[j])
        data = {'input_ids':ids, 'mask':mask, 'labels':labels}
        all_data.append(data)

    return all_data, torch.FloatTensor(all_grades[-1])
