from analyser.MuTualSample import MuTualSample
from analyser.MuTualSampleSet import MuTualSampleSet

import glob
import numpy as np
import json

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

dream_no_ocn = "results_dream_no_ocn.json"
dream_ocn = "results_dream_ocn.json"
mutual_no_ocn = "results_mutual_no_ocn.json"
mutual_ocn = "results_mutual_ocn.json"
mutual_no_ocn_frozen = "results_mutual_no_ocn_frozen_bert.json"
mutual_ocn_frozen = "results_mutual_ocn_frozen_bert.json"

## CHANGE LINES BELOW TO SWITCH
results_file = mutual_ocn_frozen
mutual_type  = ['middle', 'high'] # high or middle (if MuTual selected)
compare_with = False # if False, no compare

## WARNING: CODE BELOW IS NOT OPTIMALLY ROBUST, BUT WORKS FOR OUR PURPOSE

files_to_check = [results_file]

if compare_with:
    files_to_check.append(compare_with)

total_samplesets = []

for file_to_check in files_to_check:
    results_handle = open("results/" + file_to_check, mode='r')
    results_data   = json.loads(results_handle.read())

    samples       = []
    correct_false = [] # prediction in case of MuTual
    is_prediction = False

    if len(results_data) == 2: # MuTual
        # we only take dev (test has no right answer annotation...)
        for type in results_data['dev']:
            if type in mutual_type: # legal type
                for sample, prediction in results_data['dev'][type].items():
                    sample_file   = "data/MuTual/" + sample[:-2] + ".txt"
                    sample_handle = open(sample_file)
                    sample_JSON   = sample_handle.read()
                    sample_object = MuTualSample(sample_JSON)
                    samples.append(sample_object)
                    correct_false.append(prediction)
        is_prediction = True
    else: # DREAM
        for context, info, _ in results_data:
            article = " ".join(context) + " " + info[0]['question'] # question appended to context
            options = info[0]['choice']
            answer  = info[0]['choice'].index(info[0]['answer'])
            sample_object = MuTualSample(article, fromJSON=False, options=options, answer=answer)
            samples.append(sample_object)
            correct_false.append((answer == info[0]['prediction']))
        
    sampleset = MuTualSampleSet(samples, correct_false, is_prediction=is_prediction)
    total_samplesets.append(sampleset)

print('Printing scores for', files_to_check)

if compare_with:
    total_samplesets[0].adjust_split(total_samplesets[1], type=2)
    #print("Format: ALL_EXCEPT_CORRECT_ONLY_SET2 / CORRECT_ONLY_SET2 (type 1)")
    print("Format: INCORRECT_BEFORE_EXCEPT_CORRECT_NOW / CORRECT_ONLY_SET2 (type 2)")
else:
    print("Format: INCORRECTLY_CLASSIFIED / CORRECTLY_CLASSIFIED")


TF_IDF_results = total_samplesets[0].TF_IDF(tokenizer=word_tokenize, stemmer=ps)

print('Num samples: %d' % len(samples))

print('TF-IDF score (avg): %.4f / %.4f' % (np.mean(TF_IDF_results[False]), np.mean(TF_IDF_results[True])))
                        
for local_stat in ['matching_token_fraction', 'context_number_count', 'context_length', 'average_sentence_context_length']:
    additional_parameters = {}
    if local_stat == 'matching_token_fraction':
        additional_parameters = {'tokenizer': word_tokenize, 'stemmer': ps}
        
    res = total_samplesets[0].local_statistic(local_stat, additional_parameters)
                        
    print('%s (avg): %.4f / %.4f' % (local_stat, np.mean(res[False]), np.mean(res[True])))
                        