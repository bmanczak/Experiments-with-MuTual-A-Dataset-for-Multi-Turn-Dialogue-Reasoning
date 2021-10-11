from analyser.MuTualSample import MuTualSample
from analyser.MuTualSampleSet import MuTualSampleSet

import glob
import numpy as np
import json

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from scipy import stats

ps = PorterStemmer()

dream_no_ocn = "results_dream_no_ocn.json"
dream_ocn = "results_dream_ocn.json"
mutual_no_ocn = "results_mutual_no_ocn.json"
mutual_ocn = "results_mutual_ocn.json"
mutual_no_ocn_frozen = "results_mutual_no_ocn_frozen_bert.json"
mutual_ocn_frozen = "results_mutual_ocn_frozen_bert.json"

## CHANGE LINES BELOW TO SWITCH
results_file = mutual_no_ocn_frozen
mutual_type  = ['middle', 'high'] # high or middle (if MuTual selected)
compare_with = mutual_ocn_frozen # if False, no compare
experiment_type = 2 # 1 = analyse metrics; 2 = tf-idf accuracy bins
split_type      = 1 # see error analysis section in paper
number_of_bins  = 3

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

if experiment_type == 1:
    ##### PRINT METRICS
    
    if compare_with:
        total_samplesets[0].adjust_split(total_samplesets[1], type=split_type)
        
        if split_type == 1:
            print("Format: ALL_EXCEPT_CORRECT_ONLY_SET2 / CORRECT_ONLY_SET2 (type 1)")
        elif split_type == 2:
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
        
elif experiment_type == 2:

    # First, the bins are determined
    TF_IDF_results = total_samplesets[0].TF_IDF(tokenizer=word_tokenize, stemmer=ps)
    
    bin_probs = [0]
    bin_incrs = 1. / number_of_bins
    
    for b in range(number_of_bins - 1):
        bin_probs.append(bin_incrs * (b + 1))
        
    bin_probs.append(1)
    
    bin_edges = stats.mstats.mquantiles(TF_IDF_results['global'], bin_probs)
    
    # for preventing rounding errors
    bin_edges[0] -= 1
    bin_edges[-1] += 1
    
    # Now, we assign samples to bins and calculate accuracy scores
    
    bin_inds = np.digitize(TF_IDF_results['global'], bin_edges)
    
    bins = {}
    
    for i in range(len(total_samplesets)):
        bins[i] = {}
        
        for b in range(number_of_bins):
            bins[i][b + 1] = {'correct': 0, 'total': 0}
        
        for s, correct in enumerate(total_samplesets[i].correct_false):
            bins[i][bin_inds[s]]['total'] += 1
            
            if correct:
                bins[i][bin_inds[s]]['correct'] += 1
        
        print('Results for file %s' % files_to_check[i])
        
        for b in range(number_of_bins):
            print('Bin #%d accuracy: %.3f (%d correct out of %d)' % (b, bins[i][b+1]['correct']/bins[i][b+1]['total'], bins[i][b+1]['correct'], bins[i][b+1]['total']))
                        