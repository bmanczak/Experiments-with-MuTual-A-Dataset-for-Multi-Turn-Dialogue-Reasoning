from analyser.MuTualSample import MuTualSample
from analyser.MuTualSampleSet import MuTualSampleSet

import glob
import numpy as np

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps = PorterStemmer()

## Note: read through all files in class/ and the utils.py file for
##       more information! They have been thoroughly commented.


## First, we load a few samples by iterating through the samples/-dir

samples = []

for sample_file in glob.glob("samples/train_*.txt"):
    sample_handle = open(sample_file, mode='r')
    
    # we create a MuTualSample() object from the JSON data and append
    sample_JSON = sample_handle.read()
    sample_object = MuTualSample(sample_JSON)
    samples.append(sample_object)
    
## Now, let's calculate the average sentence length (within the context) of
## all our samples. Note that we use a second object here, which hold all our
## individual-sample objects: MuTualSampleSet()

sampleset1 = MuTualSampleSet(samples[:-1])

print('Line 34', sampleset1.local_statistic('average_sentence_context_length'))

## Another example: sampleset1.local_statistic('matching_token_fraction', {'tokenizer': word_tokenize, 'stemmer': ps})

## Of course, we can add samples later on etc. etc.

sampleset1.add_sample(samples[-1])

print('Line 42', sampleset1.local_statistic('average_sentence_context_length'))

## Now let's assume that these samples were already classified by our model;
## for better analysis, we can pass the classification results (True/False)
## as well. For now, we just classify them randomly.

samples_correct_false = np.random.choice([True, False], len(samples))

sampleset2 = MuTualSampleSet(samples, samples_correct_false)

## The results for the same statistic now look like:

print('Line 54', sampleset2.local_statistic('average_sentence_context_length'))

## But much cooler things are possible:

ps = PorterStemmer()
TF_IDF_results = sampleset2.TF_IDF(tokenizer=word_tokenize, stemmer=ps)

print('Line 61', 'Average TF-IDF score for falsely/correctly classified samples: %.4f / %.4f' \
                        % (np.mean(TF_IDF_results[False]), np.mean(TF_IDF_results[True])))