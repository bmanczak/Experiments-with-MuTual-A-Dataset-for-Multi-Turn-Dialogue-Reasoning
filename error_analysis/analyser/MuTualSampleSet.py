from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class MuTualSampleSet():
    """
    This class holds a list of MuTualSample() objects and calculates global
    statistics based on all fed samples. These statistics can in turn be used
    to evaluate a single MuTualSample()'s characteristics. Statistics can be
    calculated over the whole population in general, or with respect to
    (in)correctly classified samples. Some of the statistics are defined with
    regard to a single sample (such as context lengths) while some are defined
    relative to the whole population of samples (such as TF-IDF). The latter
    statistics are contained in this class, while the former are contained in
    MuTualSample().
    """
    
    def __init__(self, samples = [], correct_false = None, is_prediction = False):
        """
        Initializes a list of held samples.
        
        Input:
        
        @samples       - A list of MuTualSample() objects. If null, an empty
                         list will be used to initialize the datastructure.
        @correct_false - An optional list containing boolean entries
                         corresponding to entries in @samples.
        @is_prediction - If True, @correct_false is not a list of booleans, but
                         a list of prediction indices (that will be converted
                         to booleans).
                   
        Output: void
        """
        
        self.samples = samples
        
        if correct_false is not None:
            if len(correct_false) != len(samples):
                raise ValueError("(Parsing error) Sample and correct/false \
                                                        counts do not add up")
            
            if is_prediction:
                correct_false_converted = []
                
                for i, sample in enumerate(samples):
                    if sample.answer == correct_false[i]:
                        correct_false_converted.append(True)
                    else:
                        correct_false_converted.append(False)
                
                self.correct_false = correct_false_converted
            else:
                self.correct_false = correct_false
        else:
            self.correct_false = None
        
    def adjust_split(self, sampleset, type=1):
        """
        Changes correct_false by setting as correct only the samples that
        were correctly classified only in a given second sampleset (in
        contrast to current sample set).
        
        Input:
        
        @sampleset - A MuTualSampleSet() object
        @type      - Either:
                        1 => Compare all samples vs samples that are correct
                             ONLY in @sampleset
                        2 => Compare false samples in current sampleset except
                             those that are correct in @sampleset versus these
                             samples that are correct in @sampleset.
        
        Output: void
        """
        
        #self.correct_false = sampleset.correct_false - self.correct_false
        
        false_before_now_correct = 0
        correct_before_now_false = 0
        
        delete_from_sampleset = []
        
        for i, x in enumerate(self.correct_false):
            y = sampleset.correct_false[i]
            
            if x == y:
                if type == 1:
                    self.correct_false[i] = False
                elif type == 2:
                    if not x:
                        self.correct_false[i] = False
                    else:
                        delete_from_sampleset.append(i)
            elif x:
                correct_before_now_false += 1
                if type == 1:
                    self.correct_false[i] = False
                elif type == 2:
                    delete_from_sampleset.append(i)
            else:
                false_before_now_correct += 1
                self.correct_false[i] = True
        
        if delete_from_sampleset:
            delete_from_sampleset.reverse()
            
            for i in delete_from_sampleset:
                del self.samples[i]
                del self.correct_false[i]
        
        print('%d were false before, now true; %d were correct before, now false' % (false_before_now_correct, correct_before_now_false))
        
    def add_sample(self, sample, correct_false = None):
        """
        Adds a sample to the list of currently held samples.
        
        Input:
        
        @sample        - A MutualSample() object.
        @correct_false - Boolean conveying whether the sample was (in)correctly
                         classified. Required if a correct_false list was given
                         upon initialisation.
        
        Output: void
        """
        
        self.samples.append(sample)
        
        if self.correct_false is not None and correct_false is None:
            raise ValueError("(Parsing error) Correct/false is required")
        elif correct_false is not None:
            self.correct_false.append(correct_false)
        
    def clear(self):
        """
        Clears the list of samples (list will become empty).
        
        Input: void
        
        Output: void
        """
        
        self.samples = []
        
        if self.correct_false is not None:
            self.correct_false = []
        
    def local_statistic(self, statistic, additional_parameters = {}):
        """
        Calculates a given statistic for every sample in the set, for which
        the calculation only depends on the sample itself.
        
        Input:
        
        @statistic             - The statistic to be computed. Should exist
                                 as a method name affixed with "get_" in
                                 MuTualSample(). Currently either one out of:
                                   => context_length
                                   => average_sentence_context_length
                                   => context_number_count
                                   => matching_token_fraction
                                 See the respective function definitions for
                                 more information on the statistics.
        @additional_parameters - Additional parameters that are required to
                                 compute @statistic. See the method definition
                                 in MuTualSample().get_@statistic.
                                 
        Output:
        
        @results - Dictionary structured as:
                     ['global'] => Results for every sample in the same order
                                   as the list of samples upon initialisation
                                   (including possible later add-ons)
                     [True]     => If correct/false classification annotations
                                   are given, this list contains the values
                                   for all True samples in the order with wich
                                   they occur in the list of samples.
                     [False]    => As above, for False(ly) classified samples.
        """
        
        results = {'global': []}
        
        if self.correct_false is not None:
            results[True]  = []
            results[False] = []
            
        for i, sample in enumerate(self.samples):
            result = getattr(sample, "get_" + statistic)(**additional_parameters)
            results['global'].append(result)
            
            if self.correct_false is not None:
                results[self.correct_false[i]].append(result)
                
        return results
        
    def TF_IDF(self, type_c = 'context', tokenizer = None, stemmer = None):
        """
        Calculates the TF_IDF score for every listed sample, based on the
        whole population of samples.
        
        Input:
        
        @type_c    - The type of text to compute TF-IDFs for. Either:
                       => context
                       => answer
        @tokenizer - An NLTK embodied tokenizer. If not given, then it is
                     assumed that sentences were already tokenized (by a
                     call to a previous method involving tokenization).
        @stemmer   - An NLTK embodied stemmer. If not given, then it is
                     assumed that sentences were already stemmed (by a call
                     to a previous method involving tokenization).
                     WARNING: stemmer is only valid in combination with a
                              tokenizer!
                                 
        Output:
        
        @results - Dictionary structured as:
                     ['global'] => Results for every sample in the same order
                                   as the list of samples upon initialisation
                                   (including possible later add-ons)
                     [True]     => If correct/false classification annotations
                                   are given, this list contains the TF-IDFs
                                   for all True samples in the order with wich
                                   they occur in the list of samples.
                     [False]    => As above, for False(ly) classified samples.
        """
        if type_c not in ['context', 'answer']:
            raise ValueError("(Parsing error) Wrong type TF-IDF corpus data")
            
        vectorizer = TfidfVectorizer()
        corpus     = []
        
        for sample in self.samples:
            sample.tokenize_stem(tokenizer, stemmer) # only if not done before
            
            if type_c == 'context':
                corpus.append(" ".join(sample.article_tokenized_stemmed))
            elif type_c == 'answer':
                corpus.append(" ".join(sample.answer_tokenized_stemmed))
                
        matrix = vectorizer.fit_transform(corpus)
        
        results = {'global': []}
        
        if self.correct_false is not None:
            results[True]  = []
            results[False] = []
        
        for i, result in enumerate(matrix.sum(axis=1)):
            result = result.item(0)
            results['global'].append(result)
            
            if self.correct_false is not None:
                results[self.correct_false[i]].append(result)
        
        results['global'] = np.array(results['global'])
        
        return results