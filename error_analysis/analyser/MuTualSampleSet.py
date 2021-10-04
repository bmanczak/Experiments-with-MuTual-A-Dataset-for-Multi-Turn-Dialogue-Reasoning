from sklearn.feature_extraction.text import TfidfVectorizer

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
    
    def __init__(self, samples = [], correct_false = None):
        """
        Initializes a list of held samples.
        
        Input:
        
        @samples       - A list of MuTualSample() objects. If null, an empty
                         list will be used to initialize the datastructure.
        @correct_false - An optional list containing boolean entries
                         corresponding to entries in @samples.
                   
        Output: void
        """
        
        self.samples = samples
        
        if correct_false is not None:
            if len(correct_false) != len(samples):
                raise ValueError("(Parsing error) Sample and correct/false \
                                                        counts do not add up")
            
            self.correct_false = correct_false
        else:
            self.correct_false = None
        
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
        
    def TF_IDF(self, type = 'context', tokenizer = None, stemmer = None):
        """
        Calculates the TF_IDF score for every listed sample, based on the
        whole population of samples.
        
        Input:
        
        @type      - The type of text to compute TF-IDFs for. Either:
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
        if type not in ['context', 'answer']:
            raise ValueError("(Parsing error) Wrong type TF-IDF corpus data")
            
        vectorizer = TfidfVectorizer()
        corpus     = []
        
        for sample in self.samples:
            sample.tokenize_stem(tokenizer, stemmer) # only if not done before
            
            if type == 'context':
                corpus.append(" ".join(sample.article_tokenized_stemmed))
            elif type == 'answer':
                corpus.append(" ".join(sample.answer_tokenized_stemmed))
                
        matrix = vectorizer.fit_transform(corpus)
        
        results = {'global': []}
        
        if self.correct_false is not None:
            results[True]  = []
            results[False] = []
        
        for i, result in enumerate(matrix.sum(axis=1)):
            results['global'].append(result)
            
            if self.correct_false is not None:
                results[self.correct_false[i]].append(result)
        
        return results