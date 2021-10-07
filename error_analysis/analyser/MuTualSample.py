import json
import re
from utils import convert_answer_to_index, convert_context_to_sentences

class MuTualSample():
    def __init__(self, article, fromJSON = True, options = None, answer = None):
        """
        Converts and stores a sample from the MuTual dataset.
        
        Input:
        
        @article  - Contains either the sample's context or the entire sample
                    structured in JSON depending on @fromJSON input parameter.
        @fromJSON - If set to True, then article is expected to be in JSON
                    and to hold the entire sample.
        @options  - List of options (if @fromJSON is False).
        @answer   - Either index of correct option (in list @options) or a
                    char (starting from ASCII 65) (if @fromJSON is False).
                    
        Output: void
        """
        
        if fromJSON:
            sample_data = json.loads(article)
            
            article = sample_data['article']
            options = [option for option in sample_data['options']]
            answer  = sample_data['answers']
            
        if not isinstance(answer, int):
            answer = ord(answer) - 65
            
        if answer > len(options):
            raise ValueError("(Parsing error) Answer exceeds no. of options")
            
        self.article = article
        self.options = options
        self.answer  = answer
        
        self.statistics = {} # statistics will only be calculated when called
        
        ## all variables below will only be filled when needed (for efficiency)
        self.article_sentences         = None # all these values #
        self.answer_tokenized_stemmed  = None # are represented  #
        self.article_tokenized_stemmed = None # as lists!        #
        
    def get_statistics(self):
        """
        Returns possibly non-exhaustive dictionary containing the statistics
        of the sample.
        
        Input: void
        
        Output:
        
        @statistics - Dictionary of all explicitly called sample statistics.
        """
        
        return self.statistics
        
    def tokenize_stem(self, tokenizer=None, stemmer=None,
                                                   answer=True, article=True):
        """
        Tokenizes and stems the correct answer and/or context. Stores the
        results so that no unnecessary computation is done when method is
        called subsequentually.
        
        Input:
        
        @tokenizer - An NLTK embodied tokenizer. If not given, then it is
                     assumed that sentences were already tokenized (by a
                     call to a previous method involving tokenization).
        @stemmer   - An NLTK embodied stemmer. If not given, then it is
                     assumed that sentences were already stemmed (by a call
                     to a previous method involving tokenization).
                     WARNING: stemmer is only valid in combination with a
                              tokenizer!
        @answer    - If set to True, the correct answer sentence will be
                     stemmed/tokenized (if not already done).
        @article   - If set to True, the context will be stemmed/tokenized
                     (if not already done).
        
        Output: void
        """
        if answer:
            if self.answer_tokenized_stemmed is None:
                if not tokenizer or not stemmer:
                    raise ValueError("(Parsing error) No means of \
                                                       tokenization/stemming")
                                                       
                # obtaining tokenized-stemmed answer
                self.answer_tokenized_stemmed = [stemmer.stem(w).lower() \
                                for w in tokenizer(self.options[self.answer])]
               
        if article:
            if self.article_tokenized_stemmed is None:
                if not tokenizer or not stemmer:
                    raise ValueError("(Parsing error) No means of \
                                                       tokenization/stemming")
                
                # obtaining tokenized-stemmed context
                self.article_tokenized_stemmed = [stemmer.stem(w).lower() \
                                             for w in tokenizer(self.article)]
        
    def get_context_length(self):
        """
        Calculates (if not stored yet) and returns the context length of the
        sample. Stores output so that recalculations are not necessary.
        
        Input: void
        
        Output:
        
        @result - Integer representing character length of context.
        """
        
        if "context_length" not in self.statistics:
            result = len(self.article)
            self.statistics['context_length'] = result
        else:
            result = self.statistics['context_length']
        
        return result
        
    def get_average_sentence_context_length(self):
        """
        Calculates (if not stored yet) and returns the context length of the
        sample. Stores output so that recalculations are not necessary.
        
        Input: void
        
        Output:
        
        @result - Integer representing average character length of a sentence
                  in the context of the sample.
        """
        
        if self.article_sentences is None:
            self.article_sentences = convert_context_to_sentences(self.article)
        
        if "average_sentence_context_length" not in self.statistics:
            result = float(sum([len(s) for s in self.article_sentences])) \
                                                 / len(self.article_sentences)
            self.statistics['average_sentence_context_length'] = result
        else:
            result = self.statistics['average_sentence_context_length']
        
        return result
        
    def get_context_number_count(self):
        """
        Finds the count of number occurrences in the context.
        
        Input: void
        
        Output:
        
        @result - Count of numbers in context
        """
        
        if "context_number_count" not in self.statistics:
            result = len(re.findall('\d+', self.article))
            self.statistics['context_number_count'] = result
        else:
            result = self.statistics['context_number_count']
            
        return result
        
    def get_matching_token_fraction(self, tokenizer = None, stemmer = None):
        """
        Calculates the fraction of stemmed tokens in the correct answer that
        overlap with the sample context.
        
        Input:
        
        @tokenizer - An NLTK embodied tokenizer. If not given, then it is
                     assumed that sentences were already tokenized (by a
                     call to a previous method involving tokenization).
        @stemmer   - An NLTK embodied stemmer. If not given, then it is
                     assumed that sentences were already stemmed (by a call
                     to a previous method involving tokenization).
                     WARNING: stemmer is only valid in combination with a
                              tokenizer!
        
        Output:
        
        @result - Frequency of token-overlap between answer sentence and
                  context.
        """
        
        self.tokenize_stem(tokenizer, stemmer) # only if not done before
        
        if "matching_token_fraction" not in self.statistics:
            answer_set  = set(self.answer_tokenized_stemmed)
            context_set = set(self.article_tokenized_stemmed)
            
            result = len(answer_set & context_set)/len(answer_set)
            self.statistics['matching_token_fraction'] = result
        else:
            result = self.statistics['matching_token_fraction']
            
        return result