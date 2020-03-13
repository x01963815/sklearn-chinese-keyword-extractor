#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import re
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted, check_array


class KeywordExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, min_keyword_len=2, 
                 n_keyword=5, ngram_range=(0,10),
                 regex_pattern=r'[a-zA-Z0-9~!@#$%^&*()_&#43;\-\+=,.'
                           '<>?\|/ \[\]\-"\　０-\９（）？【】]'):
        '''
        '''
        self.min_keyword_len = min_keyword_len
        self.n_keyword = n_keyword
        self.regex_pattern = regex_pattern
        self.ngram_range = ngram_range
        
    
    def fit(self, X, y=None):
        '''
        '''
        
        corpus = list(map(self._string_modify, set(X)))
        
        vectorizer = CountVectorizer(ngram_range=self.ngram_range, 
                                     analyzer='char')
        
        corpus_vec = vectorizer.fit_transform(corpus)
               
        self.keywords_components_ = OrderedDict(zip(
            vectorizer.get_feature_names(),
            np.ravel(corpus_vec.sum(axis=0))
        ))
        # a lot of useless emtpy string, set to 0
        self.keywords_components_[''] = 0  
        
        return self
    
    def transform(self, X):
        
        # Check is fit had been called
        check_is_fitted(self, ['keywords_components_'])
        
        get_keyword_label = lambda s: self._parse_string(s)[0]
        
        # Input is str
        if isinstance(X, str):
            return get_keyword_label(X)
        
        # Input validation, only accept 1 dim and str
        X = check_array(X, ensure_2d=False, dtype=str)
        labels = np.stack(list(map(get_keyword_label, X)))
        
        return labels
    
    def transform_proba(self, X):
        
        # Check is fit had been called
        check_is_fitted(self, ['keywords_components_'])
        
        get_gainvalue = lambda s: self._parse_string(s)[1]
        
        # Input is str
        if isinstance(X, str):
            return get_gainvalue(X)
        
        # Input validation, only accept 1 dim and str
        X = check_array(X, ensure_2d=False, dtype=str)
        gainvalues = np.stack(list(map(get_gainvalue, X)))
        
        return gainvalues
    
    def fit_transform(self, X, y=None):
        self.fit(X) 
        return self.transform(X)
    
    def _string_modify(self, string):     
        modified = re.sub(self.regex_pattern , '', string)
        
        return modified
    
    
    def _parse_string(self, string):
        
        string = self._string_modify(string)
        
        # Pascal Structure
        len_str = max(len(string), 1) # min 1, prevent empty string
        start_position = np.tril(np.indices((len_str,len_str))[1])    
        end_position = sum(map(
            lambda x: np.eye(len_str, k=x, dtype=int) * (len_str + x), 
            range(- len_str + 1, 1)))   
        slices = np.stack([start_position, end_position], axis=2)
        
        keywords = np.apply_along_axis(
            func1d=lambda x: string[x[0]:x[1]], 
            axis=2, 
            arr=slices)
        
        # Vectorize Numpy function
        get_value_vf = np.vectorize(
            lambda x: self.keywords_components_.get(x, 0))     
        
        keywords_val = get_value_vf(keywords)
        
        left_parents_val = np.pad(keywords_val, 
                                  pad_width=(1,0), 
                                  mode='constant', 
                                  constant_values=0)[:-1, :-1]
        
        right_parents_val = np.pad(keywords_val, 
                                   pad_width=(1,0), 
                                   mode='constant', 
                                   constant_values=0)[:-1, 1:]
        
        gain_value = keywords_val - left_parents_val - right_parents_val
        
        filter_mask = (
            (gain_value > 0) 
            & (np.char.str_len(keywords) >= self.min_keyword_len))
        
        keywords_fltr = keywords[filter_mask]
        gain_value_fltr = gain_value[filter_mask]
        
        # Sort by gain_value
        dtype = [('keyword', 'U32'), ('gain_value', int)]
        data = np.array(list(zip(keywords_fltr, gain_value_fltr)),
                        dtype=dtype)
        data = np.sort(data, order='gain_value')[::-1][0:self.n_keyword]
        
        # Make sure that output has same shape
        keyword_result = np.pad(data['keyword'], 
                                (0, self.n_keyword - data.shape[0]),
                                constant_values='')
        gain_value_result = np.pad(data['gain_value'], 
                                   (0, self.n_keyword - data.shape[0]), 
                                   constant_values=0)        
        
        
        return keyword_result, gain_value_result