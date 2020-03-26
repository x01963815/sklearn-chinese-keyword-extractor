#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import pandas as pd
import re
import six
from string import ascii_letters
from collections import OrderedDict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted, check_array


class KeywordExtractor(BaseEstimator, TransformerMixin):
    ''' Chinese Keyword Extractor
    
    The model will process a brute-force search algorithm to find the 
    most repeated substrings in a string. The substrings are defined as
    keywords. The frequency of a keyword will be calibrated by its 
    parental strings.


    Parameters
    ----------
    n_keyword: int, default=5
        Number of keywords to keep.
        If number of extracted keywords is less than n_keyword, model 
        will pad with "" string to make sure the same shape.
    
    min_ch_keyword_len: int, default=2
        Filter the min length of Chinese keyword. Model would drop the 
        keyword with length less that min_ch_keyword_len.
    
    min_en_keyword_len: int, default=3
        Filter the min length of English keyword. Model would drop the 
        keyword with length less that min_en_keyword_len.
    
    ngram_range: tuple of int, default=(0,10)
        CountVectorizer parameter. The value of min_ch_keyword_len and
        min_en_keyword_len should be in ngram_range.
    
    enable_english: bool, default=True
        If False, the model would filter out English keywords, and 
        min_en_keyword_len and to_lowercase parameter would be 
        meaningless.
    
    to_lowercase: bool, default=False
        If True, the model would convert English string to lower case.
        It might extract different keywords.
    
    Attributes
    ----------
    keywords_components_: OrderedDict
        
    '''
    def __init__(self, 
                 n_keyword=5, 
                 ngram_range=(0,10),
                 min_ch_keyword_len=2, 
                 min_en_keyword_len=3,
                 enable_english=True,
                 to_lowercase=False, 
                 max_parent=False):

        self.n_keyword = n_keyword
        self.min_ch_keyword_len = min_ch_keyword_len
        self.min_en_keyword_len = min_en_keyword_len
        self.ngram_range = ngram_range
        self.enable_english = enable_english
        self.to_lowercase = to_lowercase
        self.max_parent = max_parent
        
    
    def fit(self, X, y=None):
        '''Fit the model with X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, ), List of string
            One dimension array or list of names
        
        y : None
            Ignored variable.
            
        Returns
        -------
        self: object
            Returns the instance itself.
        '''
        
        corpus = list(map(lambda s: ''.join(self._split_string(s)),
                          set(X)))
        
        vectorizer = CountVectorizer(ngram_range=self.ngram_range, 
                                     analyzer='char',
                                     lowercase=self.to_lowercase)
        
        corpus_vec = vectorizer.fit_transform(corpus)
               
        self.keywords_components_ = OrderedDict(zip(
            vectorizer.get_feature_names(),
            np.ravel(corpus_vec.sum(axis=0))
        ))
        # a lot of useless emtpy string, set to 0
        self.keywords_components_[''] = 0  
        return self
    
    def transform(self, X):
        '''Extract keywords from X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, ), List of string
            One dimension array or list of names
            
        
        Returns
        -------
        labels: array-like, shape (n_samples, n_keyword)
            Return keyword strings extracted from sample.
            If the number of extraced keywords is less than n_keyword,
            model will pad with "" string.
        
        '''
        
        # Check is fit had been called
        check_is_fitted(self, ['keywords_components_'])
        
        get_keyword_label = lambda s: self._parse_string(s)[0]
        
        # Input is str
        if isinstance(X, six.string_types):
            return get_keyword_label(X)
        
        # Input validation, only accept 1 dim and str
        X = check_array(X, ensure_2d=False, dtype=six.string_types)
        labels = np.stack(list(map(get_keyword_label, X)))
        return labels
    
    def transform_proba(self, X):
        '''Extract gain value of keywords from X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, ), List of string
            One dimension array or list of names
            
        
        Returns
        -------
        gainvalues: array-like, shape (n_samples, n_keyword)
            Return gain value(int) of keywords extracted from sample.
            If the number of extraced keywords is less than n_keyword,
            model will pad with 0.
        
        '''
        # Check is fit had been called
        check_is_fitted(self, ['keywords_components_'])
        
        get_gainvalue = lambda s: self._parse_string(s)[1]
        
        # Input is str
        if isinstance(X, six.string_types):
            return get_gainvalue(X)
        
        # Input validation, only accept 1 dim and str
        X = check_array(X, ensure_2d=False, dtype=six.string_types)
        gainvalues = np.stack(list(map(get_gainvalue, X)))
        return gainvalues
    
    
    def fit_transform(self, X, y=None):
        '''Fit the model with X and extract keywords from X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, ), List of string
            One dimension array or list of names
            
        y : None
            Ignored variable.
            
        Returns
        -------
        labels: array-like, shape (n_samples, n_keyword)
            Return keyword strings extracted from sample.
            If the number of extraced keywords is less than n_keyword,
            model will pad with "" string.
        
        '''
        self.fit(X) 
        return self.transform(X)
    
    
    def _modify_string(self, string):
        '''Apply strip string and lowercases  '''
        modified_string = string
        
        if self.to_lowercase:
            modified_string = modified_string.lower()

        modified_string = modified_string.strip()
        return modified_string
    
    
    def _split_string(self, string):  
        '''Split sample name into list of strings'''
        string = self._modify_string(string)
        
        if not string:
            return ['']
        
        if self.enable_english:
            pattern = r"[\u4e00-\u9fa5]+|[a-zA-Z]+"
        else:
            pattern = r"[\u4e00-\u9fa5]+"
            
        return re.findall(pattern, string, re.UNICODE)
        
    
    def _parse_en_substring(self, substring):
        '''Parse English string to get keyword and gain value'''
        if len(substring) >= self.min_en_keyword_len:
            keywords_fltr = np.array([substring])
            gain_value_fltr = np.array([self.keywords_components_
                                            .get(substring, 0)])
        else:
            keywords_fltr = np.array([], dtype='U32')
            gain_value_fltr = np.array([], dtype=int)
        
        return keywords_fltr, gain_value_fltr
    
    
    def _parse_ch_substring(self, substring):
        '''Parse Chinese string to get keywords and gain values'''
        # Pascal Structure
        len_str = max(len(substring), 1) # min 1, prevent empty string
        start_position = np.tril(np.indices((len_str,len_str))[1])    
        end_position = sum(map(
            lambda x: np.eye(len_str, k=x, dtype=int) * (len_str + x), 
            range(- len_str + 1, 1)))   
        slices = np.stack([start_position, end_position], axis=2)
        
        keywords = np.apply_along_axis(
            func1d=lambda x: substring[x[0]:x[1]], 
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
        
        if self.max_parent:
            gain_value = keywords_val - np.where(
                                            left_parents_val>right_parents_val,
                                            left_parents_val, right_parents_val)
        else:
            gain_value = keywords_val - left_parents_val - right_parents_val
            
        
        filter_mask = (
            (gain_value > 0) 
            & (np.char.str_len(keywords) >= self.min_ch_keyword_len))
        
        keywords_fltr = keywords[filter_mask]
        gain_value_fltr = gain_value[filter_mask]
        
        return keywords_fltr, gain_value_fltr
    
    
    def _parse_string(self, string):
        '''Parse sample string'''
        substring_list = self._split_string(string)
        
        keywords = np.array([], dtype='U32')
        gain_values = np.array([], dtype=int)
        for substr in substring_list:
            if not substr: continue
            
            if substr[0] in ascii_letters:
                keyword, gain_value = self._parse_en_substring(substr)
            else:
                keyword, gain_value = self._parse_ch_substring(substr)
            
            keywords = np.concatenate([keywords, keyword])
            gain_values = np.concatenate([gain_values, gain_value])   
        
        
        # bug/fix-duplicated-keyword
        _, dupli_idx = np.unique(keywords, return_index=True)
        keywords = keywords[dupli_idx]
        gain_values = gain_values[dupli_idx]       
        
        # Sort by gain_value
        dtype = [('keyword', 'U32'), ('gain_value', int)]
        data = np.array(list(zip(keywords, gain_values)), dtype=dtype)
        data = np.sort(data, order='gain_value')[::-1][0:self.n_keyword]
        
        # Make sure that output has same shape
        keyword_result = np.pad(data['keyword'], 
                                (0, self.n_keyword - data.shape[0]),
                                mode='constant',
                                constant_values='')
        gain_value_result = np.pad(data['gain_value'], 
                                   (0, self.n_keyword - data.shape[0]), 
                                   mode='constant',
                                   constant_values=0)  
        
        return keyword_result, gain_value_result