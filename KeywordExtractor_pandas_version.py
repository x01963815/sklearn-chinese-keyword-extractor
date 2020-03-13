#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_is_fitted, check_array


class KeywordExtractor(BaseEstimator, ClassifierMixin):
    def __init__(self, min_keyword_len=2, 
                 num_keyword=5, ngram_range=(0,10),
                 regex_str=r'[a-zA-Z0-9~!@#$%^&*()_&#43;\-\+=,.'
                           '<>?\|/ \[\]\-"\　０-\９（）？【】]'):
        '''
        '''
        self.min_keyword_len = min_keyword_len
        self.num_keyword = num_keyword
        self.regex_str = regex_str
        self.ngram_range = ngram_range
        
    
    def fit(self, X, y=None):
        '''
        '''

        corpus = pd.Series(np.unique(X)).str.replace(pat=self.regex_str, 
                                                     repl='',
                                                     regex=True)
        
        vectorizer = CountVectorizer(ngram_range=self.ngram_range, 
                                     analyzer='char')
        
        corpus_vec = vectorizer.fit_transform(corpus)

        self.keywords_components_ = (pd.Series(
                np.ravel(corpus_vec.sum(axis=0)),
                index=vectorizer.get_feature_names()
            ).sort_values(ascending=False))
        # a lot useless emtpy string, set to 0
        self.keywords_components_[''] = 0  
        
        return self
    
    def predict(self, X):
        
        # Check is fit had been called
        check_is_fitted(self, ['keywords_components_'])
        
        # Input validation, only accept 1 dim and str
        X = check_array(X, ensure_2d=False, dtype=str).reshape(-1)
        
        labels = pd.Series(X).apply(
            lambda x: np.resize(self._parse_string(x)['keyword'].values, 
                                (self.num_keyword,)))
        return np.stack(labels)
    
    def predict_proba(self, X):
        
        # Check is fit had been called
        check_is_fitted(self, ['keywords_components_'])
        
        # Input validation, only accept 1 dim and str
        X = check_array(X, ensure_2d=False, dtype=str).reshape(-1)
        
        labels = pd.Series(X).apply(
            lambda x: np.resize(self._parse_string(x)['importance'].values, 
                                (self.num_keyword,)))
        return np.stack(labels)
    
    def fit_predict(self, X, y=None):
        raise NotImplementedError
        return 
    
    def _string_modify(self, string):
        if not isinstance(string, str):
            raise ValueError('Input should be string')
        
        modified = pd.Series([string,]).str.replace(
            pat=self.regex_str,
            repl='', 
            regex=True).iloc[0]
        
        return modified
    
    
    def _parse_string(self, string):
        
        string = self._string_modify(string)
        
        len_str = len(string)

        start_position = np.tril(np.indices((len_str,len_str))[1])
        
        end_position = sum(map(
            lambda x: np.eye(len_str, k=x, dtype=int) * (len_str + x), 
            range(-len_str+1, 1)))

        slices = np.stack([start_position, end_position], axis=2)
        keywords = np.apply_along_axis(
            func1d=lambda x: string[x[0]:x[1]], 
            axis=2, 
            arr=slices)
        
        left_parents = np.pad(array=keywords, 
                              pad_width=(1,0), 
                              mode='constant', 
                              constant_values='')[:-1, :-1]
        
        right_parents = np.pad(array=keywords, 
                               pad_width=(1,0), 
                               mode='constant', 
                               constant_values='')[:-1, 1:]
        
        node_array = np.stack([keywords, left_parents, right_parents], 
                             axis=2)

        node_df = (pd.DataFrame(node_array.reshape(-1,3), 
                               columns=['keyword', 
                                        'left_parent', 
                                        'right_parent'])
            .query('keyword != ""')
            .assign(length = lambda df: df['keyword'].apply(len))
            .query('length >= {0}'.format(self.min_keyword_len))                  
            .assign(string = string)
            .assign(keyword_val = lambda df: (df['keyword'].apply(
                        lambda x: self.keywords_components_.get(x, 0))),
                    
                    left_val = lambda df: (df['left_parent'].apply(
                        lambda x: self.keywords_components_.get(x, 0))),

                    right_val = lambda df: (df['right_parent'].apply(
                        lambda x: self.keywords_components_.get(x, 0))))
            .assign(gain_value = lambda df: 
                df['keyword_val'] - df['left_val'] - df['right_val'])
            .query('gain_value > 0')
            .assign(importance = lambda df: (df['gain_value'] 
                                             / df['gain_value'].sum()))
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
            .head(self.num_keyword)
        )
        return node_df