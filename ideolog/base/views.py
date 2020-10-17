from django.shortcuts import render
from django.http import JsonResponse
import json

import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pandas as pd

# Create your views here.

class ModelSerializer:

    def __init__(self, whateverJoshMade):
        self._joshArtifact = whateverJoshMade

    def predict(self, congress_member_id: str, statement: str) -> float:
        raise NotImplementedError("Abstract!")

    def load(self):
        raise NotImplementedError("Abstract!")

    def build(self):
        raise NotImplementedError("Abstract!")

    def load_or_build(self):
        raise NotImplementedError("Abstract!")

class BaseModel(ModelSerializer):
    
    def __init__(self, *args, **kwargs):
        self.cache_location = 'base-model.pepengra'

    def predict(self, congress_member_id: str, statement: str):
        pass

    def load(self):
        pass

    def build(self):
        members = pd.read_csv('../data/house_table.csv')['id'].unique()
        vote_table = pd.read_csv('../data/vote_table.csv')
        self.model = {}
        for member in members:
            self.model[member] = pipeline = Pipeline([
                ('vect', CountVectorizer(ngram_range=(1,2))),
                ('tfidf', TfidfTransformer(use_idf=True)),
                ('clf', MultinomialNB(alpha=1e-3, fit_prior=True))
            ]).fit(
                
            )


        with open(self.cache_location, 'wb') as handle:
            pickle.dump(self.model, handle)



def predict(request):
    member = request.POST.get('member')
    statement = request.POST.get('statement')
    responseData = {
        'ok': True,
        'member': member,
        'statement': statement
    }

    return JsonResponse(responseData)