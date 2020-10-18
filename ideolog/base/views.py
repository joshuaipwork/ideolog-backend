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

HOUSE_TABLE = pd.read_csv('../data/house_table.csv')

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

def predict(request):
    statement = request.GET.get('statement')
    chamber = request.GET.get('chamber')
    members = HOUSE_TABLE[HOUSE_TABLE['chamber'] == chamber]
    
    names = list(members['name'])
    state = list(members['state'])

    import random

    responseData = {
        'ok': True,
        'statement': statement,
        'chamber': chamber,
        'results': [
            {
                'name': name,
                'state': state[i],
                'agree': random.random()
            } for i, name in enumerate(names)
        ]
    }

    return JsonResponse(responseData)


def senators():
    pass