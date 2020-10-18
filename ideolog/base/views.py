from django.shortcuts import render
import sys
sys.path.insert(0, './model/')
from VoteClassifier import VoteClassifier
from django.http import JsonResponse
import json

import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from threading import Thread
import pandas as pd

HOUSE_TABLE = pd.read_csv('../data/house_table.csv')

# Create your views here.

def predict(request):
    statement = request.GET.get('statement', '')
    chamber = request.GET.get('chamber', '')
    members = HOUSE_TABLE[(HOUSE_TABLE['chamber'] == chamber) & (HOUSE_TABLE['iteration'] == 116)]
    
    if chamber not in {'house', 'senate'}:
        responseData = {'ok': False, 'msg': "No valid chamber supplied"}

    else:

        names = list(members['name'])
        state = list(members['state'])
        clf = VoteClassifier()

        votes = clf.classify_all_legislators(statement)

        responseData = {
            'ok': True,
            'statement': statement,
            'chamber': chamber,
            'results': [
                {
                    'name': name,
                    'state': state[i],
                    'agree': float(votes.get(name, 0.5))
                } for i, name in enumerate(names)
            ]
        }

    return JsonResponse(responseData)


def senators():
    pass