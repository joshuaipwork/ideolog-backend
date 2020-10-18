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


def predict(request):
    member = request.POST.get('member')
    statement = request.POST.get('statement')
    responseData = {
        'ok': True,
        'member': member,
        'statement': statement
    }

    return JsonResponse(responseData)