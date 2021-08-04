from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.http import JsonResponse
from django.contrib import messages
import gensim
import json

from . models import news
from . forms import NewsForm
import joblib
from nltk import word_tokenize


def FormAPI(request):
	if(request.method == 'POST'):
		form = NewsForm(request.POST)
		if(form.is_valid()):
			Number = form.cleaned_data['number']
			Headline = form.cleaned_data['headline']
			Name = form.cleaned_data['name']
			Date = form.cleaned_data['date']
			Content = form.cleaned_data['content']
			print(Content)

			model = joblib.load('MyAPI/pretrained_models/d2v.joblib')
			logreg = joblib.load('MyAPI/pretrained_models/logreg.joblib')

			b = word_tokenize(Content.lower())
			fv = model.infer_vector(b)

			p = logreg.predict([fv])
			print(p)
			res = p[0]
			messages.success(request, 'Application Status: {}'.format(res))

	form = NewsForm()
	return render(request, 'Myform/nform.html', {'form': form})




