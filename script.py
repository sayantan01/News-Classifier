import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from nltk import word_tokenize
import joblib


class LogisticRegressionModel:
	def __init__(self, dataset_path = './dataset.csv', docvec_path = 'MyAPI/pretrained_models/d2v.joblib', logreg_path = 'MyAPI/pretrained_models/logreg.joblib'):
		self.df = pd.read_csv(dataset_path, usecols = ['category', 'text'])
		self.df.dropna()
		self.train_x = self.df['text']
		self.train_y = self.df['category']
		self.docvec_path = docvec_path
		self.logreg_path = logreg_path


	def generate_document_vectors(self):
		tagged_data = generate_tagged_data(self.train_x, self.train_y)
		d2v_model = Doc2Vec(dm = 1, vector_size = 100, alpha = 0.025, min_alpha = 0.001, min_count = 2)
		d2v_model.build_vocab([x for x in tagged_data])
		d2v_model.train(tagged_data, epochs = 10, total_examples = len(tagged_data))
		joblib.dump(d2v_model, self.docvec_path, compress = 3)
		self.train_y, self.train_x = zip(*[(doc.tags[0], d2v_model.infer_vector(doc.words)) for doc in tagged_data])


	def generate_logisticRegression_models(self):
		self.generate_document_vectors()
		logreg_model = LogisticRegression(C = 1e5)
		logreg_model.fit(self.train_x, self.train_y)
		joblib.dump(logreg_model, self.logreg_path, compress = 3)


def generate_tagged_data(X, Y):
	tagged_data = []
	for i in range(len(X)):
		tagged_data.append(TaggedDocument(word_tokenize(list(X)[i].lower()),[str(list(Y)[i])]))
	return tagged_data



if __name__ == '__main__':
	logreg = LogisticRegressionModel()
	logreg.generate_logisticRegression_models()