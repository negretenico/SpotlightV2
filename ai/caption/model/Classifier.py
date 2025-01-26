import os

import joblib
from sklearn.naive_bayes import BernoulliNB

from ModelData import ModelData


class Classifier:
    def __init__(self, feature_data: ModelData = None, label_data: ModelData = None, vectorizer=None):
        self.feature_data = feature_data
        self.label_data = label_data
        self.__model__ = BernoulliNB()
        self.__vectorizer__ = vectorizer

    def train(self):
        print(f"We have begun training")
        self.__model__.fit(self.feature_data.train_data, self.label_data.train_data)
        print(f"We have finished training")

    def test(self):
        print(f"We have begun testing the model")
        score = self.__model__.score(self.feature_data.test_data, self.label_data.test_data)
        print(f"We have finished testing the model we achieved a score of {score}")

    def predict(self, feature):
        return self.__model__.predict(self.__vectorizer__.transform(feature))

    def save(self, path):
        print(f"Saving model and vectorizer to {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.__model__, "vectorizer": self.__vectorizer__}, path)

    @staticmethod
    def load(path):
        print(f"Loading model and vectorizer from {path}")
        data = joblib.load(path)
        classifier = Classifier()
        classifier.__model__ = data["model"]
        classifier.__vectorizer__ = data["vectorizer"]
        return classifier
