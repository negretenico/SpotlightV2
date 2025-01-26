import pandas as pd
import sklearn.model_selection
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from Classifier import Classifier
from ModelData import ModelData

vectorizer = TfidfVectorizer(max_features=1000)


def train_test_save():
    # Load and preprocess the data
    x_train, x_test, y_train, y_test = get_data()

    # Use the same vectorizer for all stages
    _vectorizer = TfidfVectorizer(max_features=1000)
    x_train_transformed = _vectorizer.fit_transform(x_train)
    x_test_transformed = _vectorizer.transform(x_test)

    # Create and train the classifier
    classifier = Classifier(
        feature_data=ModelData(test_data=x_test_transformed, train_data=x_train_transformed),
        label_data=ModelData(test_data=y_test, train_data=y_train),
        vectorizer=_vectorizer,
    )
    classifier.train()
    classifier.test()

    # Save the model and vectorizer
    classifier.save("./saved_models/1.joblib")


def load_and_test():
    model = Classifier.load("./saved_models/1.joblib")
    print(model.predict(['Hello I am nico']))
    print(model.predict(['You fukcing idiot']))


def train_test_predict():
    # This is used for local developemtn of hte mnodel
    x_train, x_test, y_train, y_test = get_data()
    classifier = Classifier(feature_data=ModelData(test_data=x_test, train_data=x_train),
                            label_data=ModelData(test_data=y_test, train_data=y_train))
    classifier.train()
    classifier.test()
    print(classifier.predict(vectorizer.transform(['Hello I am nico'])))
    print(classifier.predict(vectorizer.transform(['You fukcing idiot'])))


def get_data():
    # SChema is column1, column2
    #           text    , isToxic
    df = pd.read_csv('data/train.csv')
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(df['is_toxic'])

    return sklearn.model_selection.train_test_split(df['text'], labels, test_size=0.1, random_state=42)


if __name__ == "__main__":
    load_and_test()
