import os

from Classifier import Classifier


def load_test():
    classifier = Classifier()
    classifier.load('saved_model/model1.keras')
    for file in os.listdir('data/live'):
        file_path = os.path.join('data/live', file)
        prediction = classifier.predict(file_path)
        print(f"For file {file} we predicted {prediction}")


if __name__ == "__main__":
    load_test()
