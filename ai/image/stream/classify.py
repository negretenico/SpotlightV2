from image.model.Classifier import Classifier


def classify_using_latest(img):
    classifier = Classifier()
    classifier.load('saved_model/model1.keras')
    mappings = {0: 'PASS', 1: 'FAIL'}
    return mappings.get(classifier.predict(img), 'NOT_TOXIC')


if __name__ == "__main__":
    pass
