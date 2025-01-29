from caption.model.Classifier import Classifier


def classify_caption_using_latest_model(caption: str) -> str:
    model = Classifier.load("./saved_models/1.joblib")
    mappings = {0: 'NOT_TOXIC', 1: 'TOXIC'}
    return mappings.get(model.predict([caption]), 'NOT_TOXIC')


if __name__ == "__main__":
    pass
