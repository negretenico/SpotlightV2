import tensorflow as tf

from Classifier import Classifier


def train():
    # List and configure GPUs
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the first GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)

    classifier = Classifier()
    classifier.compile()
    classifier.train()
    classifier.save('saved_model/model1.keras')


if __name__ == "__main__":
    train()
