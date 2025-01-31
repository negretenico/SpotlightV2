import tensorflow as tf

image_size = (180, 180)
batch_size = 128


class Classifier:
    def __init__(self):
        self.train_ds = tf.keras.utils.image_dataset_from_directory('data/train', validation_split=0.2,
                                                                    subset="training",
                                                                    seed=123,
                                                                    image_size=image_size,
                                                                    batch_size=batch_size,
                                                                    label_mode='binary')
        self.val_ds = tf.keras.utils.image_dataset_from_directory('data/test', validation_split=0.2,
                                                                  subset="validation",
                                                                  seed=123,
                                                                  image_size=image_size,
                                                                  batch_size=batch_size,
                                                                  label_mode='binary'
                                                                  )
        self.__base_model__ = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(image_size[0], image_size[1], 3)
        )
        self.__base_model__.trainable = False
        self.model = tf.keras.models.Sequential([
            self.__base_model__,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(384, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

    def compile(self) -> None:
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=['accuracy'])

    def train(self) -> None:
        # Add early stopping and more epochs
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)
        self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=20,
            callbacks=[early_stopping]
        )

    def save(self, path: str) -> None:
        self.model.save(path)

    def predict(self, img_path: str) -> int:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        prediction = self.model.predict(img_array)
        print(f"Raw prediction: {prediction}")  # Diagnostic print
        return int(prediction[0][0] > 0.5)  # Return class based on threshold

    def load(self, filepath):
        """Load a saved model from disk."""
        self.model = tf.keras.models.load_model(filepath)
