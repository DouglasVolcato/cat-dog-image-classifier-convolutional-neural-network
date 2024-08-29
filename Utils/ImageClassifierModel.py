import tensorflow as tf


class ImageClassifierModel:

    def trainAndSaveModel(self) -> None:
        model = self.__getModel()
        dataset = self.__mapDataset(self.__getTrainDataset())
        self.__trainModel(model, dataset)
        self.__saveModel(model)

    def loadAndEvaluateModel(self) -> None:
        model = self.__getModel()
        test_dataset = self.__mapDataset(self.__getTestDataset())
        self.__evaluateModel(model, test_dataset)

    def predict(self, img_path: str) -> str:
        dataset = tf.data.Dataset.from_tensors((img_path, 0))
        dataset = self.__mapDataset(dataset)
        model = self.__getModel()
        predictions = model.predict(dataset)
        score = predictions[0][0]
        return self.__getLabel(score)

    def __getLabel(self, score) -> str:
        if score > 0.5:
            return "dog"
        else:
            return "cat"

    def __evaluateModel(self, model: tf.keras.Model, test_dataset: tf.data.Dataset):
        model.evaluate(test_dataset)

    def __getTrainDataset(self) -> tf.data.Dataset:
        # Load the dataset with labels
        cat_files = tf.data.Dataset.list_files(
            'cache/train/cats/*.jpg')
        dog_files = tf.data.Dataset.list_files(
            'cache/train/dogs/*.jpg')

        # Create a labeled dataset (0 for cats, 1 for dogs)
        cat_labels = tf.data.Dataset.from_tensor_slices(
            tf.zeros(len(cat_files), dtype=tf.int32))
        dog_labels = tf.data.Dataset.from_tensor_slices(
            tf.ones(len(dog_files), dtype=tf.int32))

        # Pair each image with its corresponding label
        cat_dataset = tf.data.Dataset.zip((cat_files, cat_labels))
        dog_dataset = tf.data.Dataset.zip((dog_files, dog_labels))

        # Combine the cat and dog datasets
        dataset = cat_dataset.concatenate(dog_dataset)

        return dataset

    def __getTestDataset(self) -> tf.data.Dataset:
        # Load the dataset with labels
        cat_files = tf.data.Dataset.list_files('cache/validation/cats/*.jpg')
        dog_files = tf.data.Dataset.list_files('cache/validation/dogs/*.jpg')

        # Create a labeled dataset (0 for cats, 1 for dogs)
        cat_labels = tf.data.Dataset.from_tensor_slices(
            tf.zeros(len(cat_files), dtype=tf.int32))
        dog_labels = tf.data.Dataset.from_tensor_slices(
            tf.ones(len(dog_files), dtype=tf.int32))

        # Pair each image with its corresponding label
        cat_dataset = tf.data.Dataset.zip((cat_files, cat_labels))
        dog_dataset = tf.data.Dataset.zip((dog_files, dog_labels))

        # Combine the cat and dog datasets
        test_dataset = cat_dataset.concatenate(dog_dataset)

        return test_dataset.concatenate(test_dataset)

    def __mapDataset(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        def load_and_resize_image(img_path, label):
            img_raw = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img_raw, channels=3)
            img = tf.image.resize(img, [150, 150])  # Resize to 150x150
            img = img / 255.0  # Normalize to [0, 1] range
            return img, label  # Return both the image and its label

        # Apply the image processing pipeline
        dataset = dataset.map(load_and_resize_image,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # Shuffle, batch, and prefetch the dataset
        batch_size = 32
        dataset = dataset.shuffle(buffer_size=1000).batch(
            batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    def __getModel(self) -> tf.keras.Model:
        # Define the model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(
                32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def __trainModel(self, model: tf.keras.Model, dataset: tf.data.Dataset):
        model.fit(dataset, epochs=12)

    def __saveModel(self, model: tf.keras.Model):
        model.save('cache/models/image_classifier_model.keras')
