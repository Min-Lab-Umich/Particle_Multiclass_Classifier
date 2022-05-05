import sys
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import shutil
import matplotlib.pyplot as plt

class Predictor:
    """Given a saved model in tensorflow, run the prediction."""

    def __init__(self):
        self.model_path = str(sys.argv[1])
        self.dir = str(sys.argv[2])
        self.classes = ['1-1', '1-3', '1-12', '1-18', '2-10', '2-17']
        self.dataset = tf.data.Dataset.list_files(f"{self.dir}/*.png")
        self.img_size = (50, 50)
        self.model = None
        # used to store the inputs for the predictions
        self.image_arrays = []
        self.image_paths = []

        self.load_model()
        self.result = None

        # store the prediction array
        self.predictions = None

    def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.summary()

    def load_data(self):
        """Read in the data from the directory.
        Convert to tensorflow format."""
        print("loading data")
        for image in os.listdir(self.dir):
            if ".png" in image:
                image_path = os.path.join(self.dir, image)
                # print(image_path)
                self.image_paths.append(image)
                image = tf.keras.preprocessing.image.load_img(image_path,
                                                              color_mode='grayscale',
                                                              target_size=self.img_size)
                image_arr = tf.keras.preprocessing.image.img_to_array(image)
                image_arr = tf.image.grayscale_to_rgb(tf.convert_to_tensor(image_arr)).numpy()
                # plt.imshow(image_arr,  cmap=plt.get_cmap('gray'))
                # plt.show()
                self.image_arrays.append(image_arr)

        self.image_arrays = np.array(self.image_arrays)
        print("loaded data")

    def predict(self):
        self.predictions = np.argmax(self.model.predict(self.image_arrays), axis=1)
        print(self.predictions)

    def output(self):
        for class_name in self.classes:
            os.makedirs(f"{self.dir}/{class_name}", exist_ok=True)

        for index, val in enumerate(self.predictions):
            print(f"{self.image_paths[index]}: {val}")
            shutil.move(f"{self.dir}/{self.image_paths[index]}",
                        f"{self.dir}/{self.classes[val]}/{self.image_paths[index]}")


if __name__ == "__main__":
    predictor = Predictor()
    predictor.load_data()
    predictor.predict()
    predictor.output()