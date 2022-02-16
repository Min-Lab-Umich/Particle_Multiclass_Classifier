# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math

# import pandas as pd
import os
import time
import shutil
import sys

import numpy
from numpy import newaxis
import tensorflow as tf
import numpy as np
from itertools import cycle
import csv
from PIL import Image
import cv2
# import keras_tuner as kt

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
# from scipy import interp
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
# from google.colab import files
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD

class DataSetProp:
    
    def __init__(self):
        self.files = None
        self.total = 0
        self.training = 0
        self.validation = 0
        

class Classifier:

    def __init__(self, should_preprocess=False, data_dir="./tmp",
                 load_path=None) -> None:
        self.should_preprocess = should_preprocess
        self.classes = ["1-3", "1-4", "2-2", "2-17", '2-17-cell-tak']
        self.count = {
            class_name : DataSetProp()
            for class_name in self.classes
        }
        self.epochs = 15
        self.dirs = {
            "data": data_dir,
            "train": './tmp/train', "validation": "./tmp/validation",
            "test": "./tmp/test"
        }
        self.process_types = ["training", "validation", "test"]
        # self.data_frames = {}
        self.generators = {}
        self.datasets = {}
        self.batch_sizes = {
            "train": 32,
            "validation": 16,
            "test": 16
        }
        # a tf object
        self.model = None
        self.history = None
        self.load_path = load_path
        if self.load_path is not None:
            self.save_path = self.load_path
        else:
            self.save_path = f"saved_models/saved_model{time.time()}/"

    def preprocessing(self):
        """All the processing steps."""

        prefix = "./tmp/"
        for folder in self.process_types:
            for sub_dir in self.classes :
                os.makedirs(f"{prefix}{folder}/{sub_dir}", exist_ok=True)

        for which_type in self.classes:
            _, _, files = next(os.walk(f"{self.dirs['data']}/{which_type}"))
            self.count[which_type].total = len(files)
            self.count[which_type].files = files

        for which_type in self.classes:
            self.count[which_type].training = math.floor(
                self.count[which_type].total * 0.7)
            self.count[which_type].validation = math.floor(
                self.count[which_type].total * 0) + self.count[which_type].training

        for idx, files in enumerate([
            self.count[class_name].files
            for class_name in self.classes
        ]):
            which_sub_dir = self.classes[idx]
            for i, file in enumerate(files):
                if i <= self.count[which_sub_dir].training:
                    shutil.copyfile(f"{self.dirs['data']}/{which_sub_dir}/{file}",
                                    f"{prefix}{'train'}/{which_sub_dir}/{file}")
                elif i <= self.count[which_sub_dir].validation:
                    shutil.copyfile(f"{self.dirs['data']}/{which_sub_dir}/{file}",
                                    f"{prefix}{'validation'}/{which_sub_dir}/{file}")
                else:
                    shutil.copyfile(f"{self.dirs['data']}/{which_sub_dir}/{file}",
                                    f"{prefix}{'test'}/{which_sub_dir}/{file}")

    def generate_datasets(self):
        self.datasets["training"] = tf.keras.preprocessing.\
                image_dataset_from_directory(f"{self.dirs['data']}/training",
                    validation_split=0.2,
                    subset="training",
                    label_mode='categorical',
                    seed=123, color_mode='rgb',
                    image_size=(100, 100),
                    batch_size=32)
        self.datasets["test"] = tf.keras.preprocessing. \
            image_dataset_from_directory(f"{self.dirs['data']}/test",
                                         seed=123, color_mode='rgb',
                                         label_mode='categorical',
                                         image_size=(100, 100),
                                         batch_size=32)

    def visualize_data(self):
        """Display some sample images."""
        class_names = self.datasets['training'].class_names
        plt.figure(figsize=(10, 10))
        for images, labels in self.datasets['training'].take(1):
            for i in range(5):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.title(class_names[numpy.nonzero(labels[i])[0][0]])
                plt.axis("off")
        plt.show()

    # def generate_generators(self):
    #     """Generate the generators."""
    #     scale = 1 / 255
    #     for process_type in self.process_types:
    #         keras_gen = ImageDataGenerator(rescale=scale)
    #         self.generators[process_type] = keras_gen.flow_from_directory(
    #             self.dirs[process_type],
    #             # This is the source directory for training images
    #             classes=self.classes,
    #             target_size=(100, 100),
    #             # All images will be resized to 200x200
    #             batch_size=self.batch_sizes[process_type],
    #             color_mode='grayscale',
    #             shuffle=False,
    #             # Use binary labels
    #             class_mode='categorical')
    #     pass

    def create_model(self):
        IMG_SHAPE = (100, 100, 3)
        base_model = tf.keras.applications.resnet50.ResNet50(
            input_shape=IMG_SHAPE,include_top=False,weights='imagenet')
        base_model.trainable = False
        base_model.summary()
        prediction_layer = tf.keras.layers.Dense(5)
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.model = tf.keras.Sequential([
            base_model,
            global_average_layer,
            prediction_layer
        ])

        self.model.summary()


        # # All images will be rescaled by 1./255
        # self.model = tf.keras.models.Sequential([
        #     # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        #     # This is the first convolution
        #     tf.keras.layers.Conv2D(16, (3, 3), activation="relu",
        #                            input_shape=(100, 100, 1)),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     # The second convolution
        #     tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     # The third convolution
        #     tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        #     tf.keras.layers.MaxPooling2D(2, 2),
        #     # The fourth convolution
        #     tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        #     # tf.keras.layers.MaxPooling2D(2, 2),
        #     # # The fifth convolution
        #     # tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        #     tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        #     tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        #     tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        #     # tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        #     # tf.keras.layers.MaxPooling2D(2, 2),
        #     # Flatten the results to feed into a DNN
        #     tf.keras.layers.Flatten(),
        #     # 512 neuron hidden layer
        #     tf.keras.layers.Dense(512, activation="relu"),
        #     # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
        #     tf.keras.layers.Dense(len(self.classes), activation='softmax')])
        base_learning_rate = 0.0001
        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self):

        # https://towardsdatascience.com/10-minutes-to-building-a-cnn-binary-image-classifier-in-tensorflow-4e216b2034aa
        # X_train, Y_train, X_test, Y_test = self.get_split_binary_data(
        # )

        self.create_model()
        # tuner = kt.RandomSearch(
        #     self.model,
        #     objective='val_loss',
        #     max_trials=5)

        self.history = self.model.fit(self.datasets['training'],
                                      epochs=self.epochs,
                                      verbose=1,
                                      # validation_data=self.datasets[
                                      #     'validation'],
                                      # validation_steps=8
                                      )

    def visualize_training_result(self):
        acc = self.history.history['accuracy']
        # val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        # val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        # plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    # def draw_roc_curve(self):
    #     STEP_SIZE_TEST = self.generators['test'].n \
    #                      // self.generators['test'].batch_size
    #     self.generators['test'].reset()
    #
    #     preds = self.model.predict_proba(self.generators['test'], verbose=1)
    #     false_positive_rate, true_positive_rate, _ = \
    #         roc_curve(self.generators['test'].classes, preds)
    #     roc_auc = auc(false_positive_rate, true_positive_rate)
    #     plt.figure()
    #     lw = 2
    #     plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
    #              lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title('Receiver Operating Characteristic')
    #     plt.legend(loc="lower right")
    #     plt.show()

    def validate(self):
        """Validate."""
        metrics = self.model.evaluate(self.datasets['validation'])
        print("validation loss and accuracy:")
        print(metrics)

        # self.draw_roc_curve()

    def test(self):
        """Test."""
        metrics = self.model.evaluate(self.datasets['test'])
        print("Testing metrics")
        print(metrics)

        # csv_output_path = "./tmp/pred_labels.csv"
        # with open(csv_output_path, 'w') as opened_csv:
        #     csv_writer = csv.writer(opened_csv)
        #
        #     for dir in ["./tmp/test/1-3", "./tmp/test/1-4", "./tmp/test/2-2", "./tmp/test/2-17"]:
        #         for file in os.listdir(dir):
        #             # for path, directories, files in os.walk("./tmp/test/"):
        #             #     for file in directories:
        #             if file.endswith(f".{'png'}"):
        #                 # path = "labelled_data/" + file
        #                 path = dir + "/" + file
        #                 # im = Image.open(path)
        #                 # im.show()
        #
        #                 # img = image.load_img(path, target_size=(100, 100))
        #                 # x = image.img_to_array(img)
        #
        #                 # img = cv2.imread(path)
        #                 # img = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_NEAREST)
        #                 # b, g, r = cv2.split(img)
        #                 # x = cv2.merge((r, g, b))
        #
        #                 x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #                 x = cv2.resize(x, dsize=(100, 100),
        #                                interpolation=cv2.INTER_NEAREST)
        #                 x = x[:, :, newaxis]
        #                 # x = Image.fromarray(img)
        #
        #                 plt.imshow(x / 255.)
        #                 plt.show()
        #                 x = np.expand_dims(x, axis=0)
        #                 images = np.vstack([x])
        #                 classes = self.model.predict(images, batch_size=10)
        #                 classes = round(classes[0][0])
        #                 csv_writer.writerow([path, classes])

    def load_model(self):
        self.model = tf.keras.models.load_model(self.save_path)

    def plot_model(self):
        if self.load_path is not None:
            self.save_path = self.load_path

        tf.keras.utils.plot_model(self.model,
                                  to_file=f"{self.save_path}/model.png",
                                  rankdir='LR',
                                  dpi=300)

    def save(self):
        if self.load_path is None:
            os.makedirs(self.save_path)
            self.model.save(self.save_path)
            self.plot_model()

    def run(self):
        """Entry point."""
        if self.should_preprocess:
            self.preprocessing()
        self.generate_datasets()
        self.visualize_data()

        # self.generate_generators()
        if self.load_path is None:
            self.train()
            self.visualize_training_result()
            self.save()
        else:
            self.load_model()
        # self.validate()
        self.test()

        # uploaded = files.upload()

        # for fn in uploaded.keys():

        #     # predicting images
        #     path = '/content/' + fn
        #     img = image.load_img(path, target_size=(100, 100))
        #     x = image.img_to_array(img)
        #     plt.imshow(x / 255.)
        #     x = np.expand_dims(x, axis=0)
        #     images = np.vstack([x])
        #     classes = model.predict(images, batch_size=10)
        #     print(classes[0])
        #     if classes[0] < 0.5:
        #         print(fn + " is a particle")
        #     else:
        #         print(fn + " is not a particle")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # saved_model_path = str(sys.argv[1])
    classifier = Classifier(should_preprocess=False)
    classifier.run()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
