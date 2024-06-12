import os
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf

from torch.utils.data import Dataset
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# =============================================================================
# Create a custom dataset in Pytorch dataframe
# =============================================================================


class Pytorch_CustomDataset(Dataset):
    def __init__(self, data_dir, transform, data_type="train"):
        # path to images
        data_path = os.path.join(data_dir, data_type)
        # list of images
        self.img_names = os.listdir(data_path)
        # get the full path to images
        self.full_filenames = [
            os.path.join(data_path, image_name)
            for image_name in self.img_names
        ]
        # load the labels saved in csv file
        labels_path = os.path.join(data_dir, data_type + "_labels.csv")
        labels_df = pd.read_csv(labels_path)
        # set data frame index to id
        labels_df.set_index("id", inplace=True)

        # obtain labels from data frame
        self.labels = [
            labels_df.loc[image_name[:-4]].values[0]
            for image_name in self.img_names
        ]

        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.full_filenames)

    def __getitem__(self, idx):
        # open image, apply transforms and return with label
        img = Image.open(self.full_filenames[idx])
        img = self.transform(img)
        return img, self.labels[idx]

# =============================================================================
# custom dataset for Keras
# =============================================================================


class Keras_CustomSequence(Sequence):
    def __init__(self, images, labels, batch_size,
                 horizontal_flip=True, vertical_flip=True, rotation_range=45):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size

        # Initialize ImageDataGenerator for data augmentation
        self.data_generator = ImageDataGenerator(
            rescale=1./255,  # Rescale pixel values to [0, 1]
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rotation_range=rotation_range
        )

    def __len__(self):
        # return number of batches
        return len(self.images) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.images[idx *
                                   self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx *
                                   self.batch_size:(idx + 1) * self.batch_size]

        images = []
        for img in batch_images:
            img = load_img(img)
            img = img_to_array(img)  # Convert image to array
            img = img.astype(np.float32)  # Convert image to float32
            img = self.data_generator.random_transform(
                img)  # Apply transformations
            images.append(img)

        return np.array(images), np.array(batch_labels)
# =============================================================================
#   learning schedule for keras mdoel
# =============================================================================


def lr_schedule(epoch, lr):
    if epoch < 20:
        return lr
    else:
        return lr * tf.math.exp(-0.1)  # Exponential decay after 50 epoc
