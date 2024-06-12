# =============================================================================
# import librarires
# =============================================================================
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from download_dataset import download_datasets
from sklearn.metrics import accuracy_score
from PIL import Image, ImageDraw
from util import Keras_CustomSequence, lr_schedule
from sklearn.model_selection import train_test_split
from model import CNN_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler


import warnings

# Filter out the specific UserWarning
warnings.filterwarnings("ignore")

"""
1- Exploring the dataset
2- Read the dataaset
3- Splitting the dataset
4- Creating dataloaders
5- Building the classification model
6- Defining a callback
7- compile and fit the model
10- Deploying the model
11- Performance of the model on the test data
12- Submit the results 
"""

# =============================================================================
# 1- Exploaring the datatset
# =============================================================================
# Change this to your desired directory
# destination_dir = '/home/saeid/Desktop/Machine_Learning_Projects/Datasets'
data_dir = os.path.join(os.getcwd(), 'dataset')
print(f'data_dir: {data_dir}')
# Make sure that the destination_dir is available
if not os.path.exists(data_dir):
    os.mkdir(data_dir)
# Download the dataset
# download_datasets(data_dir)

# =============================================================================
# 2- Read the dataset
# =============================================================================
# Read the CSV file containing filenames and labels
csv_file_path = os.path.join(data_dir, "train_labels.csv")
data_df = pd.read_csv(csv_file_path)
# Extract filenames and labels from the DataFrame
full_img_names = [os.path.join(data_dir, 'train', filename)
                  for filename in data_df["id"]]
full_img_names = [path + '.tif' for path in full_img_names]
labels = data_df["label"].values
# =============================================================================
# 3- Splitting the dataset
# =============================================================================
train_img_names, val_img_names, train_labels, val_labels = train_test_split(
    full_img_names,
    labels,
    test_size=0.2,
    random_state=42)

# =============================================================================
# 4- Creating Datalaoders
# =============================================================================
batch_size = 32
train_data_generator = Keras_CustomSequence(train_img_names,
                                            train_labels,
                                            batch_size=batch_size,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            rotation_range=45)

val_data_generator = Keras_CustomSequence(val_img_names,
                                          val_labels,
                                          batch_size=batch_size,
                                          horizontal_flip=True,
                                          vertical_flip=True,
                                          rotation_range=45)

# =============================================================================
# 5- Building the classification model
# =============================================================================
params_model = {
    "input_shape": (3, 96, 96),
    "initial_filters": 8,
    "num_fc1": 100,
    "dropout_rate": 0.25,
    "num_classes": 2,
    "regularization_strength": 0.01
}
model = CNN_model(params_model)
model.summary()

# =============================================================================
# 6- Defining a callback
# =============================================================================
path4save = os.path.join(os.getcwd(), 'result') 
if not os.path.exists(path4save):
    os.mkdir(path4save)
# Define the filename where the model will be saved
filepath = os.path.join(path4save, 'model_checkpoint.h5')
callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=filepath,
                                    save_best_only=True,
                                    monitor="val_loss"),
    LearningRateScheduler(lr_schedule)

]

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=filepath,
                                    save_best_only=True,
                                    monitor="val_loss"),
    # Adding the learning rate scheduler callback
    LearningRateScheduler(lr_schedule)
]
# =============================================================================
# 6- compile
# =============================================================================
lr = 1e-4
model.compile(optimizer=Adam(learning_rate=lr),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_data_generator,
                    epochs=100,
                    validation_data=val_data_generator,
                    callbacks=callbacks)
