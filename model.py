# =============================================================================
# import libraries
# =============================================================================
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import regularizers

# =============================================================================
# Define a CNN model
# =============================================================================


def CNN_model(params):
    C_in, H_in, W_in = params["input_shape"]
    init_f = params["initial_filters"]
    num_fc1 = params["num_fc1"]
    num_classes = params["num_classes"]
    dropout_rate = params["dropout_rate"]
    regularization_strength = params["regularization_strength"]

    model = Sequential()
    model.add(Conv2D(init_f, kernel_size=3, activation='relu',
              kernel_regularizer=regularizers.l2(regularization_strength),
              input_shape=(H_in, W_in, C_in)
                     )
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(2*init_f, kernel_size=3, activation='relu',
                     kernel_regularizer=regularizers.l2(
                         regularization_strength)
                     )
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(4*init_f, kernel_size=3, activation='relu',
                     kernel_regularizer=regularizers.l2(
                         regularization_strength)
                     )
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(8*init_f, kernel_size=3, activation='relu',
                     kernel_regularizer=regularizers.l2(
                         regularization_strength)
                     )
              )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(num_fc1, activation='relu',
                    kernel_regularizer=regularizers.l2(regularization_strength)
                    )
              )
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes-1, activation='sigmoid',
                    kernel_regularizer=regularizers.l2(regularization_strength)
                    )
              )

    return model
