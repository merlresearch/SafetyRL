# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.models import Model


class Custom64CNN(Model):
    def __init__(self, input_shape, output_dim, output_activation=None, fine_tuning=False):
        super().__init__()
        self._input_shape = input_shape

        # base layers
        self.conv1 = Conv2D(
            32, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu", trainable=not fine_tuning
        )
        self.conv2 = Conv2D(
            64, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu", trainable=not fine_tuning
        )
        self.conv3 = Conv2D(
            64, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu", trainable=not fine_tuning
        )
        self.gp = GlobalAveragePooling2D()
        self.fc1 = Dense(256, activation="relu")
        self.out = Dense(output_dim, activation=output_activation)

        dummy_inputs = tf.constant(np.zeros(shape=(1,) + input_shape, dtype=np.float32))

        with tf.device("/cpu:0"):
            self(dummy_inputs)

    def call(self, inputs):
        # To save memory, convert standardize image to [0, 1] only on run time
        goselo_img = tf.divide(tf.cast(inputs, tf.float32), tf.constant(255.0))
        x = self.conv1(goselo_img)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.gp(x)
        x = self.fc1(x)
        return self.out(x)

    def get_waypoints_and_feature(self, inputs):
        # To save memory, convert standardize image to [0, 1] only on run time
        goselo_img = tf.divide(tf.cast(inputs, tf.float32), tf.constant(255.0))
        x = self.conv1(goselo_img)
        x = self.conv2(x)
        x = self.conv3(x)
        cnn_feature = self.gp(x)
        x = self.fc1(cnn_feature)
        waypoints = self.out(x)
        return waypoints, cnn_feature

    def extract_feature(self, inputs):
        # To save memory, convert standardize image to [0, 1] only on run time
        goselo_img = tf.divide(tf.cast(inputs, tf.float32), tf.constant(255.0))
        x = self.conv1(goselo_img)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.gp(x)

    @property
    def feature_size(self):
        dummy_inputs = tf.constant(np.zeros(shape=(1,) + self._input_shape, dtype=np.float32))
        return self.extract_feature(dummy_inputs).shape[1]


if __name__ == "__main__":
    net = Custom64CNN(input_shape=(64, 64, 6), output_dim=10)
    inputs = np.ones(shape=(32, 64, 64, 6))
    way_points = net(inputs)
    way_points, feature = net.get_waypoints_and_feature(inputs=inputs)
    print("feature shape: {}, feature size: {}".format(feature.shape, net.feature_size))
