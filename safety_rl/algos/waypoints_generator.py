# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import tensorflow as tf

from safety_rl.experiments.generate_dataset import decode_way_points_to_world_coord
from safety_rl.networks.cnn import Custom64CNN


class WayPointsGenerator:
    """
    Train GOSELO model whose output is categorical distribution
    """

    def __init__(self, input_shape, output_dim, batch_size=32, lr=0.0001):
        self.model = Custom64CNN(input_shape, output_dim, output_activation="linear")

        # for supervised learning
        self._train_accuracy = tf.keras.metrics.MeanSquaredError()
        self._test_accuracy = tf.keras.metrics.MeanSquaredError()
        self._compute_loss = tf.keras.losses.MeanSquaredError()
        self._train_loss = tf.keras.metrics.Mean()
        self._test_loss = tf.keras.metrics.Mean()

        self._lr = lr
        self.output_dim = int(output_dim)
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.batch_size = batch_size

    @tf.function
    def train(self, inputs, labels):
        with tf.GradientTape() as tape:
            preds = self.model(inputs)
            loss = self._compute_loss(labels, preds)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self._optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self._train_loss(loss)
        self._train_accuracy(labels, preds)

    @tf.function
    def evaluate(self, inputs, labels):
        preds = self.model(inputs)
        loss = self._compute_loss(labels, preds)
        self._test_loss(loss)
        self._test_accuracy(labels, preds)

    def reset_stats(self):
        self._train_loss.reset_states()
        self._test_loss.reset_states()
        self._train_accuracy.reset_states()
        self._test_accuracy.reset_states()

    def get_action(self, inputs, test=False, policy="normalize"):
        """
        We have various ways to compute next actions because the output of NN is waypoints.
        :param inputs: GOSELO-like images
        :param test: if True, choose action greedily
        :param policy:
            "normalize": move second closest way-point direction with the size is 1 (np.linalg.norm(action) = 1)
            "all": return all way-points obtained from neural network
        :return:
        """
        single_input = inputs.ndim == 3  # (width, height, 6) channels
        if single_input:
            inputs = np.expand_dims(inputs, axis=0)

        model_out = self._get_way_points_body(inputs)
        if len(model_out) == 2:
            way_points = model_out[1].numpy()
        else:
            way_points = model_out.numpy()

        if policy == "normalize":
            # Convert GOSELO coord. to World coordinate
            way_points_w = decode_way_points_to_world_coord(self, way_points.reshape(-1, 2))

            # Output normalized data whose norm of output should be 1
            nearest_point = np.ravel(way_points_w)[2:4]
            return nearest_point / np.linalg.norm(nearest_point)
        elif policy == "all":
            return way_points[0]
        else:
            raise NotImplementedError

    @tf.function
    def _get_way_points_body(self, inputs):
        """
        Compute way points by forward propagation
        """
        return self.model(inputs)

    def _convert_relative_pos(self, relative_way_points):
        relative_way_points = relative_way_points.reshape(-1, 2)
        absolute_way_points = np.zeros_like(relative_way_points)
        for i in range(relative_way_points.shape[0] - 1):
            absolute_way_points[i + 1] = absolute_way_points[i] + relative_way_points[i + 1]
        return np.ravel(absolute_way_points)

    def extract_feature(self, inputs):
        single_input = inputs[0].ndim == 3  # (width, height, 6) channels
        if single_input:
            inputs = (np.expand_dims(inputs[0], axis=0), np.expand_dims(inputs[1], axis=0))

        features = self._extract_feature_body(inputs)

        if single_input:
            return features.numpy()[0]
        else:
            return features.numpy()

    @tf.function
    def _extract_feature_body(self, inputs):
        return self.model.extract_feature(inputs)

    @property
    def train_loss(self):
        return self._train_loss.result()

    @property
    def train_accuracy(self):
        return self._train_accuracy.result()

    @property
    def test_loss(self):
        return self._test_loss.result()

    @property
    def test_accuracy(self):
        return self._test_accuracy.result()

    def get_feature_size(self):
        return self.model.feature_size


def move_toward_kth_waypoint(abs_way_points, current_pos, goal_pos, k=0):
    assert isinstance(abs_way_points, np.ndarray) and abs_way_points.ndim == 1
    # Convert GOSELO coord to World coord
    abs_way_points_world = decode_way_points_to_world_coord(current_pos, goal_pos, abs_way_points.reshape(-1, 2))

    # Output normalized data whose norm of output should be 1
    nearest_point = np.ravel(abs_way_points_world)[2 * k : 2 * (k + 1)]
    return nearest_point / (k + 1)
    # action = nearest_point / np.linalg.norm(nearest_point)
    # return action
