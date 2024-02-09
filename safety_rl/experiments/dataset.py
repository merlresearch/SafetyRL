# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import glob
import logging
import os
import random
import re

import joblib
import numpy as np
from tqdm import tqdm


def save_path(samples, filename):
    joblib.dump(samples, filename, compress=3)


def load_a_path(filename):
    assert os.path.exists(filename)
    return joblib.load(filename)


def get_filenames_from_prefix(dir_name, prefix=None, n_path=None, shuffle=True):
    # if prefix is None:
    #     candidates = glob.glob(os.path.join(dir_name, "*.pkl"))
    #     prefix = candidates[0].split("_dataset_")[0].split("/")[-1]
    #     print(prefix)
    # itr_reg = re.compile(r"{}_dataset_(?P<idx>[0-9]+).pkl".format(prefix))
    itr_reg = re.compile(r"(?P<idx>[0-9]+).pkl")

    itr_files = []
    for _, filename in enumerate(os.listdir(dir_name)):
        m = itr_reg.match(filename)
        if m:
            itr_count = m.group("idx")
            itr_files.append((itr_count, filename))

    n_path = n_path if n_path is not None else len(itr_files)
    itr_files = sorted(itr_files, key=lambda x: int(x[0]), reverse=True)[:n_path]

    filenames = []
    for itr_file_and_count in itr_files:
        filenames.append(os.path.join(dir_name, itr_file_and_count[1]))
    if shuffle:
        random.shuffle(filenames)
    return filenames


def load_paths_from_prefix(dir_name, prefix=None, max_size=None):
    """Load paths generated by A*.

    Args:
        filename_prefix: Prefix to dataset filename.
            If you generate paths using following command:
            $ python experiments/generate_dataset.py --path-dir results/dataset/ --save-data --dataset-size 10000
            then, prefix will be `results/dataset/YYYYMMddTHHmmss`.
    """
    logger = logging.getLogger("safety_rl")

    filenames = get_filenames_from_prefix(dir_name, prefix)
    assert len(filenames) > 0, "Cannot find file at {}".format(dir_name)

    logger.info("loading files...")
    for i, filename in enumerate(tqdm(filenames)):
        if i == 0:
            path = load_a_path(filename)
            inputs, way_points = path["inputs"], path["way_points"]
        else:
            path = load_a_path(filename)
            _inputs, _way_points = path["inputs"], path["way_points"]
            inputs = np.concatenate([inputs, _inputs])
            way_points = np.concatenate([way_points, _way_points])
        if max_size is not None and inputs.shape[0] > max_size:
            logger.info(
                "Loaded size {} is over specified maximum size {}, so quit loading.".format(inputs.shape[0], max_size)
            )
            break
    return inputs, way_points


def load_dataset(
    max_size=None, dataset_dir=None, dataset_name_prefix=None, split_ratio=0.1, normalize_factor=1.0, shuffle=False
):
    assert (dataset_name_prefix is not None) or (dataset_dir is not None)

    logger = logging.getLogger("safety_rl")
    if dataset_dir is not None:
        inputs, way_points = load_paths_from_prefix(dataset_dir, prefix=None, max_size=max_size)
    else:
        dir_name, prefix = dataset_name_prefix.rsplit("/", 1)
        inputs, way_points = load_paths_from_prefix(dir_name, prefix, max_size=max_size)

    # Divide dataset into training and test
    divide_idx = int(split_ratio * inputs.shape[0])
    if shuffle:
        inputs, way_points = shuffle_ndarray_in_unison(inputs, way_points)

    x_test, y_test = inputs[:divide_idx], way_points[:divide_idx]
    x_train, y_train = inputs[divide_idx:], way_points[divide_idx:]

    logger.info("Loading GOSELO waypoints...")
    output_dim = y_train.shape[1]
    logger.info("Output is normalized by a factor of {}".format(normalize_factor))

    y_train = y_train.astype(np.float32) / normalize_factor
    y_test = y_test.astype(np.float32) / normalize_factor
    return x_train, y_train, x_test, y_test, output_dim


def shuffle_ndarray_in_unison(a, b, c):
    assert a.shape[0] == b.shape[0] == c.shape[0]
    indices = np.random.permutation(a.shape[0])
    return a[indices], b[indices], c[indices]
