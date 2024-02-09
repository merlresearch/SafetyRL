# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import logging

import cv2
import numpy as np

from safety_rl.experiments.dataset import load_dataset
from safety_rl.experiments.trainer import Trainer
from safety_rl.misc.logger import initialize_logger


def visualize_data(goselo_imgs, sleep_ms=int(1000 / 30)):
    temp = np.ones(shape=(goselo_imgs.shape[2], 3)) * 255
    for goselo_img in goselo_imgs:
        concat_img = np.concatenate(
            (
                goselo_img[:, :, 0],
                temp,
                goselo_img[:, :, 1],
                temp,
                goselo_img[:, :, 2],
                temp,
                goselo_img[:, :, 3],
                goselo_img[:, :, 4],
                goselo_img[:, :, 5],
            ),
            axis=1,
        )
        height = concat_img.shape[0]
        width = concat_img.shape[1]
        cv2.imshow("GoseloEnv/GoseloImg", cv2.resize(concat_img, (int(width * 8), int(height * 8))))
        cv2.waitKey(sleep_ms)


if __name__ == "__main__":
    parser = Trainer.get_argument()
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--field-size", type=int, default=2)
    parser.add_argument("--sleep-ms", type=int, default=int(1000 / 30))  # 30FPS
    args = parser.parse_args()

    logger = initialize_logger(logging_level=logging.INFO, save_log=False)

    # Load dataset
    x_train, y_train, x_test, y_test, output_dim = load_dataset(
        max_size=args.max_size, dataset_dir=args.dataset_dir, dataset_name_prefix=args.dataset_prefix, split_ratio=0
    )

    visualize_data(x_train, sleep_ms=args.sleep_ms)
