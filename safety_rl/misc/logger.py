# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import datetime
import logging
import os
from logging import FileHandler, Formatter, StreamHandler, getLogger


def initialize_logger(logging_level=logging.INFO, output_dir="results/", filename=None, save_log=True):
    logger = getLogger("safety_rl")
    logger.setLevel(logging_level)

    handler_format = Formatter(
        "%(asctime)s.%(msecs)03d [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s", "%H:%M:%S"
    )  # If you want to show date, add %Y-%m-%d
    stream_handler = StreamHandler()
    stream_handler.setLevel(logging_level)
    stream_handler.setFormatter(handler_format)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if save_log:
        if filename is not None:
            filename = filename
        else:
            filename = os.path.join(output_dir, datetime.datetime.now().strftime("%Y%m%dT%H%M%S.%f") + ".log")

        file_handler = FileHandler(filename, "a")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(handler_format)

    if len(logger.handlers) == 0:
        logger.addHandler(stream_handler)
        if save_log:
            logger.addHandler(file_handler)
    else:
        # Overwrite logging setting
        logger.handlers[0] = stream_handler
        if save_log and len(logger.handlers) > 1:
            logger.handlers[1] = file_handler

    logger.propagate = False

    return logger
