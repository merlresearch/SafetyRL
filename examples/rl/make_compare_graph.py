# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import glob
import logging
import os
import re
import struct

import matplotlib.pylab as plt
import numpy as np
from scipy.interpolate import interp1d
from tensorboard.backend.event_processing.event_accumulator import DEFAULT_SIZE_GUIDANCE, TENSORS, EventAccumulator


def get_score_trajectory_from_tfsummary(path_file, query="Common/average_test_return"):
    assert os.path.exists(path_file)

    size_guidance = DEFAULT_SIZE_GUIDANCE.copy()
    size_guidance[TENSORS] = 0

    accumulator = EventAccumulator(path_file, size_guidance=size_guidance)
    accumulator.Reload()

    data = accumulator.Tensors(query)

    steps = [tmp.step for tmp in data]
    scores = [struct.unpack("f", tmp.tensor_proto.tensor_content)[0] for tmp in data]

    return steps, scores


def aggregate_actual_scores(list_file, query):
    list_trials = []

    for cur_file in list_file:
        try:
            steps, scores = get_score_trajectory_from_tfsummary(cur_file, query)
        except KeyError:
            continue

        list_trials.append((steps, scores))

    return list_trials


def average_actual_scores(list_scores, sampling_interval=20000, max_steps=0, allow_interpolate=True, force=False):
    """

    :return: (step_axis, mean_scores, error_values, all_scores)

    step_axis : グラフのx軸の各値
    mean_scores : step_axisに対応するスコアの平均値
    error_values : step_axisに対応するスコアのぶんさん
    all_scores: mean_scoresとerror_valuesの元になったすべてのスコア値
    """
    logger = logging.getLogger(__name__)

    if max_steps == 0:
        for steps, scores in list_scores:
            print(np.min(steps))
            if np.max(steps) > max_steps:
                max_steps = np.max(steps)

    new_list_scores = []

    for steps, scores in list_scores:
        min_steps = np.min(steps)
        if np.max(steps) >= max_steps:
            new_steps = []
            new_scores = []
            for cur_step, cur_score in zip(steps, scores):
                if cur_step > max_steps:
                    break

                new_steps.append(cur_step)
                new_scores.append(cur_score)

            steps = np.array(new_steps)
            scores = np.array(new_scores)

            new_list_scores.append((steps, scores))
        else:
            logger.info("skip experiment length {}".format(np.max(steps)))

    list_scores = new_list_scores

    if len(list_scores) == 0:
        logger.warning("skip incompleted experiment")
        return None

    mean_scores = []
    errors = []
    all_scores = []
    sampling_interval = max(sampling_interval, steps[1] - steps[0])
    step_axis = np.arange(min_steps, max_steps + 1, step=sampling_interval)
    # step_axis = np.arange(sampling_interval, max_steps + 1, step=sampling_interval)

    if allow_interpolate:
        list_functions = []

        for steps, scores in list_scores:
            list_functions.append(interp1d(steps, scores))

        for cur_step in step_axis:
            values = [f(cur_step) for f in list_functions]
            mean_value = np.mean(values)
            mean_scores.append(mean_value)
            all_scores.append(values)
    else:
        list_dicts = []

        for steps, scores in list_scores:
            cur_dict = {}
            for cur_step, cur_score in zip(steps, scores):
                cur_dict[cur_step] = cur_score
            list_dicts.append(cur_dict)

        for cur_step in step_axis:
            try:
                values = [d[cur_step] for d in list_dicts]
                all_scores.append(values)
                mean_value = np.mean(values)
                error_value = np.std(values)
                mean_scores.append(mean_value)
                errors.append(error_value)
            except KeyError:
                logger.error("Key {} does not exist".format(cur_step))
                continue

    mean_scores = np.array(mean_scores)
    error_values = np.array(errors)
    all_scores = np.array(all_scores).transpose()

    return step_axis, mean_scores, error_values


def enumerate_flat(root):
    """
    iros_data/complete_data_1m/
    ├── baseline
    │   ├── car
    │   │   ├── 1
    │   │   │   └── 20200222T133206.948857_SAC_
    │   │   ├── 2
    │   │   │   └── 20200222T133211.700154_SAC_
    |   |   ...
    │   ├── doggo
    |    ...
    ...
    └── ours
        ├── car
        │   ├── 1
        │   │   └── 20200222T161159.521367_SAC_
        │   ├── 2
        │   │   └── 20200222T161158.678132_SAC_
        ...
    Args:
        root: Root directory (iros_data/complete_data_1m/)

    Returns:

    """
    # key=> method_name, value => list_file
    dict_methods = {}
    for exp_type in ["baseline", "ours"]:
        list_file = glob.glob("{}/{}/*/*/*/*tfevents*".format(root, exp_type), recursive=True)
        if len(list_file) == 0:
            list_file = glob.glob("{}/{}/*/*/*tfevents*".format(root, exp_type), recursive=True)

        for cur_file in list_file:
            names = cur_file.split("/")
            env_name = os.path.relpath(cur_file, start=root).split("/")[1]
            name = "{}_{}".format(exp_type, env_name)

            if name not in dict_methods:
                dict_methods[name] = []

            dict_methods[name].append(cur_file)

    return dict_methods


def main():
    logging.basicConfig(
        datefmt="%d/%Y %I:%M:%S",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)s) %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", dest="dir_log", required=True)
    parser.add_argument("--max_steps", default=int(1e6), type=int)
    parser.add_argument("--color", action="store_true", default=False)
    parser.add_argument("--legend", action="store_true", default=False)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    max_steps = args.max_steps

    dict_methods_tf_summary = enumerate_flat(args.dir_log)
    dict_color = {
        "ours_point": ("blue", "-"),
        "baseline_point": ("steelblue", "-"),
        "ours_doggo": ("red", "-"),
        "baseline_doggo": ("orange", "-"),
        "ours_car": ("lime", "-"),
        "baseline_car": ("darkgreen", "-"),
    }
    random_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

    figsize = (8, 5)

    method_order = ["ours_point", "baseline_point", "ours_car", "baseline_car", "ours_doggo", "baseline_doggo"]

    queries = [
        ["Common/average_test_return", "average return"],
        ["Common/average_n_collision", "average times to collide"],
        ["Common/average_reach_goal", "average goal reach rate"],
        ["Common/average_steps_to_reach_goal", "average steps to reach goal"],
        ["Common/hazards_cost", "average hazards cost"],
    ]

    for idx in range(len(queries)):
        query, label = queries[idx]
        logger.info("Creating figure: {}".format(label))

        plt.close()
        plt.figure(figsize=figsize, dpi=300)

        for idx, method in enumerate(method_order):
            if args.color and method not in dict_color:
                print("Detected unknown method name {}. Skip this.".format(method))
                continue

            try:
                list_file = dict_methods_tf_summary[method]
            except KeyError:
                logger.warning("Cannot find {}. Skip it.".format(method))
                continue
            logger.info("analyze {} has {} seeds".format(method, len(list_file)))

            list_trials = aggregate_actual_scores(list_file, query)

            cur_steps, cur_mean_scores, cur_errors = average_actual_scores(
                list_scores=list_trials,
                max_steps=max_steps,
                sampling_interval=30000,
                allow_interpolate=False,
                force=args.force,
            )

            if args.color:
                color, line_style = dict_color[method]
            else:
                color = random_colors[idx % len(random_colors)]
                line_style = "-"

            cur_steps = cur_steps / 1000000.0

            method_name = method.replace("_", "/")
            plt.plot(cur_steps, cur_mean_scores, color=color, label=method_name, linewidth=1.0, linestyle=line_style)
            # https://matplotlib.org/gallery/recipes/fill_between_alpha.html

            plt.fill_between(
                cur_steps, cur_mean_scores - cur_errors, cur_mean_scores + cur_errors, facecolor=color, alpha=0.2
            )

        font_size = 12
        plt.xlabel("million steps", fontsize=font_size)
        plt.ylabel(label, fontsize=font_size)

        plt.tick_params(labelsize=font_size)
        plt.grid(which="major", color="black", linestyle="-", alpha=0.15)
        plt.grid(which="minor", color="black", linestyle="-", alpha=0.15)

        if args.legend:
            # legend = plt.legend(loc='lower right', borderaxespad=0, fontsize=15)
            legend = plt.legend(fontsize=font_size, framealpha=1.0)
            fig = legend.figure
            fig.canvas.draw()
            bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig("legend.png", dpi="figure", bbox_inches=bbox)

        filename = "results_{}_{:.1f}M_steps.png".format(label.replace(" ", "_"), max_steps / 1e6)
        plt.savefig(filename, figsize=figsize, dpi=300)


if __name__ == "__main__":
    main()
