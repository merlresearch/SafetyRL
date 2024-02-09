# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import glob
import logging
import os

import matplotlib.pylab as plt

from examples.rl.make_compare_graph import aggregate_actual_scores, average_actual_scores


def enumerate_flat(root):
    """
    iros_data/compare_frequency/
    ├── 16
    │   ├── 0
    │   │   └── 20200223T143518.542726_SAC_
    │   └── 1
    │       └── 20200223T143522.719885_SAC_
    ├── 32
    │   ├── 0
    │   │   └── 20200223T143520.174526_SAC_
    │   └── 1
    │       └── 20200223T143523.325624_SAC_
    ...
    └── 8
        ├── 0
        │   └── 20200223T143516.512919_SAC_
        └── 1
            └── 20200223T143522.308663_SAC_
        ...
    Args:
        root: Root directory (iros_data/complete_data_1m/)

    Returns:

    """
    # key=> method_name, value => list_file
    dict_methods = {}

    list_file = glob.glob("{}/*/*/*/*tfevents*".format(root), recursive=True)

    for cur_file in list_file:
        name = os.path.relpath(cur_file, start=root).split("/")[0]

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
    parser.add_argument("--dir-log", type=str, default="iros_data/pillars/compare_frequency_no_lidar")
    parser.add_argument("--max-steps", default=0, type=int)
    parser.add_argument("--color", action="store_true", default=False)
    parser.add_argument("--legend", action="store_true", default=False)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    max_steps = args.max_steps

    dict_methods_tf_summary = enumerate_flat(args.dir_log)

    random_colors = ["orange", "g", "m", "b", "c", "r", "k", "w"]

    figsize = (8, 5)

    # method_order = ["4", "8", "16", "32", "64", "safe"]
    method_order = ["4", "16", "64", "ours (8.5)"]

    queries = [
        ["Common/average_test_return", "average return"],
        ["Common/average_n_collision", "average times to collide"],
        ["Common/average_reach_goal", "average goal reach rate"],
        ["Common/average_steps_to_reach_goal", "average steps to reach goal"],
    ]

    for query_idx in range(len(queries)):
        query, label = queries[query_idx]
        logger.info("Creating figure: {}".format(label))

        plt.close()
        plt.figure(figsize=figsize, dpi=300)

        # for method, list_file in dict_methods.items():
        for idx, method in enumerate(method_order):
            list_file = dict_methods_tf_summary[method]
            logger.info("analyze {} has {} seeds".format(method, len(list_file)))

            list_trials = aggregate_actual_scores(list_file, query)

            cur_steps, cur_mean_scores, cur_errors = average_actual_scores(
                list_scores=list_trials,
                max_steps=max_steps,
                sampling_interval=20000,
                allow_interpolate=False,
                force=args.force,
            )

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

        filename = "results_compare_freq_{}_{:.1f}M_steps.png".format(label.replace(" ", "_"), max_steps / 1e6)
        plt.savefig(filename, figsize=figsize, dpi=300)


if __name__ == "__main__":
    main()
