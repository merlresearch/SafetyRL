# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def orig():
    labels = ["G1", "G2", "G3", "G4", "G5"]
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(labels))  # the label locations
    width = 0.5  # the width of the bars

    plt.figure(figsize=(8, 5), dpi=300)

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, men_means, width, label="Men")
    rects2 = ax.bar(x + width / 2, women_means, width, label="Women")

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Scores")
    ax.set_title("Scores by group and gender")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    plt.show()


def bar_plot(labels, values, score_label, title):
    x = np.arange(len(labels))
    n_exp = len(values.items())
    # print(labels, values, score_label, title)
    # print(len(values.items()))
    # exit()

    width_all = 0.5
    width_each = width_all / n_exp

    fig, ax = plt.subplots()
    pos = np.linspace(-width_all / 2.0, width_all / 2.0, n_exp)
    print(title)
    for idx, value in enumerate(values.items()):
        label, data = value
        mean = np.array(data)[:, 0]
        std = np.array(data)[:, 1]
        print(label, mean, std)
        ax.bar(x + pos[idx], mean, width_each, yerr=std, label=label)

    if score_label == "average_return":
        label = "average return"
    elif score_label == "average_steps":
        label = "average steps to reach goal"
    elif score_label == "average_reach_goal":
        label = "average goal reach rate"

    ax.set_ylabel(label)
    # ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    # ax.legend()

    fig.tight_layout()

    # legend = plt.legend(fontsize=16, framealpha=1.0)
    # fig = legend.figure
    # fig.canvas.draw()
    # bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig("legend.png", dpi="figure", bbox_inches=bbox)

    plt.savefig("results_generalization_{}.png".format(title.replace(" ", "_")))
    plt.close()


def main():
    # orig()
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument(
        "--root-dir", type=str, default="iros_data/pillars", help="Root directory under which tf models are saved"
    )
    args = parser.parse_args()

    # result_file_path = os.path.join(args.root_dir, "results_only_sequnce_edited.csv")
    result_file_path = os.path.join(args.root_dir, "results_only_sequence_changed_with_std.csv")
    assert os.path.exists(result_file_path), "{} does not exist".format(result_file_path)

    results_env_dict = {"doggo": {}}
    score_labels = ["average_return", "average_reach_goal", "average_steps"]

    method_names = []
    with open(result_file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            method, robot_type, str_finetune, exp_type, *scores = row
            if exp_type == "size_2_pillars_10":
                continue
            method += "_finetune" if str_finetune == "finetune" else ""
            scores = [float(score) for score in scores]
            # average_return, average_reach_goal, average_steps, average_n_collision = scores
            if not exp_type in results_env_dict[robot_type]:
                results_env_dict[robot_type][exp_type] = {}
            if not method in method_names:
                method_names.append(method)
            results_env_dict[robot_type][exp_type][method] = scores

    # Create graph for robot_type(3) * exp_type(5) = 15
    for robot_type in results_env_dict:  # [point, car, doggo]
        for idx, score_label in enumerate(score_labels):
            labels = []
            values = {}
            for method_name in method_names:
                values[method_name] = []

            for exp_type in results_env_dict[
                robot_type
            ]:  # [size_2_hazards_10, size_3_hazards_25, size_4_hazards_40, two_room, four_room]
                if exp_type == "size_2_pillars_10":
                    labels.append("pillar (2, 2, 10)")
                elif exp_type == "size_3_pillars_25":
                    labels.append("pillar (3, 3, 25)")
                elif exp_type == "size_4_pillars_40":
                    labels.append("pillar (4, 4, 40)")
                elif exp_type == "size_2_gremlins_10":
                    labels.append("gremlin")
                elif "room" in exp_type:
                    labels.append(exp_type.replace("_room", "-room"))
                else:
                    raise NotImplementedError("Unexpected exp_type: {} came".format(exp_type))
                print(score_label, exp_type, results_env_dict[robot_type][exp_type][method_name])
                for method_name in method_names:
                    values[method_name].append(
                        [
                            results_env_dict[robot_type][exp_type][method_name][2 * idx],  # mean
                            results_env_dict[robot_type][exp_type][method_name][2 * idx + 1],
                        ]
                    )  # std
            title = "{} {}".format(robot_type, score_label)
            bar_plot(labels, values, score_label, title)


if __name__ == "__main__":
    main()
