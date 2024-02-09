# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import gc
import logging
import multiprocessing
import os
from multiprocessing import Process

# https://sefiks.com/2019/03/20/tips-and-tricks-for-gpu-and-multiprocessing-in-tensorflow/
multiprocessing.set_start_method("spawn", force=True)

import numpy as np
import tensorflow as tf
from tf2rl.algos.sac import SAC

from examples.config import (
    DEFAULT_EXP_NAME,
    get_all_exp_settings,
    get_config,
    get_config_argument,
    get_policy_model_directories,
)
from examples.rl.run_sac_waypoints_generator import load_way_points_generator
from safety_rl.envs.engine_wrapper import EngineWrapper
from safety_rl.envs.rl_env_way_points_generator_wrapper import RLEnvWayPointsGeneratorEvalWrapper
from safety_rl.envs.rl_env_wrapper import RLEnvWrapper
from safety_rl.envs.rl_interface_env import RLInterfaceEnv
from safety_rl.misc.logger import initialize_logger
from safety_rl.misc.task_manager import detect_num_gpu


def import_tf():
    import tensorflow as tf

    if tf.config.experimental.list_physical_devices("GPU"):
        for cur_device in tf.config.experimental.list_physical_devices("GPU"):
            print(cur_device)
            tf.config.experimental.set_memory_growth(cur_device, enable=True)
    return tf


def check_validity_of_root_dir(root_dir, method_types, robot_types, env_settings):
    for method in method_types:
        for robot_type in robot_types:
            # Check if trained SAC model exists
            logdir = os.path.join(root_dir, "dataset", DEFAULT_EXP_NAME, method.replace("_finetune", ""))
            model_directories = get_policy_model_directories(logdir, robot_type)
            assert len(model_directories) > 0, "Cannot find any SAC model in {}".format(logdir)

            # Check if waypoints model exists when specifying "ours_finetuning"
            if method == "ours_finetune":
                for env_setting in env_settings:
                    config, exp_type = get_config(robot_type=robot_type, **env_setting["kwargs"])
                    waypoints_model_dir = os.path.join(root_dir, "dataset", exp_type, "waypoints_generator", "model")
                    assert os.path.isdir(waypoints_model_dir), "Cannot find {}".format(waypoints_model_dir)


def evaluate_policy(policy, env, n_episodes, episode_max_steps, show_test_progress=False):
    test_returns = []
    reach_goals = []
    episode_steps = []

    n_evaluated_episodes = 0

    while n_evaluated_episodes < n_episodes:
        episode_return = 0.0
        obs = env.reset()
        irregular_done = False
        reached_goal = False
        for n_transition in range(episode_max_steps):
            action = policy.get_action(obs, test=True)
            next_obs, reward, done, info = env.step(action)

            if show_test_progress:
                env.render()

            episode_return += reward
            obs = next_obs
            if "exceed_limit" in info and info["exceed_limit"] is True:
                irregular_done = True
                break
            if done:
                if info["goal_met"]:
                    reached_goal = True
                break
        if irregular_done:
            continue
        test_returns.append(episode_return)
        reach_goals.append(int(reached_goal))
        episode_steps.append(n_transition)
        n_evaluated_episodes += 1
    test_returns = np.array(test_returns)
    reach_goals = np.array(reach_goals)
    episode_steps = np.array(episode_steps)

    return (
        np.mean(test_returns),
        np.std(test_returns),
        np.mean(reach_goals),
        np.std(reach_goals),
        np.mean(episode_steps),
        np.std(episode_steps),
    )


def load_policy(policy, policy_dir):
    logger = logging.getLogger("safety_rl")
    assert os.path.isdir(policy_dir)

    # Restore model
    checkpoint = tf.train.Checkpoint(policy=policy)

    latest_path_ckpt = tf.train.latest_checkpoint(policy_dir)
    checkpoint.restore(latest_path_ckpt)
    logger.info("Restored {}".format(latest_path_ckpt))
    return policy


def evaluate_core(
    method,
    env_setting,
    robot_type,
    root_dir,
    n_episodes,
    results_path,
    input_shape=(64, 64, 6),
    output_dim=2 * 10,
    show_test_progress=False,
    gpu=0,
):
    import_tf()
    logger = initialize_logger(save_log=False)

    config, exp_type = get_config(robot_type=robot_type, **env_setting["kwargs"])

    if method == "baseline":
        env = RLEnvWrapper(env=EngineWrapper(config=config))
    else:
        if "finetune" in method:
            way_points_model_dir = os.path.join(root_dir, "dataset", exp_type, "waypoints_generator", "model")
        else:
            way_points_model_dir = "{}/dataset/pillars_2_10/waypoints_generator/model".format(root_dir)
        way_points_generator = load_way_points_generator(way_points_model_dir, input_shape, output_dim)
        env = RLEnvWayPointsGeneratorEvalWrapper(
            way_points_generator,
            RLInterfaceEnv(config=config, visualize_way_points=True),
            input_raw_way_points=True,
            input_cnn_feature=False,
        )

    # policy
    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=gpu,
        memory_capacity=int(1e6),
        max_action=env.action_space.high[0],
        batch_size=100,
    )

    model_directories = get_policy_model_directories(root_dir, DEFAULT_EXP_NAME, method, robot_type)
    policy = load_policy(policy, model_directories[0])

    episode_max_steps = int(env_setting["kwargs"]["field_size"] * 150)
    if "room" in env_setting["name"]:
        episode_max_steps *= 2

    logger.info("Start evaluating {}/{}".format(method, robot_type))
    result = evaluate_policy(
        policy, env, n_episodes=n_episodes, episode_max_steps=episode_max_steps, show_test_progress=show_test_progress
    )
    if not show_test_progress:
        str_finetune = "finetune" if "finetune" in method else "normal"
        with open(results_path, "a") as f:
            f.write(
                "{},{},{},{},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n".format(
                    method.replace("_finetune", ""),
                    robot_type,
                    str_finetune,
                    exp_type["name"],
                    result[0],
                    result[1],
                    result[2],
                    result[3],
                    result[4],
                    result[5],
                )
            )


def main():
    parser = SAC.get_argument()
    # parser.add_argument("--is-pillar", action="store_true")
    parser.add_argument("--force-hazards", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--max-process", type=int, default=None)
    parser.add_argument("--root-dir", type=str, default="../safetyrl_results")
    parser = get_config_argument(parser)
    args = parser.parse_args()

    root_dir = args.root_dir
    initialize_logger(save_log=False)

    # method_types = ["baseline", "ours", "ours_finetune"]
    method_types = ["ours", "ours_finetune"]
    # robot_types = ["point", "car", "doggo"]
    robot_types = ["doggo"]

    settings = get_all_exp_settings(is_cmd_args=False)
    check_validity_of_root_dir(root_dir, method_types, robot_types, settings)

    results = {}

    n_episodes = args.n_episodes if not args.debug else 3
    results_path = os.path.join(root_dir, "results.csv")

    tasks = []

    n_gpus = detect_num_gpu()

    gpu_idx = 0
    for method in method_types:
        for robot_type in robot_types:
            results["{}_{}".format(method, robot_type)] = {}
            for env_setting in settings:
                kwargs = {
                    "method": method,
                    "env_setting": env_setting,
                    "robot_type": robot_type,
                    "root_dir": root_dir,
                    "n_episodes": n_episodes,
                    "results_path": results_path,
                    "gpu": int(gpu_idx % n_gpus),
                }
                if args.debug:
                    kwargs["show_test_progress"] = True
                    evaluate_core(**kwargs)
                else:
                    tasks.append(Process(target=evaluate_core, kwargs=kwargs))

                gpu_idx += 1

    if args.debug:
        return None

    n_tasks = len(tasks)
    max_process = args.max_process if args.max_process is not None else multiprocessing.cpu_count()
    for i in range(int(n_tasks / max_process) + 1):
        n_process = min(max_process, n_tasks - max_process * i)
        for j in range(n_process):
            gpu_idx = i * max_process + j
            tasks[gpu_idx].start()
        for j in range(n_process):
            gpu_idx = i * max_process + j
            tasks[gpu_idx].join()
        gc.collect()


if __name__ == "__main__":
    main()
