# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import logging

import tqdm
from tf2rl.algos.sac import SAC

from examples.config import get_config, get_config_argument
from examples.rl.evaluate_generalization import load_policy
from examples.rl.run_sac_waypoints_generator import load_way_points_generator
from safety_rl.envs.rl_env_way_points_generator_wrapper import RLEnvWayPointsGeneratorEvalWrapper
from safety_rl.envs.rl_interface_env import RLInterfaceEnv
from safety_rl.experiments.rl_trainer import RLTrainer
from safety_rl.misc.logger import initialize_logger


def evaluate_update_freq(env, policy, n_episodes=100, max_steps=300, show_progress=False):
    n_update = 0
    n_steps = 0
    for _ in tqdm.tqdm(range(n_episodes)):
        obs = env.reset()
        for step in range(max_steps):
            action = policy.get_action(obs, test=True)
            next_obs, _, done, info = env.step(action)
            if show_progress:
                env.render()
            n_update += int(info["update_reference"])
            if done:
                n_steps += step
                break
            obs = next_obs
    return n_steps / n_update


if __name__ == "__main__":
    parser = RLTrainer.get_argument()
    parser = SAC.get_argument(parser)
    parser = get_config_argument(parser)
    parser.set_defaults(episode_max_steps=300)
    parser.set_defaults(model_dir="iros_data/pillars/ours/doggo/1")
    parser.add_argument(
        "--way-points-model-dir",
        type=str,
        default="iros_data/way_points_model/pillars",
        help="Path to pre-trained way points generator directory",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--sac-units", type=int, default=256)
    args = parser.parse_args()

    logger = initialize_logger(logging_level=logging.INFO, save_log=False)

    input_shape = (64, 64, 6)
    output_dim = 2 * 10
    logger.info("NN output is {} way points".format(int(output_dim / 2)))

    # load model
    way_points_generator = load_way_points_generator(args.way_points_model_dir, input_shape, output_dim)

    # env
    config, _ = get_config(
        robot_type="doggo",
        field_size=2.0,
        pillars_num=10,
        observe_pillars=True,
        hazards_num=0,
        observe_hazards=False,
        pillars_order=True,
    )
    EvalEnvClass = RLEnvWayPointsGeneratorEvalWrapper

    kwargs_env = {"coef_collision_penalty": float(not args.remove_lidar) * (-1.0), "input_raw_way_points": True}

    kwargs_test_env = copy.deepcopy(kwargs_env)
    kwargs_test_env["env"] = RLInterfaceEnv(config=config, visualize_way_points=True)
    kwargs_test_env["way_points_generator"] = way_points_generator

    env = EvalEnvClass(**kwargs_test_env)

    # test environment by visualization
    if args.debug:
        print("obs shape: {}, act_shape: {}".format(env.observation_space.shape, env.action_space.shape))
        for _ in range(100):
            env.reset()
            env.render()
            for _ in range(100):
                _, _, done, _ = env.step(env.action_space.sample())
                env.render()
                if done:
                    break
        exit()

    # policy
    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=0,
        max_action=env.action_space.high[0],
    )
    policy = load_policy(policy, args.model_dir)

    update_freq = evaluate_update_freq(env, policy)
    print(update_freq)
