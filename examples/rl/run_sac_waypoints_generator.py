# Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import logging
import os

import tensorflow as tf
from tf2rl.algos.sac import SAC

from examples.config import DEFAULT_EXP_NAME, get_config, get_config_argument, get_policy_model_directories
from safety_rl.algos.waypoints_generator import WayPointsGenerator
from safety_rl.envs.rl_env_periodical_way_points_update import (
    RLEnvWayPointsGeneratorPeriodicalUpdateEvalWrapper,
    RLEnvWayPointsGeneratorPeriodicalUpdateWrapper,
)
from safety_rl.envs.rl_env_way_points_generator_wrapper import (
    RLEnvWayPointsGeneratorEvalWrapper,
    RLEnvWayPointsGeneratorWrapper,
    VisOnlyInputEnvWrapper,
)
from safety_rl.envs.rl_interface_env import RLInterfaceEnv
from safety_rl.experiments.rl_trainer import RLTrainer
from safety_rl.misc.logger import initialize_logger


def load_way_points_generator(way_points_generator_dir, input_shape, output_dim):
    logger = logging.getLogger("safety_rl")
    assert os.path.isdir(way_points_generator_dir)
    way_points_generator = WayPointsGenerator(input_shape=input_shape, output_dim=output_dim)
    checkpoint = tf.train.Checkpoint(way_points_generator=way_points_generator.model)
    latest_path_ckpt = tf.train.latest_checkpoint(way_points_generator_dir)
    checkpoint.restore(latest_path_ckpt)
    logger.info("Restored {}".format(latest_path_ckpt))
    return way_points_generator


if __name__ == "__main__":
    parser = RLTrainer.get_argument()
    parser = SAC.get_argument(parser)
    parser = get_config_argument(parser)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=10000)
    parser.set_defaults(episode_max_steps=300)
    parser.set_defaults(test_episodes=20)
    parser.set_defaults(max_steps=int(5e6))
    parser.add_argument(
        "--waypoints-model-dir", type=str, default=None, help="Path to pre-trained way points generator directory"
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--visualize-only-input", action="store_true")
    parser.add_argument("--input-cnn-feature", action="store_true")
    parser.add_argument("--sac-units", type=int, default=256)
    parser.add_argument("--periodically-update-way-points", action="store_true")
    parser.add_argument("--update-interval", type=int, default=2)
    parser.add_argument("--root-dir", type=str, default=".")
    parser.add_argument("--fine-tuning", action="store_true")
    args = parser.parse_args()

    logger = initialize_logger(logging_level=logging.INFO, save_log=False)

    # Settings
    config, exp_type = get_config(
        robot_type=args.robot_type,
        field_size=args.field_size,
        no_obs=args.no_obs,
        place_room=args.place_room,
        room_type=args.room_type,
        hazards_num=args.hazards_num,
        pillars_num=args.pillars_num,
        dummy_gremlins=args.dummy_gremlins,
        gremlins_num=args.gremlins_num,
    )
    if args.waypoints_model_dir is not None:
        waypoints_model_dir = args.way_points_model_dir
        logdir = waypoints_model_dir
    else:
        # e.g. "dataset/pillars_2_10/ours/point"
        exp_type = exp_type if args.fine_tuning else DEFAULT_EXP_NAME
        logdir = os.path.join(args.root_dir, "dataset", exp_type, "ours", args.robot_type)
        waypoints_model_dir = os.path.join(args.root_dir, "dataset", exp_type, "waypoints_generator", "model")

    if args.evaluate and args.model_dir is None:
        args.model_dir = get_policy_model_directories(
            args.root_dir, exp_type=DEFAULT_EXP_NAME, method="ours", robot_type=args.robot_type
        )[0]
        args.dir_suffix = "eval"

    input_shape = (64, 64, 6)
    output_dim = 2 * 10
    logger.info("NN output is {} way points".format(int(output_dim / 2)))

    # load model
    args.logdir = logdir
    way_points_generator = load_way_points_generator(waypoints_model_dir, input_shape, output_dim)

    # env
    EnvClass = (
        RLEnvWayPointsGeneratorPeriodicalUpdateWrapper
        if args.periodically_update_way_points
        else RLEnvWayPointsGeneratorWrapper
    )
    EvalEnvClass = (
        RLEnvWayPointsGeneratorPeriodicalUpdateEvalWrapper
        if args.periodically_update_way_points
        else RLEnvWayPointsGeneratorEvalWrapper
    )

    kwargs_env = {
        "coef_collision_penalty": float(not args.remove_lidar) * (-1.0),
        "input_raw_way_points": True,
        "input_cnn_feature": args.input_cnn_feature,
    }

    if args.periodically_update_way_points:
        kwargs_env["update_interval"] = args.update_interval

    kwargs_test_env = copy.deepcopy(kwargs_env)
    kwargs_env["env"] = RLInterfaceEnv(config=config)
    kwargs_test_env["env"] = RLInterfaceEnv(config=config, visualize_way_points=True)
    kwargs_env["way_points_generator"] = way_points_generator
    kwargs_test_env["way_points_generator"] = way_points_generator

    env = EnvClass(**kwargs_env)
    test_env = EvalEnvClass(**kwargs_test_env)

    if args.visualize_only_input:
        test_env = VisOnlyInputEnvWrapper(test_env)

    # test environment by visualization
    if args.debug:
        print("obs shape: {}, act_shape: {}".format(env.observation_space.shape, env.action_space.shape))
        for _ in range(100):
            test_env.reset()
            test_env.render()
            print(
                test_env.robot_pos,
                test_env.goal_pos,
                test_env._raw_reference_path_world_coord,
                test_env._reference_path_world_coord.shape,
            )
            for _ in range(100):
                _, _, done, _ = test_env.step(test_env.action_space.sample())
                test_env.render()
                if done:
                    break
        exit()

    # policy
    policy = SAC(
        state_shape=env.observation_space.shape,
        action_dim=env.action_space.high.size,
        gpu=args.gpu,
        actor_units=[args.sac_units, args.sac_units],
        critic_units=[args.sac_units, args.sac_units],
        memory_capacity=args.memory_capacity,
        max_action=env.action_space.high[0],
        batch_size=args.batch_size,
        n_warmup=args.n_warmup,
        alpha=args.alpha,
        auto_alpha=args.auto_alpha,
    )

    # trainer
    trainer = RLTrainer(policy, env, args, test_env=test_env)
    if args.evaluate:
        trainer.evaluate_policy_continuously()
    else:
        trainer()
