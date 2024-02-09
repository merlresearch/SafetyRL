<!--
Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Safety-RL

This is a codebase used for our IROS'20 paper [Efficient Exploration in Constrained Environments with Goal-Oriented Reference Path](https://arxiv.org/abs/2003.01641).

## Installation

### Pre-requirements

```bash
# safety-gym from keiohta's forked version
$ git clone git@github.com:keiohta/safety-gym.git
$ cd safety-gym
$ pip install -e .
```

### MuJoCo

```bash
$ mkdir -p ~/.mujoco
$ cd ~/.mujoco
$ wget https://www.roboti.us/download/mujoco200_linux.zip
$ unzip mujoco200_linux.zip
$ mv mujoco200_linux mujoco200

# Extend LD_LIBRARY_PATH with mujoco:
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin

# Now install mujoco-py via pip:
$ pip install mujoco-py
```

### Add python path

```bash
$ cd safety_rl
$ pip install -r requirements.txt
$ export PYTHONPATH=$PYTHONPATH:$PWD
```

## Examples

### Train way-points generator

#### Supervised learning with previously collected data

##### Generate optimal path using A*

Specify following information

- `--hazards-num`: number of hazards to locate
- `--field-size`: define min-max size of field. if you specify `1`, then the size will be `config[placements_extents] = [-1, -1, 1, 1]`. default is `2`.
- `--resolution`: resolution to plan a path. default is `0.1`.

```bash
$ python examples/generate_optimal_path.py
```

##### Generate supervised learning dataset

Generate a dataset for training the waypoints generator

```bash
$ python examples/generate_dataset.py --save-data --dataset-size 50000
$ python examples/generate_dataset.py --save-data --dataset-size 10000 --evaluate
```

##### Train with supervised learning

Train a model with dataset generated above.

```bash
$ python examples/train_cnn.py --epochs 100 --lr 0.0001

# Evaluate trained model
$ python examples/train_cnn.py --rollout-only --show-test-progress
```

#### Train way-points generator RL-like

```bash
$ python examples/train_cnn_rl_like.py --n-warm-up=10000 --show-test-progress --test-env-interval 10000
```

### Train SAC agent

```bash
$ python examples/rl/run_sac_waypoints_generator.py
$ python examples/rl/run_sac_waypoints_generator.py --evaluate --model-dir /path/to/results --test-episodes 10 --show-test-progress
```

### Test

```bash
$ python -m unittest discover -v
```

## Reproduce paper results

### Generate waypoints generator dataset

Generate the following datasets for training waypoints generators

- `pillars_2_10`: for Exp. 6.A, 6.B
- `pillars_3_25`: for Exp. 6.C
- `pillars_4_40`: for Exp. 6.C
- `gremlins_2_10`: for Exp. 6.C
- `two_room`: for Exp. 6.C
- `four_room`: for Exp. 6.C

```bash
$ python examples/all_generate_dataset.py --run
```

### Train waypoints generator models

```bash
$ python examples/all_train_cnn.py --run
```

### Train all RL models

```bash
# "ours" on MCS
$ python examples/rl/all_envs_ours.py --run

# "baseline" on MCS
$ python examples/rl/all_envs_baseline.py --run
```

Make graphs that show learning curves


```bash
$ python examples/rl/make_compare_graph.py -i ../safetyrl_results/dataset/pillars_2_10 --legend --color
```

Visually evaluate the trained model

```bash
$ python examples/rl/run_sac_waypoints_generator.py --evaluate --root-dir ../safetyrl_results/ --show-test-progress --robot-type doggo
```

### Generalization

Qualitatively evaluate the trained model

```bash
# Evaluate the performance of trained model on various environments on MCS
$ python examples/rl/evaluate_generalization.py
```

Visually evaluate the trained model

```bash
# pillars (3, 3, 25)
$ python examples/rl/run_sac_waypoints_generator.py --evaluate --root-dir ../safetyrl_results/ --show-test-progress --robot-type doggo --fine-tuning --field-size 3 --pillars-num 25

# pillars (4, 4, 40)
$ python examples/rl/run_sac_waypoints_generator.py --evaluate --root-dir ../safetyrl_results/ --show-test-progress --robot-type doggo --fine-tuning --field-size 4 --pillars-num 40

# two-room
$ python examples/rl/run_sac_waypoints_generator.py --evaluate --root-dir ../safetyrl_results/ --show-test-progress --robot-type doggo --fine-tuning --place-room --room-type 0

# gremlin
$ python examples/rl/run_sac_waypoints_generator.py --evaluate --root-dir ../safetyrl_results/ --show-test-progress --robot-type doggo --fine-tuning --dummy-gremlins --gremlins-num 10
```

## Citation

If you use the software, please cite the following ([TR2020-141](https://merl.com/publications/TR2020-141)):

```BibTeX
@inproceedings{ota2020efficient
    author = {Ota, Kei and Sasaki, Yoko and Jha, Devesh K and Yoshiyasu, Yusuke and Kanezaki, Asako},
    title = {Efficient exploration in constrained environments with goal-oriented reference path},
    booktitle = {2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    year = {2020},
    pages = {6061--6068},
    publisher = {IEEE},
    doi = {10.1109/IROS45743.2020.9341620},
    url = {https://ieeexplore.ieee.org/abstract/document/9341620}
}
```

## Contact

Please contact Devesh Jha at jha@merl.com

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (C) 2021, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
