# A Unified Framework for Factorizing Distributional Value Functions for Multi-Agent Reinforcement Learning

[![arXiv](https://img.shields.io/badge/arXiv-2306.02430-b31b1b.svg)](https://arxiv.org/abs/2306.02430)<br>

| Super Hard | Ultra Hard |
|------------|------------|
| [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfac-framework-factorizing-the-value-function/smac-on-smac-6h-vs-8z-1)](https://paperswithcode.com/sota/smac-on-smac-6h-vs-8z-1?p=dfac-framework-factorizing-the-value-function)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfac-framework-factorizing-the-value-function/smac-on-smac-3s5z-vs-3s6z-1)](https://paperswithcode.com/sota/smac-on-smac-3s5z-vs-3s6z-1?p=dfac-framework-factorizing-the-value-function)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfac-framework-factorizing-the-value-function/smac-on-smac-mmm2-1)](https://paperswithcode.com/sota/smac-on-smac-mmm2-1?p=dfac-framework-factorizing-the-value-function)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfac-framework-factorizing-the-value-function/smac-on-smac-27m-vs-30m)](https://paperswithcode.com/sota/smac-on-smac-27m-vs-30m?p=dfac-framework-factorizing-the-value-function)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dfac-framework-factorizing-the-value-function/smac-on-smac-corridor)](https://paperswithcode.com/sota/smac-on-smac-corridor?p=dfac-framework-factorizing-the-value-function) | [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-framework-for-factorizing/smac-on-smac-6h-vs-9z)](https://paperswithcode.com/sota/smac-on-smac-6h-vs-9z?p=a-unified-framework-for-factorizing)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-framework-for-factorizing/smac-on-smac-3s5z-vs-4s6z)](https://paperswithcode.com/sota/smac-on-smac-3s5z-vs-4s6z?p=a-unified-framework-for-factorizing)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-framework-for-factorizing/smac-on-smac-mmm2-7m2m1m-vs-8m4m1m)](https://paperswithcode.com/sota/smac-on-smac-mmm2-7m2m1m-vs-8m4m1m?p=a-unified-framework-for-factorizing)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-framework-for-factorizing/smac-on-smac-mmm2-7m2m1m-vs-9m3m1m)](https://paperswithcode.com/sota/smac-on-smac-mmm2-7m2m1m-vs-9m3m1m?p=a-unified-framework-for-factorizing)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-framework-for-factorizing/smac-on-smac-26m-vs-30m)](https://paperswithcode.com/sota/smac-on-smac-26m-vs-30m?p=a-unified-framework-for-factorizing)<br>[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-unified-framework-for-factorizing/smac-on-smac-corridor-2z-vs-24zg)](https://paperswithcode.com/sota/smac-on-smac-corridor-2z-vs-24zg?p=a-unified-framework-for-factorizing) |

This is the official repository that contain the source code for the DFAC-Extended paper:

- [[JMLR 2023] A Unified Framework for Factorizing Distributional Value Functions for Multi-Agent Reinforcement Learning](https://jmlr.org/papers/v24/22-0630.html)

> This paper is an extended version of:
>
> - [[ICML 2021] DFAC Framework: Factorizing the Value Function via Quantile Mixture for Multi-Agent Distributional Q-Learning](https://github.com/j3soon/dfac)

If you have any question regarding the paper or code, ask by [submitting an issue](https://github.com/j3soon/dfac-extended/issues).

## Extensions

The main differences between DFAC-Extended (JMLR version) and the original DFAC (ICML version) are:

1. Extend the state-of-the-art value function factorization method, QPLEX, to its DFAC variant, and demonstrate its ability to tackle non-monotonic tasks that cannot be solved in the previous work.
2. Previously, we only consider a single distributional RL method, IQN, and factorize the joint return distribution with a quantile mixture. We incorporate an additional distributional RL method, C51, into our framework, and show that the joint return distribution can be factorized in a similar way by using convolutions.
3. An additional section is added to analyze the computational complexity for different shape function implementation choices.
4. Run additional experiments on a more difficult matrix game and six additional self-designed StarCraft maps to further validate the benefits of our proposed framework.
5. Highlight the difference between the proposed method and its related works. Furthermore, we discuss new observations and insights of our work, and provide guidance on potential future works.

## Gameplay Video Preview

Learned policy of DDN on Super Hard & Ultra Hard maps:

https://youtu.be/MLdqyyPcv9U

## Installation

Install docker, nvidia-docker, and nvidia-container-runtime. You can refer to [this document](https://j3soon.com/cheatsheets/getting-started-with-python/#docker-containers) for installation instructions.

Execute the following commands in your Linux terminal to build the docker image:

```sh
# Clone the repository
git clone https://github.com/j3soon/dfac-extended.git
cd dfac-extended
# Download StarCraft 2.4.10
wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
# Extract the files to StarCraftII directory
unzip -P iagreetotheeula SC2.4.10.zip
mv SC2.4.10.zip ..
# Build docker image
docker build . --build-arg DOCKER_BASE=nvcr.io/nvidia/tensorflow:19.12-tf1-py3 -t j3soon/dfac-extended:1.0
```

Launch a docker container:

```sh
docker run --gpus all \
    --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
    --rm \
    -it \
    -v "$(pwd)"/pymarl:/root/pymarl \
    -v "$(pwd)"/results:/results \
    -e DISPLAY=$DISPLAY \
    --device /dev/snd \
    j3soon/dfac-extended:1.0 /bin/bash
```

Run the following command in the docker container for quick testing:

```sh
cd /root/pymarl
python3 src/main.py --config=ddn --env-config=sc2 with env_args.map_name=3m t_max=50000
```

After finish training, exit the container by `exit`, the container will be automatically deleted thanks to the `--rm` flag.

The results are stored in `./results`.

> We chose to release the code based on docker for better reproducibility and the ease of use. For installing directly or running the code in virtualenv or conda, you may want to refer to the [Dockerfile](Dockerfile). If you still have trouble setting up the environment, [open an issue](https://github.com/j3soon/dfac-extended/issues) and describe your encountered issue.

## Reproducing

The following is the command used for the experiments in the paper:

```sh
python3 src/main.py --config=$ALGO --env-config=sc2 with env_args.map_name=$MAP_NAME rnn_hidden_dim=$HIDDEN_DIM
```

The arguments are:
- `$ALGO`:
  - Baselines: `{iql, vdn, qmix, qplex}`.
  - DFAC Variants: `{diql, ddn, dmix, dplex}`.
- `$MAP_NAME`:
  - Super Hard Maps: `{3s5z_vs_3s6z, 6h_vs_8z, MMM2, 27m_vs_30m, corridor}`.
  - Ultra Hard Maps: `{26m_vs_30m, 3s5z_vs_4s6z, 6h_vs_9z, MMM2_7m2M1M_vs_8m4M1M, MMM2_7m2M1M_vs_9m3M1M, corridor_2z_vs_24zg}`.
- `$HIDDEN_DIM`:
  - Please refer to Table 6 in the paper for the corresponding hidden dimension setup in each setting.

If you want to modify the algorithm, you can modify the files in `./pymarl` directly, without rebuilding the docker image or restarting the docker container.

## Compare Baseline code with DFAC code

The code of DFAC is organized with minimum changes based on [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl) for readibility. You may want to compare the baselines with their DFAC variants with the following commands:

```sh
# Configs
diff pymarl/src/config/algs/iql.yaml pymarl/src/config/algs/diql.yaml
diff pymarl/src/config/algs/vdn.yaml pymarl/src/config/algs/ddn.yaml
diff pymarl/src/config/algs/qmix.yaml pymarl/src/config/algs/dmix.yaml
diff pymarl/src/config/algs/qplex.yaml pymarl/src/config/algs/dplex.yaml
# Agent
diff pymarl/src/learners/q_learner.py pymarl/src/learners/iqn_learner.py
diff pymarl/src/modules/agents/rnn_agent.py pymarl/src/modules/agents/iqn_rnn_agent.py
# Mixer
diff pymarl/src/modules/mixers/vdn.py pymarl/src/modules/mixers/ddn.py
diff pymarl/src/modules/mixers/qmix.py pymarl/src/modules/mixers/dmix.py
diff pymarl/src/modules/mixers/dmaq_qatten.py pymarl/src/modules/mixers/dplex.py
```

For comparing all modifications based on all used packages, refer to the following comparison links:
- [DFAC modifications](https://github.com/j3soon/dfac/compare/61d2a06..HEAD)
- [DFAC-Extended modifications](https://github.com/j3soon/dfac-extended/compare/5a372eb..HEAD)

## Developing new Algorithms

### Updaing Packages

Since this repository is frozen in old commits for reproducibility, you may want to use the newest packages:

- [oxwhirl/sacred](https://github.com/oxwhirl/sacred)
- [oxwhirl/smac](https://github.com/oxwhirl/smac)
- [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl)

For common baselines, you may want to refer to the following package which collected a bunch of baselines:

- [hijkzzz/pymarl2](https://github.com/hijkzzz/pymarl2)

There are also further improvements in the SMAC benchmark:

- [oxwhirl/smacv2](https://github.com/oxwhirl/smacv2)
- [osilab-kaist/smac_exp](https://github.com/osilab-kaist/smac_exp)

### Inspect the Training Progress

You can inspect the training progress in real-time by the following command:

```sh
tensorboard --logdir=./results
```

## Citing DFAC

If you used the provided code or want to cite our work, please cite the original DFAC paper and the DFAC-Extended paper.

BibTex format:

```
@InProceedings{sun21dfac,
  title = 	 {{DFAC} Framework: Factorizing the Value Function via Quantile Mixture for Multi-Agent Distributional Q-Learning},
  author =       {Sun, Wei-Fang and Lee, Cheng-Kuang and Lee, Chun-Yi},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {9945--9954},
  year = 	 {2021},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/sun21c/sun21c.pdf},
  url = 	 {http://proceedings.mlr.press/v139/sun21c.html},
}
```

```
@article{JMLR:v24:22-0630,
  author  = {Wei-Fang Sun and Cheng-Kuang Lee and Simon See and Chun-Yi Lee},
  title   = {A Unified Framework for Factorizing Distributional Value Functions for Multi-Agent Reinforcement Learning},
  journal = {Journal of Machine Learning Research},
  year    = {2023},
  volume  = {24},
  number  = {220},
  pages   = {1--32},
  url     = {http://jmlr.org/papers/v24/22-0630.html}
}
```

You will also want to [cite the SMAC paper](https://github.com/oxwhirl/smac#citing--smac) for providing the benchmark used in the paper.

## License

To maintain reproducibility, we freezed the following packages with the commit used in the paper. The licenses of these packages are listed below:

- [oxwhirl/sacred](https://github.com/oxwhirl/sacred) (at commit 13f04ad) is released under the [MIT License](https://github.com/oxwhirl/sacred/blob/master/LICENSE.txt)
- [oxwhirl/smac](https://github.com/oxwhirl/smac) (at commit 456d133) is released under the [MIT License](https://github.com/oxwhirl/smac/blob/master/LICENSE)
- [oxwhirl/pymarl](https://github.com/oxwhirl/pymarl) (at commit dd92936) is released under the [Apache-2.0 License](https://github.com/oxwhirl/pymarl/blob/master/LICENSE)
- [wjh720/QPLEX](https://github.com/wjh720/QPLEX) (at commit b672407) is released under the [Apache-2.0 License](https://github.com/wjh720/QPLEX/blob/master/pymarl-master/LICENSE)

Further changes based on the packages above are release under the [Apache-2.0 License](LICENSE).
