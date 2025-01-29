# Full Network Capacity (FNC) Training Framework

This repository implements the Full Network Capacity (FNC) training framework in JAX, building
on Dopamine. SPR [(Schwarzer et al, 2021)](spr), SR-SPR [(D'Oro et al, 2023)](sr-spr) and BBF[(Schwarzer et al, 2023)](bbf) may also be run as hyperparameter configurations.

## Setup
To install the repository, simply run `pip install -r requirements.txt`.
Note that depending on your operating system and cuda version extra steps may be necessary to
successfully install JAX: please see [the JAX install instructions](https://pypi.org/project/jax/) for guidance.
For DMControl,
intstall the osmesa with `apt install libosmesa6`.


## Training
To run a FNC agent locally for a game in the Atari 100K benchmark, run

```
python -m bbf.train \
    --agent=FNC \
    --gin_files=bbf/configs/FNC.gin \
    --base_dir=exp/fnc/Pong/seed \
    --gin_bindings="DataEfficientAtariRunner.game_name = 'Pong'"

```

To run a FNC agent locally for a game in the DMControl 100K benchmark, run

```
export MUJOCO_GL=osmesa
python -m continuous_control.train \
        --save_dir=exp_con/train/cheetah-run/seed \
        --env_name cheetah-run \
        --max_steps 100000 \
        --resets \
        --reset_interval 20000 \
        --fnc \
        --threshold 0.0
```

## References
* [Max Schwarzer, Ankesh Anand, Rishab Goel, Devon Hjelm, Aaron Courville and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. In The Ninth International Conference on Learning Representations, 2021.][spr]

* [Pierluca D'Oro, Max Schwarzer, Evgenii Nikishin, Pierre-Luc Bacon, Marc Bellemare, Aaron Courville.  Sample-efficient reinforcement learning by breaking the replay ratio barrier. In The Eleventh International Conference on Learning Representations, 2023][sr-spr]

* [Max Schwarzer and Johan Samir Obando-Ceron and Aaron C. Courville and Marc G. Bellemare and Rishabh Agarwal and Pablo Samuel Castro.  Bigger, Better, Faster: Human-level Atari with human-level efficiency. In the International Conference on Machine Learning, 2023][bbf]

[spr]: https://openreview.net/forum?id=uCQfPZwRaUu
[sr-spr]: https://openreview.net/forum?id=OpC-9aBBVJe
[bbf]: https://openreview.net/forum?id=OpC-9aBBVJe