# IAEML-assignment

A JAX robo-taxi learns to steer and control its speed on the three Innopolis course maps. The final PPO policy reached the destination in **192/192 held-out episodes**. A uniform random driver reached it in **0/192** on the same starts. The policy trained on maps 0 and 2. Map 1 was used for evaluation only. [Held-out results](results/holdout/summary.json)

| Assignment | Report | Demonstration | Model |
|---|---|---|---|
| 1 · Environment | [Dynamics, observations and reward](reports/assignment1.md) | [Scripted controller · 18 s](videos/environment_demo.mp4) | Pure pursuit |
| 2 · Machine learning | [PPO experiments and results](reports/assignment2.md) | [Trained agent · 10 s](videos/trained_agent_demo.mp4) | [Final PPO checkpoint](checkpoints/ppo.msgpack) |

## Installation

Use Python 3.12 and FFmpeg with the libx264 encoder. The pinned CPU dependencies were tested on Linux x86-64. On macOS with Homebrew, FFmpeg is available through `brew install ffmpeg`. On Debian or Ubuntu the package is `ffmpeg`.

```bash
git clone -b feature/robotaxi-ppo https://github.com/milsbnxx/IAEML-assignment.git
cd IAEML-assignment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
ffmpeg -version
```

## Run the project

Run commands from the repository root. The two video commands use map 0 and seed 2000 by default. The PPO video command reloads the included weights and performs inference without training. Files under `runs/` are scratch outputs for local experiments.

| Task | Command |
|---|---|
| Check the environment and learning code | `PYTHONPATH=src python -m unittest discover -s tests -v` |
| Generate the controller video | `python src/demo.py --policy controller` |
| Load the final model and generate its video | `python src/demo.py --policy ppo --checkpoint checkpoints/ppo.msgpack` |
| Smoke-test learning and checkpoint loading | `python src/train.py --updates 2 --envs 4 --steps 16 --epochs 2 --minibatches 2 --run-dir runs/smoke --checkpoint runs/smoke/ppo.msgpack` |
| Reproduce the full PPO run | `python src/train.py` |
| Evaluate PPO and random actions on paired starts | `python src/evaluate.py` |
| Evaluate the fresh set of held-out seeds | `python src/evaluate.py --seed 3000 --output results/holdout` |
| Regenerate the four learning curves | `python src/plot_training.py` |

The full training command writes `results/training/` and `checkpoints/ppo.msgpack`. To keep an additional run, choose a different `--run-dir` and `--checkpoint`. Evaluation runs 64 episodes per map per policy. Every row in `episodes.csv` includes the map, seed, outcome, return, episode length and final distance. The first evaluation uses seeds 2000–2063 and the held-out evaluation uses 3000–3063. [Episode records](results/holdout/episodes.csv) · [Learning curves](results/training/training_curves.png)

## Project structure

| Path | Contents |
|---|---|
| [src/env](src/env) | Course maps and base environment, plus bicycle dynamics, normalized observations and rewards in `robotaxi.py` |
| [src/algorithms/ppo.py](src/algorithms/ppo.py) | Actor, critic, distribution, GAE, PPO training update and checkpoint I/O |
| [src/train.py](src/train.py) · [src/evaluate.py](src/evaluate.py) | Training loop, measured logs and paired evaluation |
| [src/demo.py](src/demo.py) · [src/utils/taxi_renderer.py](src/utils/taxi_renderer.py) | Scripted or trained driving, course ray overlay and off-screen MP4 recording |
| [tests](tests) | Fifteen automated environment and PPO tests |
| [results](results) · [checkpoints](checkpoints) · [videos](videos) | Training curves, per-episode measurements, three model snapshots and both demos |
| [reports](reports) | Two signed technical reports with space for the author's personal reflections, plus factual troubleshooting notes |

## Reproduce the earlier experiments

The first run used map 0 only. The second used the two training maps with the original snapshot-based route. These commands keep their logs and checkpoints in separate directories.

```bash
python src/train.py --maps 0 --updates 160 --route-mode snapshot --run-dir results/single_map/training --checkpoint checkpoints/ppo_map0.msgpack
python src/evaluate.py --seed 1000 --route-mode snapshot --checkpoint checkpoints/ppo_map0.msgpack --output results/single_map/evaluation
python src/train.py --route-mode snapshot --run-dir results/snapshot/training --checkpoint checkpoints/ppo_snapshot.msgpack
python src/evaluate.py --route-mode snapshot --checkpoint checkpoints/ppo_snapshot.msgpack --output results/snapshot/evaluation
```

## Scope and limitations

The 2D simulator uses forward-only low-speed motion, a circular vehicle collider, instantaneous steering and a cached route. The policy receives waypoint guidance and clearance rays. The experiment uses one training seed and three small layouts. One held-out layout shares much of its geometry with a training map. The reported percentages apply to the recorded episode sets.

The starting point was InnoRL's course repository at commit `1056d71`. The original course code, instructions and keyboard demo are retained. The technical reports cite the bicycle model, PPO and GAE references. [Course starter](https://github.com/InnoRL/IAEML-assignment/tree/1056d711e166747cc485725e31f2535967b030dd) · [Course README](instructions/course-readme.md)
