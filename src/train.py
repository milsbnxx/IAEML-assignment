"""Train PPO, store per-update measurements and write an inference checkpoint."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time

import jax
import numpy as np

from algorithms.ppo import Config, initialize_learner, load_checkpoint, make_trainer, save_checkpoint
from env.robotaxi import RoboTaxi
from env.tasks import MAPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("seed", "updates", "envs", "steps", "epochs", "minibatches"):
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(Config, name))
    parser.add_argument("--route-mode", choices=("validated", "snapshot"), default=Config.route_mode)
    parser.add_argument("--maps", type=int, nargs="+", choices=range(len(MAPS)), default=Config.maps)
    parser.add_argument("--run-dir", type=Path, default=Path("results/training"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo.msgpack"))
    args = parser.parse_args()
    config = Config(maps=tuple(args.maps), **{k: v for k, v in vars(args).items()
                    if k not in ("run_dir", "checkpoint", "maps")})
    if len(set(config.maps)) != len(config.maps):
        parser.error("List each training map once.")
    key = jax.random.PRNGKey(config.seed)
    worlds = [RoboTaxi.init_params(key, map_id=m, route_mode=config.route_mode) for m in config.maps]
    sample = RoboTaxi.reset(key, *worlds[0])[0].as_vector()
    obs_size = sample.shape[-1]
    network, optimizer, weights, opt_state, reset_key, rollout_key = initialize_learner(config, obs_size)
    updates, carries = [], []
    for i, (params, initial) in enumerate(worlds):
        reset, update = make_trainer(config, params, initial, network, optimizer)
        # Independent streams per map. Stream zero also reproduces the single-map experiment.
        rk = jax.random.fold_in(reset_key, i) if i else reset_key
        tk = jax.random.fold_in(rollout_key, i) if i else rollout_key
        carries.append(reset(rk, tk))
        updates.append(update)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    fields = ("update", "map_id", "transitions", "short_probability", "episodes", "successes", "collisions",
              "mean_episode_return", "success_rate", "collision_rate", "mean_episode_length",
              "policy_loss", "value_loss", "entropy")
    with (args.run_dir / "updates.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i in range(1, config.updates + 1):
            short = config.initial_short_probability * max(0., 1. - (i - 1) / (0.6 * config.updates))
            world = (i - 1) % len(worlds)
            weights, opt_state, carries[world], outcomes, losses = updates[world](
                weights, opt_state, carries[world], np.float32(short))
            counts, losses = jax.device_get((outcomes, losses))
            if not np.isfinite(losses).all() or not np.isfinite(counts["return_sum"]):
                raise FloatingPointError(f"Non-finite training statistics at update {i}")
            n = int(counts["episodes"])
            row = dict(update=i, map_id=config.maps[world], transitions=i * config.envs * config.steps, short_probability=short,
                       episodes=n, successes=int(counts["successes"]), collisions=int(counts["collisions"]),
                       mean_episode_return=float(counts["return_sum"]) / n if n else "",
                       success_rate=int(counts["successes"]) / n if n else "",
                       collision_rate=int(counts["collisions"]) / n if n else "",
                       mean_episode_length=int(counts["length_sum"]) / n if n else "",
                       policy_loss=float(losses[0]), value_loss=float(losses[1]), entropy=float(losses[2]))
            writer.writerow(row)
            out.flush()
            if i == 1 or i % 10 == 0 or i == config.updates:
                print(f"update {i}/{config.updates}, steps {row['transitions']}, finished {n}, "
                      f"goals {row['successes']}, collisions {row['collisions']}", flush=True)
    elapsed = time.perf_counter() - start
    save_checkpoint(args.checkpoint, weights, config, obs_size, config.updates)
    act, _ = load_checkpoint(args.checkpoint, obs_size)
    action = np.asarray(act(sample))
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Invalid action after loading the checkpoint")
    run = dict(config=asdict(config), transitions=config.updates * config.envs * config.steps,
               seconds_including_update_compilation=elapsed, devices=[str(d) for d in jax.devices()],
               python=platform.python_version(), jax=jax.__version__, observation_size=obs_size,
               reloaded_action=action.tolist())
    (args.run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Saved {args.checkpoint}, reload action {action}, measured update loop {elapsed:.1f} s", flush=True)


if __name__ == "__main__":
    main()
