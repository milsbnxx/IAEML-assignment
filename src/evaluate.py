"""Paired evaluation of the saved policy and a uniform random baseline."""
import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.ppo import load_checkpoint
from env.robotaxi import RoboTaxi
from env.tasks import MAPS


def episode_keys(seed):
    return jax.random.split(jax.random.PRNGKey(seed))


def evaluate_policy(params, initial, act, seeds):
    def episode(seed):
        reset_key, key = episode_keys(seed)
        observation, state = RoboTaxi.reset(reset_key, params, initial)
        start = (state, observation.as_vector(), key, jnp.float32(0), jnp.bool_(False),
                 jnp.bool_(False), jnp.bool_(False))

        def keep_going(carry):
            return ~carry[4]

        def transition(carry):
            state, obs, key, total, *_ = carry
            key, action_key, step_key = jax.random.split(key, 3)
            new_obs, state, r, done, info = RoboTaxi.step(step_key, state, act(action_key, obs), params)
            return state, new_obs.as_vector(), key, total + r, done, info["success"], info["collision"]

        final, _, _, total, _, success, collision = jax.lax.while_loop(keep_going, transition, start)
        return total, success, collision, final.time, jnp.linalg.norm(final.agent_pos - final.goal_pos)

    return jax.device_get(jax.jit(jax.vmap(episode))(jnp.asarray(seeds)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo.msgpack"))
    parser.add_argument("--route-mode", choices=("validated", "snapshot"), default="validated")
    parser.add_argument("--maps", type=int, nargs="+", choices=range(len(MAPS)), default=list(range(len(MAPS))))
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2000, help="First episode seed (same for both policies)")
    parser.add_argument("--output", type=Path, default=Path("results/evaluation"))
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(args.seed, args.seed + args.episodes)
    records, summaries = [], []
    for map_id in args.maps:
        params, initial = RoboTaxi.init_params(jax.random.PRNGKey(0), map_id=map_id, route_mode=args.route_mode)
        size = RoboTaxi.get_observation(initial, params).as_vector().shape[-1]
        trained, meta = load_checkpoint(args.checkpoint, size)
        policies = {"ppo": lambda key, obs: trained(obs),
                    "random": lambda key, obs: jax.random.uniform(key, (2,), minval=-1., maxval=1.)}
        for name, act in policies.items():
            returns, goals, crashes, lengths, distances = evaluate_policy(params, initial, act, seeds)
            if not np.isfinite(returns).all() or not np.isfinite(distances).all():
                raise FloatingPointError("Non-finite evaluation episode")
            timeouts = ~(goals | crashes)
            for i, seed in enumerate(seeds):
                outcome = "goal" if goals[i] else "collision" if crashes[i] else "timeout"
                records.append(dict(map_id=map_id, policy=name, seed=int(seed), outcome=outcome,
                                    episode_return=float(returns[i]), length=int(lengths[i]),
                                    distance_to_goal_m=float(distances[i])))
            summary = dict(map_id=map_id, policy=name, episodes=len(seeds), goals=int(goals.sum()),
                           collisions=int(crashes.sum()), timeouts=int(timeouts.sum()),
                           mean_episode_return=float(returns.mean()), success_rate=float(goals.mean()),
                           collision_rate=float(crashes.mean()), mean_episode_length=float(lengths.mean()),
                           mean_distance_to_goal_m=float(distances.mean()))
            summaries.append(summary)
            print(f"map {map_id} {name}: {summary['goals']}/{len(seeds)} goals, "
                  f"{summary['collisions']} collisions, {summary['timeouts']} timeouts", flush=True)
    with (args.output / "episodes.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    report = dict(route_mode=args.route_mode, first_seed=int(seeds[0]), last_seed=int(seeds[-1]), checkpoint_update=meta["update"],
                  training_maps=meta["config"]["maps"] if "maps" in meta["config"] else [meta["config"]["map_id"]],
                  summaries=summaries)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
