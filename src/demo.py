"""Record one complete seeded episode as a silent H.264 movie."""
import argparse
from functools import partial
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import jax
import jax.numpy as jnp

from algorithms.ppo import load_checkpoint
from env.robotaxi import RoboTaxi
from env.tasks import MAPS
from evaluate import episode_keys
from utils.taxi_renderer import TaxiRenderer


@partial(jax.jit, static_argnames=("params",))
def controller(state, params):
    """Pure pursuit plus a speed controller. This policy has no learned weights."""
    offset = state.waypoint - state.agent_pos
    forward = state.agent_forward_dir
    left = jnp.array([-forward[1], forward[0]])
    curvature = 2 * jnp.dot(left, offset) / jnp.maximum(jnp.dot(offset, offset), 0.1)
    turn = jnp.arctan(params.wheelbase * curvature) / params.max_steering
    throttle = (4.0 - state.speed) / params.max_acceleration
    return jnp.clip(jnp.stack((throttle, turn)), -1., 1.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("controller", "ppo"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/ppo.msgpack"))
    parser.add_argument("--map", type=int, choices=range(len(MAPS)), default=0)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--output", type=Path, help="Defaults to the assignment video path")
    args = parser.parse_args()
    if not shutil.which("ffmpeg"):
        parser.error("Install FFmpeg with the libx264 encoder and put ffmpeg on PATH.")
    name = "environment_demo" if args.policy == "controller" else "trained_agent_demo"
    path = args.output or Path("videos") / f"{name}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    reset_key, key = episode_keys(args.seed)
    p, initial = RoboTaxi.init_params(reset_key, map_id=args.map)
    obs, s = RoboTaxi.reset(reset_key, p, initial)
    if args.policy == "ppo":
        act, meta = load_checkpoint(args.checkpoint, obs.as_vector().shape[-1])
        title = f"Trained PPO agent  /  checkpoint {meta['update']}"
    else:
        title = "Environment demo  /  scripted controller"
    renderer = TaxiRenderer(p, title, args.map, args.seed)
    width, height = renderer.screen.get_size()
    with NamedTemporaryFile(dir=path.parent, suffix=".partial.mp4", delete=False) as temporary:
        tmp = Path(temporary.name)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(p.fps),
               "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
    total = 0.
    status = "Driving"
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            encoder.stdin.write(renderer.frame(s, obs, p, status))
            for _ in range(p.max_steps_in_episode):
                key, _, step_key = jax.random.split(key, 3)
                action = act(obs.as_vector()) if args.policy == "ppo" else controller(s, p)
                obs, s, reward, done, info = RoboTaxi.step(step_key, s, action, p)
                total += float(reward)
                if bool(done):
                    status = "Goal reached" if bool(info["success"]) else "Collision" if bool(info["collision"]) else "Time limit"
                frame = renderer.frame(s, obs, p, status)
                encoder.stdin.write(frame)
                if bool(done):
                    for _ in range(p.fps):
                        encoder.stdin.write(frame)
                    break
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("FFmpeg failed to encode the simulation.")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{path}: {status}, {int(s.time)} steps, return {total:.3f}, distance {float(info['distance_to_goal']):.3f} m")


if __name__ == "__main__":
    main()
