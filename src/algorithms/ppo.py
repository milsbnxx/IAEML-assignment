"""Continuous PPO-Clip with a tanh-Gaussian policy and GAE."""
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import NamedTuple

from flax import linen as nn, serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from env.robotaxi import RoboTaxi


@dataclass(frozen=True)
class Config:
    seed: int = 7
    maps: tuple[int, ...] = (0, 2)
    route_mode: str = "validated"
    updates: int = 320
    envs: int = 32
    steps: int = 64
    epochs: int = 4
    minibatches: int = 4
    hidden: int = 64
    learning_rate: float = 0.00035
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.003
    grad_clip: float = 0.5
    initial_short_probability: float = 0.65

    def __post_init__(self):
        if min(self.envs, self.steps, self.updates, self.epochs, self.minibatches, self.hidden) < 1:
            raise ValueError("Training sizes must be positive.")
        if self.envs * self.steps % self.minibatches:
            raise ValueError("The rollout size must be divisible by the number of minibatches.")


class ActorCritic(nn.Module):
    hidden: int = 64

    @nn.compact
    def __call__(self, obs):
        def tower(x, name, outputs, output_gain):
            for i in range(2):
                x = nn.tanh(nn.Dense(self.hidden, name=f"{name}_{i}",
                                     kernel_init=nn.initializers.orthogonal(np.sqrt(2.)))(x))
            return nn.Dense(outputs, name=f"{name}_out",
                            kernel_init=nn.initializers.orthogonal(output_gain))(x)

        mean = tower(obs, "actor", RoboTaxi.action_size, 0.01)
        value = tower(obs, "critic", 1, 1.0)[..., 0]
        log_std = self.param("log_std", nn.initializers.constant(-0.45), (RoboTaxi.action_size,))
        return mean, jnp.clip(log_std, -4., 1.), value


def latent_log_prob(raw, mean, log_std):
    z = (raw - mean) * jnp.exp(-log_std)
    return -0.5 * jnp.sum(z**2 + 2 * log_std + np.log(2 * np.pi), axis=-1)


def log_tanh_jacobian(raw):
    # log(1 - tanh(x)^2) in a form that stays finite for large |x|.
    return jnp.sum(2 * (np.log(2.) - raw - jax.nn.softplus(-2 * raw)), axis=-1)


def generalized_advantages(rewards, values, next_values, terminated, done, gamma, gae_lambda):
    # A timeout bootstraps the final observation. Both kinds of ending stop the trace.
    deltas = rewards + gamma * (~terminated) * next_values - values

    def backward(following, data):
        delta, boundary = data
        advantage = delta + gamma * gae_lambda * (~boundary) * following
        return advantage, advantage

    _, advantages = jax.lax.scan(backward, jnp.zeros_like(values[0]), (deltas, done), reverse=True)
    return advantages, advantages + values


class Rollout(NamedTuple):
    observation: jax.Array
    raw_action: jax.Array
    old_log_prob: jax.Array
    value: jax.Array
    next_value: jax.Array
    reward: jax.Array
    terminated: jax.Array
    done: jax.Array
    success: jax.Array
    collision: jax.Array
    episode_return: jax.Array
    length: jax.Array


def initialize_learner(config, observation_size):
    network = ActorCritic(config.hidden)
    optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                            optax.adam(config.learning_rate, eps=1e-5))
    model_key, reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(config.seed), 3)
    weights = network.init(model_key, jnp.zeros(observation_size))["params"]
    return network, optimizer, weights, optimizer.init(weights), reset_key, rollout_key


def make_trainer(config, params, initial, network, optimizer):
    def sample_batch(key, short_probability):
        keys = jax.random.split(key, config.envs)
        states = jax.vmap(RoboTaxi.sample_start, in_axes=(0, None, None, None))(
            keys, params, initial, short_probability)
        observations = jax.vmap(lambda s: RoboTaxi.get_observation(s, params).as_vector())(states)
        return states, observations

    def initialize(reset_key, rollout_key):
        states, observations = jax.jit(sample_batch)(reset_key, config.initial_short_probability)
        return states, observations, jnp.zeros(config.envs), rollout_key

    def loss_fn(weights, data, entropy_key):
        obs, raw, old_log_prob, advantage, target = data
        mean, log_std, value = network.apply({"params": weights}, obs)
        # The tanh Jacobian cancels in the ratio for the same stored action.
        ratio = jnp.exp(latent_log_prob(raw, mean, log_std) - old_log_prob)
        clipped = jnp.clip(ratio, 1 - config.clip, 1 + config.clip)
        policy_loss = -jnp.mean(jnp.minimum(ratio * advantage, clipped * advantage))
        value_loss = 0.5 * jnp.mean((value - target)**2)
        fresh = mean + jnp.exp(log_std) * jax.random.normal(entropy_key, mean.shape)
        entropy = jnp.mean(-latent_log_prob(fresh, mean, log_std) + log_tanh_jacobian(fresh))
        total = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        return total, jnp.stack((policy_loss, value_loss, entropy))

    @jax.jit
    def update(weights, opt_state, carry, short_probability):
        def one_step(carry, _):
            states, obs, episode_return, key = carry
            key, action_key, step_key, reset_key = jax.random.split(key, 4)
            mean, log_std, value = network.apply({"params": weights}, obs)
            raw = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
            step_keys = jax.random.split(step_key, config.envs)
            following, new_states, reward, done, info = jax.vmap(
                RoboTaxi.step, in_axes=(0, 0, 0, None))(step_keys, states, jnp.tanh(raw), params)
            new_obs = following.as_vector()
            _, _, next_value = network.apply({"params": weights}, new_obs)
            episode_return = episode_return + reward
            transition = Rollout(obs, raw, latent_log_prob(raw, mean, log_std), value, next_value,
                                 reward, info["terminated"], done, info["success"], info["collision"],
                                 episode_return, info["time"])

            def reset_finished(_):
                reset_states, reset_obs = sample_batch(reset_key, short_probability)

                def select(reset_value, continuing):
                    mask = done.reshape((config.envs,) + (1,) * (continuing.ndim - 1))
                    return jnp.where(mask, reset_value, continuing)

                return jax.tree.map(select, reset_states, new_states), select(reset_obs, new_obs)

            states, obs = jax.lax.cond(jnp.any(done), reset_finished, lambda _: (new_states, new_obs), None)
            return (states, obs, jnp.where(done, 0., episode_return), key), transition

        carry, rollout = jax.lax.scan(one_step, carry, None, length=config.steps)
        advantages, targets = generalized_advantages(rollout.reward, rollout.value, rollout.next_value,
                                                      rollout.terminated, rollout.done,
                                                      config.gamma, config.gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        size = config.envs * config.steps
        data = (rollout.observation, rollout.raw_action, rollout.old_log_prob, advantages, targets)
        flat = jax.tree.map(lambda x: x.reshape((size,) + x.shape[2:]), data)
        states, obs, returns, key = carry

        def epoch(train_state, _):
            weights, opt_state, key = train_state
            key, permutation_key, entropy_key = jax.random.split(key, 3)
            batches = jax.random.permutation(permutation_key, size).reshape(config.minibatches, -1)
            entropy_keys = jax.random.split(entropy_key, config.minibatches)

            def minibatch(state, inputs):
                weights, opt_state = state
                ids, key = inputs
                batch = jax.tree.map(lambda x: x[ids], flat)
                (_, losses), grads = jax.value_and_grad(loss_fn, has_aux=True)(weights, batch, key)
                updates, opt_state = optimizer.update(grads, opt_state, weights)
                return (optax.apply_updates(weights, updates), opt_state), losses

            (weights, opt_state), losses = jax.lax.scan(minibatch, (weights, opt_state), (batches, entropy_keys))
            return (weights, opt_state, key), losses

        (weights, opt_state, key), losses = jax.lax.scan(epoch, (weights, opt_state, key), None,
                                                          length=config.epochs)
        outcomes = dict(episodes=rollout.done.sum(), successes=rollout.success.sum(),
                        collisions=rollout.collision.sum(), return_sum=(rollout.episode_return * rollout.done).sum(),
                        length_sum=(rollout.length * rollout.done).sum())
        return weights, opt_state, (states, obs, returns, key), outcomes, losses.mean(axis=(0, 1))

    return initialize, update


def save_checkpoint(path, weights, config, obs_size, update):
    metadata = {**asdict(config), "maps": list(config.maps)}
    payload = dict(version=1, config=metadata, obs_size=obs_size, update=update,
                   weights=jax.device_get(weights))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(serialization.msgpack_serialize(payload))
            output.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path, observation_size):
    payload = serialization.msgpack_restore(Path(path).read_bytes())
    if payload["version"] != 1 or payload["obs_size"] != observation_size:
        raise ValueError("Checkpoint format or observation size does not match this environment.")
    model = ActorCritic(hidden=payload["config"]["hidden"])
    weights = jax.tree.map(jnp.asarray, payload["weights"])

    @jax.jit
    def act(obs):
        mean, _, _ = model.apply({"params": weights}, obs)
        return jnp.tanh(mean)

    return act, {k: v for k, v in payload.items() if k != "weights"}
