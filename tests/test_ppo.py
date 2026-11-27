from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.ppo import (ActorCritic, Config, generalized_advantages, latent_log_prob,
                            load_checkpoint, log_tanh_jacobian, save_checkpoint)


class PPOTests(unittest.TestCase):
    def test_gae_masks_and_bootstrap(self):
        rewards = jnp.array([[1.], [2.], [3.]])
        zeros = jnp.zeros_like(rewards)
        # Transitions 1 and 2 are in different episodes. Only the timeout bootstraps.
        terminated = jnp.array([[False], [False], [True]])
        done = jnp.array([[False], [True], [True]])
        values = jnp.array([[0.], [4.], [100.]])
        adv, target = generalized_advantages(rewards, zeros, values, terminated, done, 0.5, 1.)
        np.testing.assert_allclose(adv[:, 0], [3., 4., 3.])
        np.testing.assert_array_equal(target, adv)
        # An unfinished rollout also bootstraps the following observation.
        adv, _ = generalized_advantages(rewards[:1], zeros[:1], jnp.array([[8.]]), terminated[:1], done[:1], .5, .95)
        self.assertEqual(float(adv[0, 0]), 5.)

    def test_tanh_ratio_and_numerical_stability(self):
        raw = jnp.array([[0.2, -1.], [30., -40.]])
        old = latent_log_prob(raw, jnp.zeros(2), jnp.zeros(2))
        new = latent_log_prob(raw, jnp.array([0.01, -0.01]), jnp.zeros(2))
        correction = log_tanh_jacobian(raw)
        self.assertTrue(np.isfinite(correction).all())
        np.testing.assert_allclose(new - old, (new - correction) - (old - correction), atol=2e-5)

    def test_checkpoint_reproduces_bounded_inference(self):
        config = Config()
        net = ActorCritic(config.hidden)
        obs = jnp.linspace(-1., 1., 25)
        weights = net.init(jax.random.PRNGKey(11), obs)["params"]
        mean, _, _ = net.apply({"params": weights}, obs)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.msgpack"
            save_checkpoint(path, weights, config, 25, 1)
            act, meta = load_checkpoint(path, 25)
            np.testing.assert_allclose(act(obs), jnp.tanh(mean), atol=1e-6)
            self.assertTrue(np.all(np.abs(act(obs)) <= 1))
            self.assertEqual(meta["update"], 1)
            with self.assertRaises(ValueError):
                load_checkpoint(path, 17)
