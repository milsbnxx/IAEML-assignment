import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from env.base import BaseEnv, RADIUS
from env.robotaxi import RoboTaxi, route_at, swept_contact
from utils.autoreset import AutoResetWrapper


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.key = jax.random.PRNGKey(0)
        cls.p, cls.initial = RoboTaxi.init_params(cls.key)

    def road(self):
        """Straight test track away from walls, with one obstacle of each type."""
        start, goal = jnp.array([12., 12.]), jnp.array([38., 12.])
        path = jnp.vstack((start, jnp.tile(goal, (len(self.initial.path_array) - 1, 1))))
        lengths = jnp.concatenate((jnp.zeros(1), jnp.cumsum(jnp.linalg.norm(jnp.diff(path, axis=0), axis=1))))
        progress, cross, waypoint = route_at(start, path, lengths, self.p.lookahead)
        return eqx.tree_at(lambda s: (s.agent_pos, s.goal_pos, s.agent_forward_dir, s.heading,
                                       s.static_obstacles, s.kinematic_obstacles, s.kinematic_obst_velocities,
                                       s.path_array, s.route_lengths, s.progress, s.cross_track, s.waypoint),
                           self.initial, (start, goal, jnp.array([1., 0.]), jnp.float32(0),
                                          jnp.array([[55., 25.]]), jnp.array([[54., 25.]]),
                                          jnp.zeros((1, 2)), path, lengths, progress, cross, waypoint))

    def tick(self, state, action, params=None):
        return RoboTaxi.step(self.key, state, jnp.asarray(action), self.p if params is None else params)

    def test_contract_and_all_course_maps(self):
        self.assertTrue(issubclass(RoboTaxi, BaseEnv))
        for map_id in range(3):
            p, s = RoboTaxi.init_params(self.key, map_id=map_id)
            obs, state = RoboTaxi.reset(self.key, p, s)
            self.assertEqual(state.path_array.shape, p.path_array_shape_dtype[0])
            self.assertEqual(obs.as_vector().shape, (25,))
            self.assertEqual(RoboTaxi.action_size, 2)
            self.assertEqual(obs.collision_rays.shape, (16, 2))
            self.assertTrue(np.all(np.abs(obs.as_vector()) <= 1.00001))
            self.assertEqual(int(state.time), 0)
            self.assertFalse(bool(self.tick(state, [0., 0.], p)[3]))

    def test_seeded_reset(self):
        a = RoboTaxi.reset(self.key, self.p, self.initial)[1]
        b = RoboTaxi.reset(self.key, self.p, self.initial)[1]
        c = RoboTaxi.reset(jax.random.PRNGKey(12), self.p, self.initial)[1]
        np.testing.assert_array_equal(a.kinematic_obstacles, b.kinematic_obstacles)
        self.assertEqual(float(a.heading), float(b.heading))
        self.assertNotEqual(float(a.heading), float(c.heading))
        self.assertFalse(np.array_equal(a.kinematic_obstacles, c.kinematic_obstacles))
        np.testing.assert_array_equal(a.agent_pos, self.initial.agent_pos)

    def test_accelerate_coast_brake_and_limits(self):
        road = self.road()
        _, fast, *_ = self.tick(road, [4., 0.])
        self.assertAlmostEqual(float(fast.acceleration), self.p.max_acceleration, places=5)
        self.assertGreater(float(fast.agent_pos[0]), float(road.agent_pos[0]))
        self.assertEqual(float(fast.agent_pos[1]), 12.)
        coast = self.tick(fast, [0., 0.])[1]
        brake = self.tick(fast, [-1., 0.])[1]
        self.assertLess(float(brake.speed), float(coast.speed))
        self.assertLess(float(coast.speed), float(fast.speed))
        self.assertEqual(float(self.tick(road, [-1., 0.])[1].speed), 0.)
        at_cap = eqx.tree_at(lambda s: s.speed, road, jnp.float32(self.p.max_speed))
        self.assertEqual(float(self.tick(at_cap, [1., 0.])[1].speed), self.p.max_speed)
        self.assertGreaterEqual(float(self.tick(at_cap, [-1., 0.])[1].acceleration), -self.p.max_braking - 1e-5)

    def test_steering_and_stationary_heading(self):
        road = self.road()
        self.assertEqual(float(self.tick(road, [0., 1.])[1].heading), 0.)
        rolling = eqx.tree_at(lambda s: s.speed, road, jnp.float32(4))
        left = self.tick(rolling, [0., 2.])[1]
        right = self.tick(rolling, [0., -1.])[1]
        self.assertAlmostEqual(float(left.steering), self.p.max_steering, places=6)
        self.assertGreater(float(left.heading), 0.)
        self.assertAlmostEqual(float(left.heading), -float(right.heading), places=6)
        self.assertGreater(float(left.agent_pos[1]), float(right.agent_pos[1]))

    def test_ray_range_from_car_centre(self):
        s = eqx.tree_at(lambda x: x.static_obstacles, self.road(), jnp.array([[16., 12.]]))
        obs = RoboTaxi.get_observation(s, self.p)
        forward_ray = self.p.num_ray_sensors // 2
        np.testing.assert_allclose(obs.collision_rays[forward_ray], [3., 0.], atol=1e-5)
        self.assertAlmostEqual(float(obs.as_vector()[9 + forward_ray]), 3 / self.p.perception_radius, places=6)

    def test_collision_on_step_and_collision_priority(self):
        road = self.road()
        s = eqx.tree_at(lambda x: (x.speed, x.static_obstacles, x.goal_pos), road,
                        (jnp.float32(5), jnp.array([[14.3, 12.]]), jnp.array([13.2, 12.])))
        _, _, reward, done, info = self.tick(s, [0., 0.])
        self.assertTrue(bool(done & info["collision"] & info["terminated"]))
        self.assertFalse(bool(info["success"] | info["truncated"]))
        self.assertLess(float(reward), -20.)
        self.assertTrue(bool(swept_contact(jnp.array([-3., 0.]), jnp.array([3., 0.]), 2.)))

    def test_goal_timeout_and_world_boundary(self):
        road = self.road()
        s = eqx.tree_at(lambda x: (x.speed, x.goal_pos), road, (jnp.float32(6), jnp.array([13.4, 12.])))
        _, _, reward, done, info = self.tick(s, [0., 0.])
        self.assertTrue(bool(done & info["success"]))
        self.assertGreater(float(reward), 29.)
        last = eqx.tree_at(lambda s: s.time, road, jnp.int32(self.p.max_steps_in_episode - 1))
        _, _, _, done, info = self.tick(last, [0., 0.])
        self.assertTrue(bool(done & info["truncated"]))
        self.assertFalse(bool(info["terminated"]))
        edge = eqx.tree_at(lambda s: s.agent_pos, road, jnp.array([0.75, 12.]))
        self.assertTrue(bool(self.tick(edge, [0., 0.])[4]["collision"]))

    def test_moving_obstacle_and_wrap(self):
        road = self.road()
        approaching = eqx.tree_at(lambda s: (s.kinematic_obstacles, s.kinematic_obst_velocities), road,
                                  (jnp.array([[14.05, 12.]]), jnp.array([[-1., 0.]])))
        self.assertTrue(bool(self.tick(approaching, [0., 0.])[4]["collision"]))
        wrapping = eqx.tree_at(lambda s: (s.kinematic_obstacles, s.kinematic_obst_velocities), road,
                               (jnp.array([[59.97, 12.]]), jnp.array([[1., 0.]])))
        _, new, _, done, _ = self.tick(wrapping, [0., 0.])
        self.assertLess(float(new.kinematic_obstacles[0, 0]), 1.)
        self.assertFalse(bool(done))

    def test_progress_beats_waiting_and_round_trip_costs_time(self):
        s = self.road()
        waiting = self.tick(s, [0., 0.])
        moving = self.tick(eqx.tree_at(lambda s: s.speed, s, jnp.float32(3)), [0., 0.])
        self.assertLess(float(waiting[2]), 0.)
        self.assertGreater(float(moving[2]), 0.)
        self.assertGreater(float(moving[1].progress), float(s.progress))
        # Reverse direction on the fixture to visit exactly the same two positions twice.
        rolling = eqx.tree_at(lambda s: s.speed, s, jnp.float32(3))
        hold_speed = (self.p.rolling_resistance + self.p.drag * 9) / self.p.max_acceleration
        _, forward, first_reward, *_ = self.tick(rolling, [hold_speed, 0.])
        returning = eqx.tree_at(lambda s: (s.heading, s.agent_forward_dir), forward,
                                (jnp.float32(jnp.pi), jnp.array([-1., 0.])))
        _, back, second_reward, *_ = self.tick(returning, [hold_speed, 0.])
        np.testing.assert_allclose(back.agent_pos, s.agent_pos, atol=1e-5)
        self.assertAlmostEqual(float(first_reward + second_reward), -0.04, places=5)

    def test_validated_planner_avoids_static_discs(self):
        for m in range(3):
            _, s = RoboTaxi.init_params(self.key, map_id=m)
            delta = s.path_array[:, None, :] - s.static_obstacles
            self.assertFalse(bool(jnp.any(swept_contact(delta[:-1], delta[1:], 2 * RADIUS))))
        _, base_route = RoboTaxi.init_params(self.key, map_id=2, route_mode="snapshot")
        delta = base_route.path_array[:, None, :] - base_route.static_obstacles
        self.assertTrue(bool(jnp.any(swept_contact(delta[:-1], delta[1:], 2 * RADIUS))))

    def test_jit_vmap_finite_and_scalar_agreement(self):
        keys = jax.random.split(self.key, 4)
        states = jax.vmap(lambda k: RoboTaxi.reset(k, self.p, self.initial)[1])(keys)
        actions = jax.random.uniform(self.key, (4, 2), minval=-1, maxval=1)
        batch_step = jax.jit(jax.vmap(RoboTaxi.step, in_axes=(0, 0, 0, None)), static_argnums=(3,))
        batched = batch_step(keys, states, actions, self.p)
        for leaf in jax.tree.leaves(batched):
            self.assertTrue(np.isfinite(leaf).all())
        for i in range(4):
            scalar = self.tick(jax.tree.map(lambda x: x[i], states), actions[i])
            np.testing.assert_allclose(batched[0].as_vector()[i], scalar[0].as_vector(), atol=1e-5)

    def test_course_autoreset_wrapper(self):
        p, initial = RoboTaxi.init_params(self.key, max_steps=1)
        wrapped = AutoResetWrapper(RoboTaxi, p, initial)
        _, state = wrapped.reset(self.key, p, initial)
        obs, state, _, done, info = wrapped.step(self.key, state, jnp.zeros(2), p)
        self.assertTrue(bool(done & info["truncated"]))
        self.assertEqual(int(state.time), 0)
        self.assertEqual(obs.as_vector().shape, (25,))
