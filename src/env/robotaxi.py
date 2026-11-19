"""A low-speed bicycle model on the IAEML course maps. World units are metres."""
from dataclasses import fields
from functools import partial
import math

import equinox as eqx
import jax
import jax.numpy as jnp

from env.base import BaseEnv, BaseEnvObservation, BaseEnvParams, BaseEnvState, FOV, RADIUS


class TaxiParams(BaseEnvParams):
    wheelbase: float = 1.6
    max_speed: float = 7.0
    max_acceleration: float = 3.5
    max_braking: float = 5.0
    max_steering: float = 0.60
    rolling_resistance: float = 0.15
    drag: float = 0.018
    lookahead: float = 3.8


class TaxiState(BaseEnvState):
    heading: jax.Array
    speed: jax.Array
    acceleration: jax.Array
    steering: jax.Array
    route_lengths: jax.Array  # cumulative distance at each waypoint
    progress: jax.Array       # distance along the route at the nearest projection
    cross_track: jax.Array
    waypoint: jax.Array       # lookahead point on the route


class TaxiObservation(BaseEnvObservation):
    features: jax.Array

    def as_vector(self):
        return self.features


def direction(angle):
    return jnp.stack((jnp.cos(angle), jnp.sin(angle)))


def relative_direction(vector, forward):
    unit = vector / jnp.maximum(jnp.linalg.norm(vector), 1e-6)
    left = jnp.array([-forward[1], forward[0]])
    return jnp.stack((jnp.dot(forward, unit), jnp.dot(left, unit)))


def route_at(position, path, lengths, lookahead):
    """Project onto the closest non-empty segment, then walk ahead in metres."""
    segments = path[1:] - path[:-1]
    squared = jnp.sum(jnp.square(segments), axis=1)
    fraction = jnp.clip(jnp.sum((position - path[:-1]) * segments, axis=1)
                        / jnp.maximum(squared, 1e-8), 0., 1.)
    projections = path[:-1] + fraction[:, None] * segments
    distances = jnp.sum(jnp.square(position - projections), axis=1)
    index = jnp.argmin(jnp.where(squared > 0, distances, jnp.inf))
    progress = lengths[index] + fraction[index] * (lengths[index + 1] - lengths[index])
    ahead = jnp.minimum(progress + lookahead, lengths[-1])
    # At the goal the planner pads the route with repeated waypoints.
    end = jnp.clip(jnp.sum(lengths < ahead), 1, len(path) - 1)
    weight = (ahead - lengths[end - 1]) / jnp.maximum(lengths[end] - lengths[end - 1], 1e-6)
    target = path[end - 1] + weight * segments[end - 1]
    return progress, jnp.sqrt(distances[index]), target


def swept_contact(relative_start, relative_end, radius):
    """Closest approach between two centres along their relative step segment."""
    delta = relative_end - relative_start
    fraction = jnp.clip(-jnp.sum(relative_start * delta, axis=-1)
                        / jnp.maximum(jnp.sum(delta * delta, axis=-1), 1e-9), 0., 1.)
    closest = relative_start + fraction[..., None] * delta
    return jnp.sum(closest * closest, axis=-1) <= radius**2


class RoboTaxi(BaseEnv):
    action_size = 2
    metres_per_cell = 3

    @staticmethod
    def init_params(key, map_id=0, max_steps=192, path_length=40, discretization_scale=1,
                    perception_radius=9.0, num_ray_sensors=16, fov=FOV, fps=8, route_mode="validated"):
        if path_length < 2 or max_steps < 1 or fps <= 0 or num_ray_sensors < 1 or perception_radius <= 0:
            raise ValueError("Use a positive step limit, FPS, sensor range/count and at least two waypoints.")
        base_params, base = BaseEnv.init_params(key, map_id, max_steps, path_length,
                                                discretization_scale, perception_radius,
                                                num_ray_sensors, fov, fps)
        scale = RoboTaxi.metres_per_cell
        params_fields = {f.name: getattr(base_params, f.name) for f in fields(BaseEnvParams)}
        world_shape = tuple(scale * n for n in base_params.map_height_width)
        params_fields.update(map_height_width=world_shape,
                             nav_grid_shape_dtype=(world_shape, jnp.float32))
        params = TaxiParams(**params_fields)
        start = base.agent_pos * scale
        goal = base.goal_pos * scale
        if route_mode not in ("validated", "snapshot"):
            raise ValueError("route_mode must be validated or snapshot")

        def world_path(grid_path):
            return jnp.concatenate((start[None, :], grid_path[:-1] * scale))

        def cuts_wall(path):
            relative = path[:, None, :] - (scale * base.static_obstacles)[None, :, :]
            return bool(jnp.any(swept_contact(relative[:-1], relative[1:], 2 * RADIUS)))

        path = world_path(base.path_array)
        if route_mode == "validated" and cuts_wall(path):
            # A moving row can close a corridor in the planning snapshot. The
            # same course planner can find a permanent corridor using walls only.
            path = world_path(BaseEnv._find_path(base.agent_pos, base.goal_pos,
                                                 base.static_obstacles, base_params))
            if cuts_wall(path):
                raise ValueError("The course planner did not find a route with enough static clearance.")
        if not bool(jnp.allclose(path[-1], goal)):
            raise ValueError("Increase path_length so the cached course route includes the goal.")
        lengths = jnp.concatenate((jnp.zeros(1), jnp.cumsum(jnp.linalg.norm(jnp.diff(path, axis=0), axis=1))))
        progress, cross_track, waypoint = route_at(start, path, lengths, params.lookahead)
        zero = jnp.float32(0)
        state = TaxiState(time=base.time, agent_pos=start, goal_pos=goal,
                          agent_forward_dir=base.agent_forward_dir, heading=jnp.arctan2(
                              base.agent_forward_dir[1], base.agent_forward_dir[0]),
                          static_obstacles=base.static_obstacles * scale,
                          kinematic_obstacles=base.kinematic_obstacles * scale,
                          kinematic_obst_velocities=base.kinematic_obst_velocities,
                          path_array=path, route_lengths=lengths, progress=progress,
                          cross_track=cross_track, waypoint=waypoint,
                          speed=zero, acceleration=zero, steering=zero)
        return params, state

    @staticmethod
    def sample_start(key, params, initial, short_start_probability=0.0):
        """Vary traffic phase and initial heading. Short starts are for training only."""
        heading_key, phase_key, use_short_key, position_key = jax.random.split(key, 4)
        yaw = jax.random.uniform(heading_key, (), minval=-0.25, maxval=0.25)
        phase = jax.random.uniform(phase_key, (), minval=0., maxval=6.)
        world_size = jnp.asarray(params.map_height_width[::-1])
        moving = (initial.kinematic_obstacles + initial.kinematic_obst_velocities * phase) % world_size
        # Choose a waypoint before the last few metres of the route.
        arc = jax.random.uniform(position_key, (), minval=0.12, maxval=0.85) * initial.route_lengths[-1]
        index = jnp.argmin(jnp.abs(initial.route_lengths - arc))
        candidate = initial.path_array[index]
        obstacles = jnp.concatenate((initial.static_obstacles, moving))
        clear = jnp.all(jnp.linalg.norm(candidate - obstacles, axis=1) > 2 * RADIUS + 0.25)
        short = (jax.random.uniform(use_short_key) < short_start_probability) & clear
        position = jnp.where(short, candidate, initial.agent_pos)
        progress, cross_track, waypoint = route_at(position, initial.path_array, initial.route_lengths, params.lookahead)
        tangent = waypoint - position
        angle = jnp.where(short, jnp.arctan2(tangent[1], tangent[0]), initial.heading) + yaw
        return eqx.tree_at(lambda s: (s.agent_pos, s.heading, s.agent_forward_dir,
                                       s.kinematic_obstacles, s.progress, s.cross_track, s.waypoint),
                           initial, (position, angle, direction(angle), moving, progress, cross_track, waypoint))

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def reset(key, env_params, init_state):
        state = RoboTaxi.sample_start(key, env_params, init_state)
        return RoboTaxi.get_observation(state, env_params), state

    @staticmethod
    def get_observation(env_state, env_params):
        s, p = env_state, env_params
        obstacles = jnp.concatenate((s.static_obstacles, s.kinematic_obstacles))
        rays, ranges = BaseEnv._collision_ray_intersections(s.agent_pos, s.agent_forward_dir, obstacles, p)
        goal = s.goal_pos - s.agent_pos
        waypoint = s.waypoint - s.agent_pos
        features = jnp.concatenate((
            jnp.atleast_1d(s.speed / p.max_speed), s.agent_forward_dir,
            jnp.atleast_1d(jnp.clip(jnp.linalg.norm(goal) / math.hypot(*p.map_height_width), 0, 1)),
            relative_direction(goal, s.agent_forward_dir),
            relative_direction(waypoint, s.agent_forward_dir),
            jnp.atleast_1d(jnp.clip(s.cross_track / p.perception_radius, 0, 1)),
            ranges / p.perception_radius))
        return TaxiObservation(distance_to_path=s.cross_track,
                               direction_of_path=waypoint / jnp.maximum(jnp.linalg.norm(waypoint), 1e-6),
                               collision_rays=rays * ranges[:, None], features=features)

    @staticmethod
    @partial(jax.jit, static_argnames=("env_params",))
    def step(key, env_state, action, env_params):
        del key  # the transition is deterministic given the current world and action
        s, p = env_state, env_params
        throttle, turn = jnp.clip(action, -1., 1.)
        dt = p.step_size
        drive = jnp.where(throttle >= 0, p.max_acceleration * throttle, p.max_braking * throttle)
        resistance = p.rolling_resistance * (s.speed > 0) + p.drag * s.speed**2
        net_acceleration = jnp.clip(drive - resistance, -p.max_braking, p.max_acceleration)
        speed = jnp.clip(s.speed + dt * net_acceleration, 0., p.max_speed)
        steering = turn * p.max_steering
        travel = 0.5 * (s.speed + speed) * dt
        rotation = travel * jnp.tan(steering) / p.wheelbase
        position = s.agent_pos + travel * direction(s.heading + rotation / 2)
        heading = jnp.arctan2(jnp.sin(s.heading + rotation), jnp.cos(s.heading + rotation))
        moving = BaseEnv._move_kinematic_obstacles(s, p)
        progress, cross_track, waypoint = route_at(position, s.path_array, s.route_lengths, p.lookahead)
        new = eqx.tree_at(lambda t: (t.time, t.agent_pos, t.agent_forward_dir, t.heading, t.speed,
                                     t.acceleration, t.steering, t.kinematic_obstacles, t.progress,
                                     t.cross_track, t.waypoint), s,
                          (s.time + 1, position, direction(heading), heading, speed,
                           (speed - s.speed) / dt, steering, moving, progress, cross_track, waypoint))

        static_hit = swept_contact(s.agent_pos - s.static_obstacles, position - s.static_obstacles, 2 * RADIUS)
        # Test continuous motion on the exit side. Test the entry position after wrapping
        # separately, so a wrapping obstacle never sweeps across the whole map.
        unwrapped = s.kinematic_obstacles + dt * s.kinematic_obst_velocities
        moving_hit = swept_contact(s.agent_pos - s.kinematic_obstacles, position - unwrapped, 2 * RADIUS)
        entry_hit = jnp.sum((position - moving)**2, axis=1) <= (2 * RADIUS)**2
        world_size = jnp.asarray(p.map_height_width[::-1])
        outside = jnp.any((position < RADIUS) | (position > world_size - RADIUS))
        collision = jnp.any(static_hit) | jnp.any(moving_hit | entry_hit) | outside
        success = swept_contact(s.agent_pos - s.goal_pos, position - s.goal_pos, RADIUS) & ~collision
        terminated = collision | success
        truncated = (new.time >= p.max_steps_in_episode) & ~terminated
        distance = jnp.linalg.norm(position - s.goal_pos)
        # Differences telescope. Driving back and forth earns no net progress.
        reward = (0.75 * (progress - s.progress) + 0.25 * (jnp.linalg.norm(s.agent_pos - s.goal_pos) - distance)
                  - 0.20 * (cross_track - s.cross_track) - 0.02 - 0.08 * (speed < 0.25)
                  - 0.01 * jnp.abs(steering - s.steering) / p.max_steering
                  + 30. * success - 24. * collision)
        obs = RoboTaxi.get_observation(new, p)
        info = dict(time=new.time, distance_to_path=cross_track, direction_of_path=obs.direction_of_path,
                    distance_to_goal=distance, success=success, collision=collision,
                    terminated=terminated, truncated=truncated)
        return obs, new, reward, terminated | truncated, info
