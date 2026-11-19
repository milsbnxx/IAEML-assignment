# Assignment 1 · The robo-taxi environment

**Author: Sabina Yamilova · Innopolis University · IAEML**

The taxi starts from rest. Each 0.125-second step takes two numbers from a driver and returns what the taxi can sense, a reward and an episode-ending flag. In the recorded example a simple steering controller reaches the goal in 134 steps. The car, moving obstacles, sensor rays and driven path are visible in the movie. [Environment demonstration](../videos/environment_demo.mp4)

## World and interface

`RoboTaxi` inherits from `BaseEnv`. It uses the three course maps, the course grid planner, ray-circle intersections and the motion of kinematic obstacles. The new state and observation extend the course types. The calls keep the course order of arguments and return values. [Environment source](../src/env/robotaxi.py) · [Original course code](https://github.com/InnoRL/IAEML-assignment/tree/1056d711e166747cc485725e31f2535967b030dd)

| Element | Definition |
|---|---|
| World | A 20 × 10 grid covers 60 × 30 m. One map cell spans 3 m. The car, target and obstacle discs each have a radius of 1 m. |
| State | Position, heading, forward unit vector, speed, applied acceleration, front steering angle and step count. The state also holds obstacle geometry and velocities, the goal and the cached route. |
| Action | Two values in [-1, 1]. The first requests acceleration or braking. The second sets the front steering angle. |
| Observation | 25 floats. One speed, two heading components, one goal distance, two components of the relative goal direction, two of the relative waypoint direction, one route offset and sixteen obstacle ranges. |
| Ending | Goal arrival, collision or a time limit of 192 steps (24 s). A collision takes priority if both contacts happen in one step. |

The course start on map 0 is one cell from a wall. Two one-metre discs would overlap with one-metre cell spacing. Using three metres per cell keeps the course layout and provides one metre of clearance at that start. A static obstacle marks each wall cell. Moving obstacles travel at 1 m/s and wrap at the edge of the world. The taxi stays inside the world with one metre of clearance. Entering the goal requires the taxi's reference point to pass within 1 m of the goal centre.

## Vehicle dynamics

The car uses a low-speed kinematic bicycle model. Its position represents the rear-axle reference point and the centre of the collision disc. With wheelbase 1.6 m and steering angle $\delta$, the heading changes by $\Delta\psi=s\tan(\delta)/1.6$, where $s$ is the distance travelled in this step. This captures two effects that matter here. A tighter steering angle makes a smaller turning circle, and turning the wheel while stopped leaves the heading fixed. The model assumes a flat surface, instant steering and rolling tyres without sideways slip. It omits reverse gear and wheel dynamics. [Bicycle model and derivation](https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html)

1. Clip the two action components. Positive throttle requests up to 3.5 m/s². Full braking requests −5 m/s². The maximum steering angle is 0.60 rad, about 34°.
2. Subtract rolling resistance of 0.15 m/s² while moving and drag of $0.018v^2$. These are resistance forces per unit mass. Limit the net acceleration to [−5, 3.5] m/s² and speed to [0, 7] m/s.
3. Multiply the average of the old and new speed by the time step to find $s$. Move along the midpoint heading $\psi+\Delta\psi/2$. Store the actual change in speed per second as the acceleration.
4. Move the other vehicles, check both kinds of contact, then calculate the reward and the new observation. For a collision, use the closest approach during the step and a combined radius of 2 m. The straight segment approximates the small arc over a single step.

## What the policy sees

Goal and waypoint directions use a forward/left frame. For example, a positive left component means the point is to the driver's left. Speed is divided by 7, goal distance by the world diagonal, and route offset by 9 m and clipped to one. The sixteen rays cover 360° and measure the distance from the car centre to the first obstacle surface up to 9 m. Each range is divided by 9. The vector stays within [-1, 1].

The course planner supplies a route at initialization. The taxi projects its position onto the nearest segment and takes a point 3.8 m ahead. The default route check also tests the clearance along each segment. If the route crosses a fixed obstacle, the same planner runs once more using fixed obstacles. Live rays track moving obstacles on every step. This keeps planning out of the training loop and makes the problem one of driving with navigation assistance.

## Reward

Let $s$ denote distance along the planned route, $d$ the direct distance to the goal and $e$ the distance to the route. A prime denotes the new state.

| Term per step | Purpose |
|---|---|
| $0.75(s'-s)$ | Reward progress around a wall, even when the detour initially points away from the goal. |
| $0.25(d-d')$ | Give extra feedback for moving closer to the destination. |
| $0.20(e-e')$ | Reward returning towards the route. |
| −0.02 every step, with an extra −0.08 when the new speed is below 0.25 m/s | Use time efficiently. A stationary taxi loses 0.10 each step. |
| $-0.01\lvert\delta'-\delta\rvert/0.60$ | Reduce unnecessary changes in steering. |
| +30 on arrival or −24 on collision | Make the outcome matter. A timeout receives the ordinary last-step reward. |

The three progress terms use differences. On an undiscounted round trip to the same position they add up to zero. The time and control costs still accumulate. A test checks equal-distance outward and return movements. Another checks that forward progress on a clear road earns more than waiting. Waiting briefly for crossing traffic can still be a reasonable action.

## Technical difficulties

- The course observation stores a scalar, a direction and a ray matrix. Calling its default vector conversion raised a dimension error. The subclass builds one flat policy vector and keeps the original ray displacements for drawing.
- On map 2 the first planner route passed directly through a fixed obstacle at grid position (5, 1). Segment clearance checks caught it. Planning with static obstacles gave a clear route while moving obstacles remained in the simulation.
- A moving disc can leave one edge and appear at the opposite edge. Sweeping straight between those two coordinates would give a false collision in the middle. The transition checks the continuous exit motion and the wrapped entry position separately.
- The renderer originally drew markers half a cell wide. The off-screen renderer uses physical metres for their radii and reuses the course ray overlay. The course wrapper now keeps its initial array state as normal Equinox leaves. [Technical notes and test evidence](technical_notes.md)

## Reproduce

Use the installation commands in the [project README](../README.md). From the repository root run the following commands.

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python src/demo.py --policy controller --map 0 --seed 2000
```

The movie uses a pure-pursuit steering controller and a 4 m/s target speed. The controller sees the waypoint and car state. On the recorded seed, its return is 81.820 and its final goal distance is 0.709 m. The initial and final frames are included, with a one-second hold at the end. [Video](../videos/environment_demo.mp4) · [Controller and recorder](../src/demo.py)

## Personal Reflection

*For Sabina to complete in her own words.* Choose one observed problem above. Explain how you recognized it, what you checked and what you would try on a different map.
