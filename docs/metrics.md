# Evaluation Metrics

Metrics adapted from Wen et al. (2021) for safety, efficiency, and smoothness;
cross-track error follows the tracking-accuracy approach used in comparable
Nav2 planner studies. Angular velocity smoothness is introduced in this study
to capture the rotational component of motion, which linear velocity
smoothness alone does not represent.

## Success Rate

A trial is recorded as successful if the robot reaches the goal without:

- colliding with an obstacle,
- becoming unable to make further progress (stuck), or
- exceeding the time limit.

## Completion Time (T)

$$T = t_N - t_1$$

where $t_1$ and $t_N$ are the timestamps of the first and last recorded poses.

## Path Length (L_p)

$$L_p = \sum_{i=1}^{N-1} \lVert \mathbf{x}_{i+1} - \mathbf{x}_i \rVert$$

where $\mathbf{x}_i$ is the robot's ground-truth position at step $i$.

## Linear Velocity Smoothness (f_vs)

$$f_{vs} = \frac{1}{N-1} \sum \left| \frac{v_{i+1} - v_i}{\Delta t} \right|$$

Mean absolute linear acceleration; lower values indicate smoother motion.

## Angular Velocity Smoothness (f_avs)

$$f_{avs} = \frac{1}{N-1} \sum \left| \frac{\omega_{i+1} - \omega_i}{\Delta t} \right|$$

Mean absolute angular acceleration; captures the smoothness of rotational
motion.

## Minimum Obstacle Distance (d_min)

$$d_{\min} = \min\{d_i\} - r_{\text{robot}}$$

Net clearance between the robot's footprint and the nearest obstacle, over
the course of a trial.

## Cross-Track Error RMSE

$$\mathrm{CTE}_{\mathrm{RMSE}} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} e_{\perp,i}^2}$$

where $e_{\perp,i}$ is the perpendicular distance from the robot's position
to the initially generated global reference path at step $i$.
