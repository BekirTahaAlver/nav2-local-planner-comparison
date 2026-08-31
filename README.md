# Comparative Performance Analysis of Local Planners in Autonomous Mobile Robots

Companion code repository for an MSc thesis (İTÜ, Department of Control and
Automation Engineering) comparing three ROS 2 Nav2 local planners — **DWA**,
**MPPI**, and **RPP** — on a simulated TurtleBot3 Burger across four test
scenarios (Static, Dynamic, Narrow-U, Narrow-Z).

## Overview

This repository contains the ROS 2 package, test automation scripts, and
analysis tools used to:

- Run a parameter sensitivity analysis for each local planner.
- Benchmark the three planners under identical navigation conditions across
  four scenarios.
- Compute and compare performance metrics: success rate, completion time,
  path length, linear/angular velocity smoothness, minimum obstacle
  distance, and cross-track error (CTE RMSE).

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Nav2
- Gazebo Classic
- TurtleBot3 packages (`turtlebot3`, `turtlebot3_simulations`)
- Python 3.10+ with `pandas`, `numpy`, `matplotlib`

## Repository Structure

```
.
├── src/
│   └── my_config/              # ROS 2 package
│       ├── launch/             # launch files per planner/scenario
│       ├── params/             # DWA / MPPI / RPP navigation parameter YAMLs
│       ├── maps/                # occupancy grid maps (.pgm/.yaml)
│       └── worlds/              # Gazebo world files
├── scripts/
│   ├── automated_test.py       # runs a configured planner in a scenario, logs CSV
│   └── analyze_results.py      # computes metrics from logged CSVs
├── results/
│   └── sample/                  # small example CSVs for a quick sanity check
├── figures/                     # example plots (trajectory/velocity profiles)
└── docs/
    └── metrics.md                # metric definitions and formulas
```

## Setup

```bash
# clone into a ROS 2 workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <repo-url> my_config

cd ~/ros2_ws
colcon build --packages-select my_config
source install/setup.bash
```

## Running an Experiment

```bash
# example: DWA in the static scenario
ros2 launch my_config navfn_dwa_static_launch.py

# in a separate terminal, run the automated test script
python3 scripts/automated_test.py --planner dwa --scenario static --trials 10
```

Each run logs robot pose (odometry and ground truth), velocity commands, and
minimum laser scan distance to a CSV file.

## Analyzing Results

```bash
python3 scripts/analyze_results.py --input results/ --output results/summary.csv
```

This computes, per planner and scenario:

| Metric | Description |
|---|---|
| Success rate (%) | Fraction of trials reaching the goal without collision, becoming stuck, or timing out |
| Completion time (s) | Total trial duration |
| Path length (m) | Cumulative distance traveled |
| Linear velocity smoothness (m/s²) | Mean absolute linear acceleration |
| Angular velocity smoothness (rad/s²) | Mean absolute angular acceleration |
| Minimum obstacle distance (m) | Closest approach to an obstacle (net of robot footprint) |
| CTE RMSE (m) | Root-mean-square cross-track error relative to the global plan |

See `docs/metrics.md` for full definitions and equations.

## Test Scenarios

| Scenario | Description |
|---|---|
| Static | Open 6×6 m arena with five fixed cylindrical obstacles |
| Dynamic | Open arena with moving obstacles crossing the robot's path |
| Narrow-U | Narrow corridor with axis-aligned (90°) turns |
| Narrow-Z | Narrow corridor with identical geometry to Narrow-U but diagonal (135°) turns |

## Citation

If you use this code, please cite the associated thesis:

```
B. T. Alver, "Comparative Performance Analysis of Local Planners in
Autonomous Mobile Robots," M.Sc. thesis, Istanbul Technical University,
Department of Control and Automation Engineering, 2026.
```

## License

This repository is private and intended for academic reproducibility
purposes. Contact the author for reuse permissions.
