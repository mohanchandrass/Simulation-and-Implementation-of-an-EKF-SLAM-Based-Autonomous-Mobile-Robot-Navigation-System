# EKF-SLAM with Motion Control

This project implements a complete EKF-SLAM pipeline with motion control for a differential-drive robot in a 2D indoor environment.

## Features
- Differential-drive kinematics
- EKF-SLAM (prediction and update)
- Range-bearing landmark observations
- Waypoint-based motion control
- Obstacle avoidance
- Pygame-based visualization
- Comparison: Odometry vs EKF-SLAM

## Environment
- Room size: 10m × 8m
- Five static landmarks
- Three navigation goals

## How to Run
```bash
python run_simulation.py
