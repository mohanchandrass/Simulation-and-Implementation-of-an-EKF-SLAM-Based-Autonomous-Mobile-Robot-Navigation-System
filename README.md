# EKF-SLAM with Motion Control for a Differential-Drive Robot

## 1. Project Overview

This project implements a complete **Extended Kalman Filter Simultaneous Localization and Mapping (EKF-SLAM)** pipeline integrated with a **motion control architecture** for a differential-drive mobile robot operating in a two-dimensional indoor environment.

The system estimates both the robot pose and the positions of static landmarks while autonomously navigating through a sequence of predefined goal points. The implementation explicitly models uncertainty arising from noisy odometry and noisy range–bearing sensor measurements and demonstrates the advantages of EKF-SLAM over odometry-only localization.

The focus of this project is on **algorithmic implementation**, not on the use of high-level robotics libraries.

---

## 2. Environment Description

- **Room dimensions:** 10 m × 8 m (scaled to pixels for simulation)
- **Landmarks:** Five static point landmarks
  - A (0, 0) – Wall corner
  - B (10, 0) – Opposite wall corner
  - C (4, 3) – Pillar
  - D (8, 5) – Table corner
  - E (2, 7) – Cabinet edge
- **Goals:**
  - G₁ = (2, 2)
  - G₂ = (8, 4)
  - G₃ = (5, 7)

The environment contains static obstacles and supports collision checking and sensor visibility constraints.

---

## 3. Robot Model

### 3.1 Differential-Drive Kinematics

The robot is modeled as a differential-drive system with control inputs:

- Linear velocity: `v`
- Angular velocity: `ω`

Continuous-time motion model:

ẋ = v cos θ  
ẏ = v sin θ  
θ̇ = ω  

Discrete-time model (sampling interval Δt):

xₖ₊₁ = xₖ + vₖ Δt cos θₖ  
yₖ₊₁ = yₖ + vₖ Δt sin θₖ  
θₖ₊₁ = θₖ + ωₖ Δt  

---

## 4. Sensor Model

### 4.1 Odometry

- Derived from applied control inputs `(v, ω)`
- Corrupted by zero-mean Gaussian noise
- Used as input to the EKF prediction step

### 4.2 Range–Bearing Sensor

The robot observes landmarks using a simulated **range–bearing sensor**, providing:

- Range: `r`
- Bearing: `φ`

Measurement model for landmark *i*:

rᵢ = √((xᵢ − xᵣ)² + (yᵢ − yᵣ)²) + nᵣ  
φᵢ = atan2(yᵢ − yᵣ, xᵢ − xᵣ) − θᵣ + nφ  

where nᵣ and nφ are zero-mean Gaussian noise terms.

This sensor is functionally equivalent to a **2D LiDAR landmark detector** or **beacon-based indoor localization sensor**.

---

## 5. EKF-SLAM Architecture

### 5.1 State Representation

The EKF-SLAM state vector is defined as:

x = [xᵣ, yᵣ, θᵣ, x₁, y₁, …, xₙ, yₙ]ᵀ  

- Robot pose: `(xᵣ, yᵣ, θᵣ)`
- Landmark positions: `(xᵢ, yᵢ)`
- State dimension: `3 + 2n`

The covariance matrix:

P ∈ ℝ(3+2n)×(3+2n)

tracks uncertainty and cross-correlation between robot and landmarks.

---

### 5.2 Prediction Step

**Inputs**
- Control commands `(v, ω)`
- Time step `Δt`

**Operations**
1. Propagate robot pose using differential-drive motion model
2. Compute motion Jacobian Gᵣ
3. Construct full Jacobian G
4. Update covariance:

P̄ₖ = G Pₖ₋₁ Gᵀ + Q  

5. Symmetrize covariance to prevent numerical drift

**Outputs**
- Predicted state `x̄ₖ`
- Predicted covariance `P̄ₖ`

---

### 5.3 Update Step

For each observation `(r, φ, landmark_id)`:

1. Validate measurement range and bearing
2. If the landmark is **new**:
   - Initialize landmark position
   - Augment state vector and covariance matrix
3. If the landmark is **known**:
   - Compute predicted measurement
   - Compute innovation
   - Construct observation Jacobian `H`
   - Compute innovation covariance `S`
   - Perform Mahalanobis consistency check
   - Compute Kalman gain `K`
   - Update state and covariance

Observation counts are maintained to increase confidence in repeatedly observed landmarks.

---

## 6. Motion Control Architecture

### 6.1 High-Level Planning

- Maintains a list of predefined global goals
- Goals are visited sequentially
- A goal is considered reached only when the estimated robot position remains within a tolerance region for multiple frames, ensuring robustness to noise
- Intermediate waypoints may be generated if obstacles block the direct path

---

### 6.2 Mid-Level Motion Control

**Inputs**
- Estimated pose `(x̂, ŷ, θ̂)` from EKF-SLAM
- Current goal or waypoint
- Nearby obstacle information
- Localization uncertainty

**Error terms**
- Distance error: `dₑ`
- Heading error: `θₑ`

**Control laws**
v = Kd · dₑ  
ω = Kθ · θₑ  

Linear velocity is adaptively reduced during sharp turns, near obstacles, or under high localization uncertainty.

---

### 6.3 Low-Level Execution

- Applies velocity commands to the robot
- Uses differential-drive kinematics
- Performs collision checking before accepting motion updates
- Motion is suppressed and replanning is triggered if a collision is detected

---

## 7. Control Loop Execution

The system executes at **30 Hz** with the following loop:

1. **Sense:** Acquire noisy odometry and landmark observations
2. **Estimate:** EKF prediction and update
3. **Plan:** Compute desired heading and distance to goal
4. **Control:** Generate velocity commands
5. **Execute:** Apply motion with collision checking
6. **Visualize:** Update simulation display

---

## 8. Simulation and Visualization

The simulation is implemented using **Pygame** and visualizes:

- True robot trajectory
- Odometry-only trajectory
- EKF-estimated trajectory
- True and estimated landmark positions
- Sensor rays and bearing lines
- Controller state and error statistics

---

## 9. Results and Observations

- EKF-SLAM significantly outperforms odometry-only localization
- Landmark estimation accuracy depends on revisit frequency
- Landmarks not re-observed after initialization exhibit higher final error
- The EKF-estimated robot trajectory closely follows the planned path
- The true robot may lag slightly due to noise and unmodeled disturbances

---

## 10. Limitations and Future Work

**Current limitations**
- No active exploration strategy
- Fixed noise parameters
- Limited landmark re-observations

**Future improvements**
- Active SLAM exploration
- Adaptive noise estimation
- Multi-sensor fusion
- Deployment on a real robot platform

---

## 11. How to Run

```bash
python run_simulation.py
```

##12. Repository Structure
project/
├── run_simulation.py
├── ekf_slam.py
├── controller.py
├── environment.py
├── plots/
├── README.md
└── report/
