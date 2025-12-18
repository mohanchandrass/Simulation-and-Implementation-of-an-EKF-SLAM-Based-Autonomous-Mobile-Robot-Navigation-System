"""
controller.py - Enhanced Smooth Controller with Path Planning
Implements look-ahead, predictive control, and dynamic replanning
"""

import math
import numpy as np
from ekf_slam import wrap_to_pi


class PathPlanner:
    """Simple path planner for obstacle avoidance"""
    
    def __init__(self, environment):
        self.env = environment
    
    def is_path_blocked(self, start, goal, obstacles, clearance=50.0):
        """Check if direct path to goal is blocked by obstacles"""
        sx, sy = start
        gx, gy = goal
        
        # Check each obstacle
        for ox, oy in obstacles:
            # Distance from obstacle to line segment
            dist = self._point_to_segment_distance(ox, oy, sx, sy, gx, gy)
            if dist < clearance:
                return True
        
        # Check dynamic obstacles from environment
        for obs in self.env.dynamic_obstacles:
            dist = self._point_to_segment_distance(
                obs['x'], obs['y'], sx, sy, gx, gy
            )
            if dist < obs['radius'] + clearance:
                return True
        
        return False
    
    def find_clear_waypoint(self, start, goal, obstacles, num_samples=12):
        """Find intermediate waypoint to avoid obstacles"""
        sx, sy = start
        gx, gy = goal
        
        # Vector from start to goal
        dx = gx - sx
        dy = gy - sy
        dist = math.hypot(dx, dy)
        
        if dist < 10:
            return None
        
        # Try waypoints in a circle around the midpoint
        mid_x = sx + dx * 0.5
        mid_y = sy + dy * 0.5
        
        best_waypoint = None
        best_score = float('inf')
        
        # Sample points around the obstacle
        search_radius = min(150, dist * 0.6)
        
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            wx = mid_x + search_radius * math.cos(angle)
            wy = mid_y + search_radius * math.sin(angle)
            
            # Check if waypoint is valid
            if self.env.check_collision((wx, wy), 30):
                continue
            
            # Check if paths to/from waypoint are clear
            if self.is_path_blocked((sx, sy), (wx, wy), obstacles, clearance=40):
                continue
            
            if self.is_path_blocked((wx, wy), (gx, gy), obstacles, clearance=40):
                continue
            
            # Score based on total path length
            dist1 = math.hypot(wx - sx, wy - sy)
            dist2 = math.hypot(gx - wx, gy - wy)
            score = dist1 + dist2
            
            if score < best_score:
                best_score = score
                best_waypoint = (wx, wy)
        
        return best_waypoint
    
    def _point_to_segment_distance(self, px, py, x1, y1, x2, y2):
        """Calculate minimum distance from point to line segment"""
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)))
        
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        
        return math.hypot(px - proj_x, py - proj_y)


class WaypointTrackingController:
    """
    Enhanced waypoint controller with smooth control and look-ahead
    """
    
    def __init__(self, goals, Kd=0.8, Ktheta=3.5, goal_tolerance=35.0, 
                 min_obstacle_dist=70.0, max_linear_vel=15.0, max_angular_vel=1.5,
                 look_ahead_dist=80.0):
        """
        Initialize enhanced controller
        
        Args:
            look_ahead_dist: Distance to look ahead for smoother turning
        """
        # Goal management
        self.goals = goals
        self.current_goal_idx = 0
        self.goal_reached_frames = 0
        self.intermediate_waypoint = None  # For obstacle avoidance
        
        # Control gains
        self.Kd = Kd
        self.Ktheta = Ktheta
        
        # Parameters
        self.goal_tolerance = goal_tolerance
        self.min_obstacle_dist = min_obstacle_dist
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        self.look_ahead_dist = look_ahead_dist
        
        # State tracking
        self.total_distance_traveled = 0.0
        self.last_position = None
        self.stuck_counter = 0
        self.recovery_mode = False
        self.recovery_timer = 0
        
        # Smoothing
        self.last_v = 0.0
        self.last_w = 0.0
        self.velocity_alpha = 0.6  # Low-pass filter coefficient
        
        # Path history
        self.path_history = []
        
    def set_intermediate_waypoint(self, waypoint):
        """Set intermediate waypoint for obstacle avoidance"""
        self.intermediate_waypoint = waypoint
        print(f"→ Intermediate waypoint set at ({waypoint[0]:.0f}, {waypoint[1]:.0f})")
    
    def clear_intermediate_waypoint(self):
        """Clear intermediate waypoint"""
        if self.intermediate_waypoint:
            self.intermediate_waypoint = None
    
    def compute_control_smooth(self, est_pose, covariance=None, nearby_obstacles=None, dt=0.1):
        """
        Compute smooth control with look-ahead and predictive behavior
        """
        # Check if finished
        if self.finished():
            return 0.0, 0.0
        
        x, y, theta = est_pose
        
        # Record path
        self.path_history.append((x, y))
        if len(self.path_history) > 10000:
            self.path_history.pop(0)
        
        # Track distance
        if self.last_position is not None:
            dist_moved = math.hypot(x - self.last_position[0], y - self.last_position[1])
            self.total_distance_traveled += dist_moved
        
        # Determine target (intermediate waypoint or goal)
        if self.intermediate_waypoint:
            gx, gy = self.intermediate_waypoint
            target_tolerance = 40.0
            
            # Check if intermediate waypoint reached
            dist_to_intermediate = math.hypot(gx - x, gy - y)
            if dist_to_intermediate < target_tolerance:
                print("✓ Intermediate waypoint reached")
                self.intermediate_waypoint = None
                gx, gy = self.goals[self.current_goal_idx]
        else:
            gx, gy = self.goals[self.current_goal_idx]
        
        # Calculate errors
        dx = gx - x
        dy = gy - y
        de = math.hypot(dx, dy)
        
        # Goal reached check
        if de < self.goal_tolerance and not self.intermediate_waypoint:
            self.goal_reached_frames += 1
            if self.goal_reached_frames > 5:
                print(f"✓ Goal G{self.current_goal_idx + 1} reached")
                self.current_goal_idx += 1
                self.goal_reached_frames = 0
                self.stuck_counter = 0
                self.recovery_mode = False
                self.intermediate_waypoint = None
                
                if not self.finished():
                    next_gx, next_gy = self.goals[self.current_goal_idx]
                    print(f"→ Next goal: G{self.current_goal_idx + 1}")
            
            # Smooth deceleration
            v = self.Kd * de * 0.3
            w = 0.0
            
            # Apply smoothing
            v = self.velocity_alpha * self.last_v + (1 - self.velocity_alpha) * v
            w = self.velocity_alpha * self.last_w + (1 - self.velocity_alpha) * w
            
            self.last_v = v
            self.last_w = w
            return v, w
        else:
            self.goal_reached_frames = 0
        
        # Recovery mode
        if self.recovery_mode:
            self.recovery_timer += 1
            if self.recovery_timer < 12:
                return -6.0, 0.0  # Back up
            elif self.recovery_timer < 25:
                return 0.0, 0.9  # Turn
            else:
                self.recovery_mode = False
                self.recovery_timer = 0
                self.stuck_counter = 0
                self.intermediate_waypoint = None
                print("→ Recovery complete")
                return 0.0, 0.0
        
        # Look-ahead point for smoother turning
        if de > self.look_ahead_dist:
            # Use look-ahead point
            look_ahead_ratio = self.look_ahead_dist / de
            look_gx = x + dx * look_ahead_ratio
            look_gy = y + dy * look_ahead_ratio
        else:
            # Close to goal, aim directly
            look_gx = gx
            look_gy = gy
        
        # Calculate heading to look-ahead point
        look_dx = look_gx - x
        look_dy = look_gy - y
        desired_heading = math.atan2(look_dy, look_dx)
        theta_e = wrap_to_pi(desired_heading - theta)
        
        # Obstacle avoidance with smoother repulsion
        avoidance_angle = 0.0
        obstacle_detected = False
        
        if nearby_obstacles and len(nearby_obstacles) > 0:
            avoidance_angle, obstacle_detected = self._compute_smooth_avoidance(
                x, y, theta, nearby_obstacles, desired_heading
            )
        
        # Combine with obstacle avoidance
        final_theta_e = wrap_to_pi(theta_e + avoidance_angle)
        
        # Proportional control
        vc = self.Kd * de
        wc = self.Ktheta * final_theta_e
        
        # Adaptive speed based on heading error (slow down when turning)
        if abs(final_theta_e) > math.radians(60):
            vc *= 0.3  # Very sharp turn
        elif abs(final_theta_e) > math.radians(30):
            vc *= 0.5  # Sharp turn
        elif abs(final_theta_e) > math.radians(15):
            vc *= 0.7  # Moderate turn
        
        # Slow down near obstacles
        if obstacle_detected:
            vc *= 0.6
        
        # Slow down based on uncertainty
        if covariance is not None:
            position_uncertainty = math.sqrt(covariance[0, 0] + covariance[1, 1])
            uncertainty_factor = max(0.6, 1.0 - position_uncertainty / 60.0)
            vc *= uncertainty_factor
        
        # Apply velocity limits
        v = np.clip(vc, 0.0, self.max_linear_vel)
        w = np.clip(wc, -self.max_angular_vel, self.max_angular_vel)
        
        # Smooth velocity changes with low-pass filter
        v = self.velocity_alpha * self.last_v + (1 - self.velocity_alpha) * v
        w = self.velocity_alpha * self.last_w + (1 - self.velocity_alpha) * w
        
        # Stuck detection
        if self.last_position is not None:
            moved = math.hypot(x - self.last_position[0], y - self.last_position[1])
            
            if moved < 0.4 and de > self.goal_tolerance:
                self.stuck_counter += 1
                if self.stuck_counter > 35:
                    print("⚠ Robot stuck - initiating recovery")
                    self.recovery_mode = True
                    self.recovery_timer = 0
                    self.intermediate_waypoint = None
                    return -6.0, 0.0
            else:
                self.stuck_counter = max(0, self.stuck_counter - 3)
        
        self.last_position = (x, y)
        self.last_v = v
        self.last_w = w
        
        return v, w
    
    def _compute_smooth_avoidance(self, x, y, theta, obstacles, desired_heading):
        """
        Smooth obstacle avoidance using weighted potential field
        """
        avoidance_vec = np.array([0.0, 0.0])
        obstacle_detected = False
        total_weight = 0.0
        
        for ox, oy in obstacles:
            dx_obs = x - ox
            dy_obs = y - oy
            dist = math.hypot(dx_obs, dy_obs)
            
            if dist < self.min_obstacle_dist and dist > 1e-3:
                obstacle_detected = True
                
                # Weight decreases with distance (smoother)
                weight = ((self.min_obstacle_dist - dist) / self.min_obstacle_dist) ** 2
                
                # Repulsive force strength
                strength = weight * 2.5
                
                # Direction away from obstacle
                avoidance_vec[0] += strength * dx_obs / (dist + 1e-3)
                avoidance_vec[1] += strength * dy_obs / (dist + 1e-3)
                total_weight += weight
        
        if np.linalg.norm(avoidance_vec) > 1e-3:
            # Normalize avoidance vector
            avoidance_vec = avoidance_vec / (np.linalg.norm(avoidance_vec) + 1e-6)
            
            # Scale by obstacle proximity
            avoidance_strength = min(1.0, total_weight)
            
            # Convert to heading adjustment
            avoidance_heading = math.atan2(avoidance_vec[1], avoidance_vec[0])
            
            # Blend with desired heading (smoother integration)
            blend_factor = avoidance_strength * 0.5
            blended_heading = (1 - blend_factor) * desired_heading + blend_factor * avoidance_heading
            
            avoidance_angle = wrap_to_pi(blended_heading - desired_heading)
            
            # Limit avoidance angle for stability
            max_avoidance = math.radians(60)
            avoidance_angle = np.clip(avoidance_angle, -max_avoidance, max_avoidance)
            
            return avoidance_angle, obstacle_detected
        
        return 0.0, obstacle_detected
    
    # Keep original compute_control for compatibility
    def compute_control(self, est_pose, covariance=None, nearby_obstacles=None, 
                       estimated_landmarks=None):
        """Wrapper for backward compatibility"""
        return self.compute_control_smooth(est_pose, covariance, nearby_obstacles, dt=0.1)
    
    def get_path_history(self):
        """Get path history for visualization"""
        return self.path_history.copy()
    
    def finished(self):
        """Check if all goals reached"""
        return self.current_goal_idx >= len(self.goals)
    
    def get_current_goal(self):
        """Get current goal position"""
        if self.finished():
            return None
        return self.goals[self.current_goal_idx]
    
    def get_progress(self):
        """Get progress percentage"""
        if len(self.goals) == 0:
            return 100.0
        return (self.current_goal_idx / len(self.goals)) * 100.0
    
    def get_status_string(self):
        """Get status string"""
        if self.finished():
            return "Mission Complete ✓"
        elif self.recovery_mode:
            return f"RECOVERING (Goal {self.current_goal_idx + 1}/{len(self.goals)})"
        elif self.intermediate_waypoint:
            return f"Avoiding obstacle → G{self.current_goal_idx + 1}"
        elif self.stuck_counter > 20:
            return f"Navigating around obstacle (G{self.current_goal_idx + 1})"
        else:
            return f"Navigating to G{self.current_goal_idx + 1}/{len(self.goals)}"
    
    def reset(self):
        """Reset controller"""
        self.current_goal_idx = 0
        self.goal_reached_frames = 0
        self.stuck_counter = 0
        self.last_position = None
        self.total_distance_traveled = 0.0
        self.recovery_mode = False
        self.recovery_timer = 0
        self.intermediate_waypoint = None
        self.path_history = []
        self.last_v = 0.0
        self.last_w = 0.0