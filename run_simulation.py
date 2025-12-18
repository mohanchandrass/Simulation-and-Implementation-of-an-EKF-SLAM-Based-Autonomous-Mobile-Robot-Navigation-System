"""
run_simulation.py - Enhanced Interactive GUI with Odometry Comparison
Demonstrates EKF-SLAM vs Odometry-only navigation with comprehensive analysis
"""

import pygame
import math
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from ekf_slam import EKFSLAM, wrap_to_pi
from controller import WaypointTrackingController, PathPlanner
from environment import (Environment, draw_robot, draw_fov_cone, 
                         draw_measurement_lines, map_data, ROBOT_RADIUS,
                         GOALS_PIXELS, LANDMARKS_METERS, pixels_to_meters,
                         ROOM_WIDTH_PX, ROOM_HEIGHT_PX, MARGIN, LANDMARK_COLORS)

# ===================== WINDOW =====================

pygame.init()
SCREEN_W, SCREEN_H = 1200, 820
WIN = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("🤖 EKF-SLAM vs Odometry Navigation Comparison")

# ===================== LAYOUT =====================

ENV_W = 900
ENV_HEADER_H = 50
ENV_Y_OFFSET = ENV_HEADER_H

STATS_X = ENV_W + 10
STATS_W = SCREEN_W - STATS_X - 10

MINI_W = STATS_W
MINI_H = 200
MINI_X = STATS_X
MINI_Y = 10

LEG_H     = 150
CONTROL_H = 200
STATUS_H  = 210 

STATUS_Y  = MINI_Y + MINI_H + 10
CONTROL_Y = STATUS_Y + STATUS_H  + 10
LEG_Y = CONTROL_Y + CONTROL_H + 10

# Enhanced Colors
WHITE = (250, 250, 250)
TRUE_BLUE = (30, 120, 255)
EST_ORANGE = (255, 147, 30)
ODOM_GREEN = (50, 200, 100)
EST_LM = (255, 100, 255)
GOAL_COLOR = (80, 200, 80)
PATH_COLOR = (100, 200, 255)
FOV_COLOR = (30, 100, 220, 50)
MEASURE_COLOR = (50, 200, 200, 120)
PANEL_BG = (25, 28, 35)
BUTTON_BG = (45, 52, 65)
BUTTON_HOVER = (60, 70, 85)
BUTTON_ACTIVE = (80, 200, 120)
DANGER_BTN = (220, 60, 60)
PLANNED_PATH_COLOR = (255, 200, 0, 150)
GRID_COLOR = (40, 40, 45)

clock = pygame.time.Clock()
FPS = 30
DT = 0.1

# Fonts
FONT = pygame.font.SysFont("Arial", 14)
TITLE_FONT = pygame.font.SysFont("Arial", 18, bold=True)
SMALL_FONT = pygame.font.SysFont("Arial", 12)
BUTTON_FONT = pygame.font.SysFont("Arial", 13, bold=True)

# Tracking paths
est_path = []
true_path = []
odom_path = []

# Data collection for plotting
class SimulationData:
    def __init__(self):
        self.timestamps = []
        self.true_positions = []
        self.ekf_positions = []
        self.odom_positions = []
        self.ekf_errors = []
        self.odom_errors = []
        self.control_v = []
        self.control_w = []
        self.landmark_errors = []
        self.ekf_uncertainties = []
        
    def record(self, t, true_pose, ekf_pose, odom_pose, v, w, slam, true_landmarks):
        self.timestamps.append(t)
        self.true_positions.append(true_pose[:2].copy())
        self.ekf_positions.append(ekf_pose[:2].copy())
        self.odom_positions.append(odom_pose[:2].copy())
        
        ekf_err = np.linalg.norm(ekf_pose[:2] - true_pose[:2])
        odom_err = np.linalg.norm(odom_pose[:2] - true_pose[:2])
        self.ekf_errors.append(ekf_err)
        self.odom_errors.append(odom_err)
        
        self.control_v.append(v)
        self.control_w.append(w)
        
        # Landmark estimation error
        est_landmarks = slam.get_landmarks()
        if len(est_landmarks) > 0:
            lm_errors = []
            for i, true_lm in enumerate(true_landmarks):
                if i < len(est_landmarks):
                    err = np.linalg.norm(est_landmarks[i] - true_lm)
                    lm_errors.append(err)
            if lm_errors:
                self.landmark_errors.append(np.mean(lm_errors))
        
        # EKF uncertainty (trace of position covariance)
        P = slam.get_covariance()
        uncertainty = np.sqrt(P[0,0] + P[1,1])
        self.ekf_uncertainties.append(uncertainty)

sim_data = SimulationData()


class Button:
    """Modern button with hover effects"""
    def __init__(self, x, y, w, h, text, color, hover_color=BUTTON_HOVER, 
                 text_color=WHITE, icon=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color
        self.icon = icon
        self.hovered = False
        self.clicked = False
    
    def draw(self, surface):
        color = self.hover_color if self.hovered else self.color
        
        shadow = self.rect.copy()
        shadow.x += 2
        shadow.y += 2
        pygame.draw.rect(surface, (0, 0, 0, 50), shadow, border_radius=6)
        
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        pygame.draw.rect(surface, (255, 255, 255, 30), self.rect, 2, border_radius=6)
        
        text = BUTTON_FONT.render(self.text, True, self.text_color)
        text_rect = text.get_rect(center=self.rect.center)
        surface.blit(text, text_rect)
    
    def handle_event(self, event, mouse_pos):
        self.hovered = self.rect.collidepoint(mouse_pos)
        
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.hovered:
                self.clicked = True
                return True
        
        return False


def draw_environment_header(surface):
    """Draw header panel above the environment"""

    # Background panel
    pygame.draw.rect(
        surface,
        PANEL_BG,
        (0, 0, ENV_W, ENV_HEADER_H)
    )

    # Title
    title = TITLE_FONT.render("ENVIRONMENT MAP (10m × 8m)", True, WHITE)
    surface.blit(title, (12, 10))

    # Subtitle / info
    subtitle = SMALL_FONT.render(
        "Global reference frame • 1m : 80px",
        True,
        (180, 180, 180)
    )
    surface.blit(subtitle, (12, 30))

    # Divider line
    pygame.draw.line(
        surface,
        (60, 60, 60),
        (0, ENV_HEADER_H - 1),
        (ENV_W, ENV_HEADER_H - 1),
        1
    )

def draw_mini_map(surface, slam, controller):
    """
    Mini-map aligned EXACTLY with environment.py
    """
    x, y = MINI_X, MINI_Y
    w, h = STATS_W, MINI_H

    # ---------------- Frame ----------------
    pygame.draw.rect(surface, (8, 8, 8), (x-2, y-2, w+4, h+4), border_radius=8)

    mini = pygame.Surface((w, h), pygame.SRCALPHA)
    mini.fill(PANEL_BG)

    title = FONT.render("EKF Estimated Map", True, WHITE)
    mini.blit(title, (8, 6))

    # ---------------- MAP TRANSFORM ----------------
    pad = 16
    top_offset = 36

    usable_w = w - 2 * pad
    usable_h = h - top_offset - pad

    scale = min(
        usable_w / ROOM_WIDTH_PX,
        usable_h / ROOM_HEIGHT_PX
    )

    ox = pad
    oy = top_offset

    def world_to_mini(px, py):
        # Convert environment pixel → mini-map pixel
        wx = px - MARGIN
        wy = py - MARGIN
        mx = ox + wx * scale
        my = oy + wy * scale
        return int(mx), int(my)
    
    PIXELS_PER_METER = 80

    # ---------------- GRID (room-based) ----------------
    grid_px = PIXELS_PER_METER  # 1m grid
    for gx in range(0, ROOM_WIDTH_PX + 1, grid_px):
        mx, _ = world_to_mini(MARGIN + gx, MARGIN)
        pygame.draw.line(mini, GRID_COLOR, (mx, oy), (mx, oy + usable_h), 1)

    for gy in range(0, ROOM_HEIGHT_PX + 1, grid_px):
        _, my = world_to_mini(MARGIN, MARGIN + gy)
        pygame.draw.line(mini, GRID_COLOR, (ox, my), (ox + usable_w, my), 1)

    # ---------------- EKF DATA ----------------
    state, P = slam.get_state(), slam.get_covariance()
    ex, ey, eth = state[:3]
    est_landmarks = slam.get_landmarks()

    # ---------------- LANDMARK UNCERTAINTY ----------------
    for i, (lx, ly) in enumerate(est_landmarks):
        cov = P[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
        px, py = world_to_mini(lx, ly)
        draw_uncertainty_ellipse(
            mini,
            (px, py),
            cov * (scale**2),
            EST_LM,
            confidence=0.95,
            alpha=45
        )

    # ---------------- ROBOT UNCERTAINTY ----------------
    r_px, r_py = world_to_mini(ex, ey)
    draw_uncertainty_ellipse(
        mini,
        (r_px, r_py),
        P[0:2, 0:2] * (scale**2),
        EST_ORANGE,
        confidence=0.95,
        alpha=45
    )

    # ---------------- LANDMARK POINTS ----------------
    for lx, ly in est_landmarks:
        px, py = world_to_mini(lx, ly)
        pygame.draw.circle(mini, EST_LM, (px, py), 4)
        pygame.draw.circle(mini, WHITE, (px, py), 4, 1)

    # ---------------- ESTIMATED TRAJECTORY ----------------
    if len(est_path) > 1:
        pts = [world_to_mini(px, py) for px, py in est_path]
        pygame.draw.lines(mini, EST_ORANGE, False, pts, 2)

    # ---------------- ROBOT ----------------
    pygame.draw.circle(mini, EST_ORANGE, (r_px, r_py), 5)
    hx = r_px + int(10 * math.cos(eth))
    hy = r_py + int(10 * math.sin(eth))
    pygame.draw.line(mini, EST_ORANGE, (r_px, r_py), (hx, hy), 2)

    # ---------------- GOALS ----------------
    for i, (gx, gy) in enumerate(GOALS_PIXELS):
        g_px, g_py = world_to_mini(gx, gy)
        if i == controller.current_goal_idx:
            pygame.draw.circle(mini, GOAL_COLOR, (g_px, g_py), 5)
        elif i < controller.current_goal_idx:
            pygame.draw.circle(mini, (120, 160, 120), (g_px, g_py), 4)
        else:
            pygame.draw.circle(mini, (160, 160, 160), (g_px, g_py), 4, 1)

    # ---------------- BLIT ----------------
    surface.blit(mini, (x, y))
    pygame.draw.rect(surface, (60, 60, 60), (x-1, y-1, w+2, h+2), 2, border_radius=8)

def draw_legend(surface):
    """Updated legend with odometry"""
    x, y = STATS_X, LEG_Y
    w = STATS_W
    panel_h = LEG_H

    pygame.draw.rect(surface, PANEL_BG, (x, y, w, panel_h), border_radius=8)

    title = TITLE_FONT.render("MAP LEGEND", True, WHITE)
    surface.blit(title, (x + 10, y + 8))

    font = SMALL_FONT
    start_y = y + 35

    col1_x = x + 10
    col2_x = x + w // 2 + 10

    # Landmarks
    LANDMARK_INFO = [
        ("A", "Wall Corner",   (0, 0)),
        ("B", "Opposite Wall", (10, 0)),
        ("C", "Pillar",        (4, 3)),
        ("D", "Table Corner",  (8, 5)),
        ("E", "Cabinet Edge",  (2, 7)),
    ]

    y_lm = start_y
    header = font.render("Landmarks", True, (200, 200, 200))
    surface.blit(header, (col1_x, y_lm))
    y_lm += 18

    for label, desc, (lx, ly) in LANDMARK_INFO:
        color = LANDMARK_COLORS[label]
        pygame.draw.circle(surface, color, (col1_x + 10, y_lm + 8), 6)
        pygame.draw.circle(surface, WHITE, (col1_x + 10, y_lm + 8), 6, 1)
        text = font.render(f"{label}: ({lx}, {ly})", True, WHITE)
        surface.blit(text, (col1_x + 22, y_lm + 2))
        y_lm += 18

    # Robot types
    y_r = start_y
    header = font.render("Robot Types", True, (200, 200, 200))
    surface.blit(header, (col2_x, y_r))
    y_r += 18

    robot_types = [
        ("True", TRUE_BLUE),
        ("EKF", EST_ORANGE),
        ("Odom", ODOM_GREEN)
    ]

    for label, color in robot_types:
        pygame.draw.circle(surface, color, (col2_x + 10, y_r + 8), 6)
        text = font.render(label, True, WHITE)
        surface.blit(text, (col2_x + 22, y_r + 2))
        y_r += 18


def draw_control_panel(surface, mode, buttons):
    """Draw control panel"""
    x, y = STATS_X, CONTROL_Y
    w = STATS_W
    h = CONTROL_H

    pygame.draw.rect(surface, PANEL_BG, (x, y, w, h), border_radius=8)

    title = TITLE_FONT.render("CONTROL PANEL", True, WHITE)
    surface.blit(title, (x + 10, y + 8))

    mode_y = y + 35
    mode_text = f"Current Mode: {mode}"
    mode_color = BUTTON_ACTIVE if mode == "AUTONOMOUS" else (200, 200, 200)
    surface.blit(FONT.render(mode_text, True, mode_color), (x + 15, mode_y))

    btn_top = y + 70
    btn_h   = 36
    gap     = 10
    btn_w   = (w - 30) // 2

    buttons[0].rect.topleft = (x + 10, btn_top)
    buttons[0].rect.size    = (btn_w, btn_h)

    buttons[1].rect.topleft = (x + 20 + btn_w, btn_top)
    buttons[1].rect.size    = (btn_w, btn_h)

    buttons[2].rect.topleft = (x + 10, btn_top + btn_h + gap)
    buttons[2].rect.size    = (w - 20, btn_h)

    for btn in buttons:
        btn.draw(surface)

    instr_y = y + h - 35
    instructions = [
        "Arrow Keys: Manual control (↑↓←→)",
        "AUTO: Controller navigates to goals"
    ]

    for i, instr in enumerate(instructions):
        surface.blit(SMALL_FONT.render(instr, True, (180, 180, 180)),
                    (x + 15, instr_y + i * 16))


def draw_status_panel(surface, slam, true_pose, odom_pose, controller, pose_error_history):
    """Enhanced status panel with EKF vs Odometry comparison (position + heading)"""
    x, y = STATS_X, STATUS_Y
    w = STATS_W
    
    panel_h = STATUS_H
    pygame.draw.rect(surface, PANEL_BG, (x, y, w, panel_h), border_radius=8)
    
    title = TITLE_FONT.render("STATUS", True, WHITE)
    surface.blit(title, (x + 10, y + 8))
    
    # --- Extract states ---
    ex, ey, eth = slam.get_state()[:3]
    tx, ty, tth = true_pose
    ox, oy, oth = odom_pose
    
    # --- Convert to meters ---
    ex_m, ey_m = pixels_to_meters(ex, ey)
    tx_m, ty_m = pixels_to_meters(tx, ty)
    ox_m, oy_m = pixels_to_meters(ox, oy)
    
    # --- Position errors (meters) ---
    ekf_error = math.sqrt((ex - tx)**2 + (ey - ty)**2) / 100.0
    odom_error = math.sqrt((ox - tx)**2 + (oy - ty)**2) / 100.0
    
    # --- Angle errors (wrapped, degrees) ---
    ekf_ang_err = abs((eth - tth + math.pi) % (2 * math.pi) - math.pi)
    odom_ang_err = abs((oth - tth + math.pi) % (2 * math.pi) - math.pi)
    
    ekf_ang_err_deg = math.degrees(ekf_ang_err)
    odom_ang_err_deg = math.degrees(odom_ang_err)
    
    stats = slam.get_statistics()
    
    info_y = y + 35
    lines = [
        f"Navigation: {controller.get_status_string()}",
        f"Progress: {controller.get_progress():.1f}%",
        f"True Pose : ({tx_m:.2f}m, {ty_m:.2f}m, {math.degrees(tth):.0f}°)",
        f"EKF  Est  : ({ex_m:.2f}m, {ey_m:.2f}m, {math.degrees(eth):.0f}°)",
        f"Odom Est  : ({ox_m:.2f}m, {oy_m:.2f}m, {math.degrees(oth):.0f}°)",
        f"Pos Error : EKF={ekf_error:.3f}m | Odom={odom_error:.3f}m",
        f"Ang Error : EKF={ekf_ang_err_deg:.1f}° | Odom={odom_ang_err_deg:.1f}°",
        "",
        f"Landmarks: {stats['landmarks_mapped']}/5 mapped",
        f"Trusted  : {stats['trusted_landmarks']}/5",
        f"Controller: v={controller.last_v:.2f}, ω={controller.last_w:.2f}"
    ]
    
    for i, line in enumerate(lines):
        color = WHITE if line else (100, 100, 100)
        text = SMALL_FONT.render(line, True, color)
        surface.blit(text, (x + 15, info_y + i * 15))


def draw_uncertainty_ellipse(surface, mean, cov, color, confidence=0.95, alpha=60):
    """Draw uncertainty ellipses"""
    try:
        cov = np.array(cov, dtype=float)

        if cov.shape != (2, 2):
            return
        if not np.all(np.isfinite(cov)):
            return

        cov = (cov + cov.T) / 2.0

        vals, vecs = np.linalg.eigh(cov)

        if np.any(vals <= 1e-6):
            return

        chi2_val = scipy.stats.chi2.ppf(confidence, df=2)

        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        w = 2 * math.sqrt(chi2_val * vals[0])
        h = 2 * math.sqrt(chi2_val * vals[1])

        w = np.clip(w, 6.0, 120.0)
        h = np.clip(h, 6.0, 120.0)

        angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))

        surf_size = int(max(w, h)) + 10
        surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)

        col = (*color[:3], alpha)

        rect = pygame.Rect((surf_size - w) / 2, (surf_size - h) / 2, w, h)

        pygame.draw.ellipse(surf, col, rect, 2)
        pygame.draw.ellipse(surf, (*color[:3], alpha // 3), rect)

        rotated = pygame.transform.rotate(surf, -angle)

        surface.blit(rotated, (mean[0] - rotated.get_width() / 2,
                              mean[1] - rotated.get_height() / 2))

    except Exception:
        pass


def draw_path_tracking(surface, true_path, est_path, odom_path):
    """Draw all three trajectories"""
    if len(true_path) > 1:
        pts_true = [(int(x), int(y)) for x, y in true_path]
        pygame.draw.lines(surface, TRUE_BLUE, False, pts_true, 3)

    if len(odom_path) > 1:
        pts_odom = [(int(x), int(y)) for x, y in odom_path]
        pygame.draw.lines(surface, ODOM_GREEN, False, pts_odom, 3)

    if len(est_path) > 1:
        pts_est = [(int(x), int(y)) for x, y in est_path]
        pygame.draw.lines(surface, EST_ORANGE, False, pts_est, 3)


def draw_planned_path(surface, path):
    """Draw planned path"""
    if len(path) < 2:
        return

    DASH_LENGTH = 14
    GAP_LENGTH = 8
    COLOR = (255, 215, 80)

    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)

        if dist < 1e-3:
            continue

        ux = dx / dist
        uy = dy / dist

        drawn = 0.0
        while drawn < dist:
            start_x = x1 + ux * drawn
            start_y = y1 + uy * drawn

            end_len = min(DASH_LENGTH, dist - drawn)
            end_x = start_x + ux * end_len
            end_y = start_y + uy * end_len

            pygame.draw.line(surface, COLOR, (int(start_x), int(start_y)),
                           (int(end_x), int(end_y)), 4)

            drawn += DASH_LENGTH + GAP_LENGTH

    for x, y in path:
        pygame.draw.circle(surface, COLOR, (int(x), int(y)), 7)
        pygame.draw.circle(surface, (40, 40, 40), (int(x), int(y)), 7, 2)


def draw_environment_panel(env, true_pose, slam, odom_pose, obs, goals, controller, planned_path):
    """Draw main environment with all robots"""
    pygame.draw.rect(WIN, WHITE, (0, ENV_Y_OFFSET, ENV_W, SCREEN_H - ENV_Y_OFFSET))

    env_surface = pygame.Surface((ENV_W, SCREEN_H - ENV_Y_OFFSET), pygame.SRCALPHA)
    env.draw(env_surface)

    if planned_path:
        draw_planned_path(env_surface, planned_path)

    draw_path_tracking(env_surface, true_path, est_path, odom_path)

    # Goals
    for i, (gx, gy) in enumerate(goals):
        gx_i, gy_i = int(gx), int(gy)

        if i == controller.current_goal_idx:
            pulse = 1.0 + 0.1 * math.sin(pygame.time.get_ticks() / 200)
            r = int(18 * pulse)
            pygame.draw.circle(env_surface, GOAL_COLOR, (gx_i, gy_i), r, 4)
            pygame.draw.circle(env_surface, (200, 255, 200), (gx_i, gy_i), 12)
        elif i < controller.current_goal_idx:
            pygame.draw.circle(env_surface, (150, 150, 150), (gx_i, gy_i), 12)
        else:
            pygame.draw.circle(env_surface, (200, 200, 200), (gx_i, gy_i), 10, 2)

        label = FONT.render(f"G{i+1}", True, (50, 50, 50))
        env_surface.blit(label, (gx_i + 22, gy_i - 8))

    # FOV and measurements
    draw_fov_cone(env_surface, true_pose, env.fov, env.max_range, env, FOV_COLOR)
    draw_measurement_lines(env_surface, true_pose, obs, env.landmarks, MEASURE_COLOR)

    # EKF landmarks and uncertainty
    state, P = slam.get_state(), slam.get_covariance()
    ex, ey, eth = state[:3]

    est_landmarks = slam.get_landmarks()
    for i, (lx, ly) in enumerate(est_landmarks):
        cov = P[3+2*i:3+2*i+2, 3+2*i:3+2*i+2]
        draw_uncertainty_ellipse(env_surface, (lx, ly), cov, (150, 50, 200), 0.90, 35)

    draw_uncertainty_ellipse(env_surface, (ex, ey), P[0:2, 0:2], EST_ORANGE, 0.90, 40)
    
    # Draw all three robots
    draw_robot(env_surface, [ex, ey, eth], EST_ORANGE, ROBOT_RADIUS)
    draw_robot(env_surface, odom_pose, ODOM_GREEN, ROBOT_RADIUS)
    draw_robot(env_surface, true_pose, TRUE_BLUE, ROBOT_RADIUS)

    WIN.blit(env_surface, (0, ENV_Y_OFFSET))


def plot_results(sim_data, true_landmarks):
    """Generate comprehensive analysis plots"""
    if len(sim_data.timestamps) == 0:
        print("No data collected for plotting")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # Convert to meters for plotting
    true_pos = np.array(sim_data.true_positions) / 100
    ekf_pos = np.array(sim_data.ekf_positions) / 100
    odom_pos = np.array(sim_data.odom_positions) / 100
    times = np.array(sim_data.timestamps)
    ekf_err = np.array(sim_data.ekf_errors) / 100
    odom_err = np.array(sim_data.odom_errors) / 100
    
    # 1. Trajectory Comparison
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(true_pos[:, 0], true_pos[:, 1], 'b-', linewidth=2, label='Ground Truth')
    ax1.plot(ekf_pos[:, 0], ekf_pos[:, 1], 'orange', linewidth=2, label='EKF-SLAM', alpha=0.8)
    ax1.plot(odom_pos[:, 0], odom_pos[:, 1], 'g--', linewidth=2, label='Odometry', alpha=0.7)
    
    # Plot landmarks
    true_lm_m = np.array([pixels_to_meters(lm[0], lm[1]) for lm in true_landmarks])
    ax1.scatter(true_lm_m[:, 0], true_lm_m[:, 1], c='red', s=100, marker='*', 
               label='Landmarks', zorder=5)
    
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    ax1.set_title('Trajectory Comparison', fontweight='bold', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # 2. Position Error Over Time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(times, ekf_err, 'orange', linewidth=2, label='EKF-SLAM Error')
    ax2.plot(times, odom_err, 'g--', linewidth=2, label='Odometry Error')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Position Error (meters)')
    ax2.set_title('Localization Error Over Time', fontweight='bold', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # 3. Control Signals
    ax3 = plt.subplot(2, 3, 3)
    ax3_twin = ax3.twinx()
    
    l1 = ax3.plot(times, sim_data.control_v, 'b-', linewidth=2, label='Linear Velocity (v)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Linear Velocity (pixels/s)', color='b')
    ax3.tick_params(axis='y', labelcolor='b')
    
    l2 = ax3_twin.plot(times, sim_data.control_w, 'r-', linewidth=2, label='Angular Velocity (ω)')
    ax3_twin.set_ylabel('Angular Velocity (rad/s)', color='r')
    ax3_twin.tick_params(axis='y', labelcolor='r')
    
    ax3.set_title('Control Signals Over Time', fontweight='bold', fontsize=12)
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='best')
    ax3.grid(True, alpha=0.3)
    
    # 4. Landmark Estimation Error
    ax4 = plt.subplot(2, 3, 4)
    if sim_data.landmark_errors:
        lm_times = times[:len(sim_data.landmark_errors)]
        ax4.plot(lm_times, np.array(sim_data.landmark_errors) / 100, 
                'purple', linewidth=2)
        ax4.set_xlabel('Time (seconds)')
        ax4.set_ylabel('Mean Landmark Error (meters)')
        ax4.set_title('Landmark Estimation Error', fontweight='bold', fontsize=12)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No landmark data', ha='center', va='center')
        ax4.set_title('Landmark Estimation Error', fontweight='bold', fontsize=12)
    
    # 5. Cumulative Error Comparison
    ax5 = plt.subplot(2, 3, 5)
    ekf_cumulative = np.cumsum(ekf_err)
    odom_cumulative = np.cumsum(odom_err)
    
    ax5.plot(times, ekf_cumulative, 'orange', linewidth=2, label='EKF-SLAM')
    ax5.plot(times, odom_cumulative, 'g--', linewidth=2, label='Odometry')
    ax5.set_xlabel('Time (seconds)')
    ax5.set_ylabel('Cumulative Error (meters)')
    ax5.set_title('Cumulative Localization Error', fontweight='bold', fontsize=12)
    ax5.legend(loc='best')
    ax5.grid(True, alpha=0.3)
    
    # 6. EKF Uncertainty (Covariance Trace)
    ax6 = plt.subplot(2, 3, 6)
    if sim_data.ekf_uncertainties:
        unc_times = times[:len(sim_data.ekf_uncertainties)]
        ax6.plot(unc_times, sim_data.ekf_uncertainties, 
                'darkblue', linewidth=2)
        ax6.set_xlabel('Time (seconds)')
        ax6.set_ylabel('Position Uncertainty (pixels)')
        ax6.set_title('EKF Position Uncertainty', fontweight='bold', fontsize=12)
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ekf_slam_results.png', dpi=300, bbox_inches='tight')
    print("\n✓ Results saved to 'ekf_slam_results.png'")
    
    # Print statistics
    print("\n" + "="*70)
    print("SIMULATION RESULTS SUMMARY")
    print("="*70)
    print(f"Simulation Duration: {times[-1]:.2f} seconds")
    print(f"\nEKF-SLAM Performance:")
    print(f"  Final Error: {ekf_err[-1]:.3f}m")
    print(f"  Mean Error: {np.mean(ekf_err):.3f}m")
    print(f"  Max Error: {np.max(ekf_err):.3f}m")
    print(f"  Std Dev: {np.std(ekf_err):.3f}m")
    
    print(f"\nOdometry-Only Performance:")
    print(f"  Final Error: {odom_err[-1]:.3f}m")
    print(f"  Mean Error: {np.mean(odom_err):.3f}m")
    print(f"  Max Error: {np.max(odom_err):.3f}m")
    print(f"  Std Dev: {np.std(odom_err):.3f}m")
    
    improvement = ((np.mean(odom_err) - np.mean(ekf_err)) / np.mean(odom_err)) * 100
    print(f"\nEKF-SLAM Improvement: {improvement:.1f}% better than odometry")
    
    if sim_data.landmark_errors:
        print(f"\nLandmark Estimation:")
        print(f"  Final Mean Error: {sim_data.landmark_errors[-1]/100:.3f}m")
        print(f"  Average Error: {np.mean(sim_data.landmark_errors)/100:.3f}m")
    
    print("="*70 + "\n")
    
    plt.show()


def main():
    """Main loop with odometry comparison"""
    global est_path, true_path, odom_path, sim_data

    env = Environment(map_data)
    
    true_pose = np.array([float(env.robot_start[0]), 
                         float(env.robot_start[1]), 
                         np.random.uniform(-np.pi, np.pi)])
    
    # Initialize odometry pose
    odom_pose = true_pose.copy()
    
    slam = EKFSLAM(
        init_pose=true_pose,
        init_cov=np.diag([2.0**2, 2.0**2, np.deg2rad(10.0)**2]),
        sigma_r=3.0,
        sigma_phi=np.deg2rad(2.0),
        motion_noise=np.diag([0.08**2, 0.08**2, np.deg2rad(1.5)**2]),
    )
    
    controller = WaypointTrackingController(
        goals=GOALS_PIXELS,
        Kd=0.8,
        Ktheta=3.5,
        goal_tolerance=35.0,
        max_linear_vel=15.0,
        look_ahead_dist=80.0
    )
    
    path_planner = PathPlanner(env)
    
    mode = "MANUAL"
    v, w = 0.0, 0.0
    pose_error_history = []
    planned_path = []
    
    buttons = [
        Button(STATS_X + 10, 0, 130, 35, "AUTO", BUTTON_ACTIVE),
        Button(STATS_X + 150, 0, 130, 35, "MANUAL", BUTTON_BG),
        Button(STATS_X + 10, 0, 270, 35, "RESET", BUTTON_BG),
    ]
    
    run = True
    frame = 0
    replan_counter = 0
    sim_time = 0.0
    
    # Get true landmarks
    true_landmarks = [(lm[0], lm[1]) for lm in env.landmarks]
    
    while run:
        clock.tick(FPS)
        frame += 1
        sim_time += DT
        mouse_pos = pygame.mouse.get_pos()
        
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                run = False
                # Plot results before quitting
                plot_results(sim_data, true_landmarks)
            
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    run = False
                    plot_results(sim_data, true_landmarks)
            
            if buttons[0].handle_event(e, mouse_pos):
                mode = "AUTONOMOUS"
                planned_path = []
                print("AUTONOMOUS mode - Controller active")
            
            if buttons[1].handle_event(e, mouse_pos):
                mode = "MANUAL"
                v, w = 0.0, 0.0
                planned_path = []
                print("MANUAL mode")
            
            if buttons[2].handle_event(e, mouse_pos):
                start_xy = env.reset_robot_start()
                true_pose = np.array([float(start_xy[0]), float(start_xy[1]),
                                     np.random.uniform(-np.pi, np.pi)])
                odom_pose = true_pose.copy()
                
                slam = EKFSLAM(
                    init_pose=true_pose,
                    init_cov=np.diag([2.0**2, 2.0**2, np.deg2rad(10.0)**2]),
                    sigma_r=3.0, sigma_phi=np.deg2rad(2.0),
                    motion_noise=np.diag([0.08**2, 0.08**2, np.deg2rad(1.5)**2])
                )
                
                controller.reset()
                path_planner = PathPlanner(env)
                
                pose_error_history.clear()
                planned_path.clear()
                est_path.clear()
                true_path.clear()
                odom_path.clear()
                sim_data = SimulationData()
                
                mode = "MANUAL"
                v, w = 0.0, 0.0
                sim_time = 0.0
                print("System RESET → Randomized start")
        
        # Control logic
        if mode == "MANUAL":
            keys = pygame.key.get_pressed()
            v_cmd = (keys[pygame.K_UP] - keys[pygame.K_DOWN]) * 10.0
            w_cmd = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * 0.5
            v = 0.7 * v + 0.3 * v_cmd
            w = 0.7 * w + 0.3 * w_cmd

            # ========================================================
            # ADDED: Manually check if goal is reached in MANUAL mode
            # ========================================================
            est_pose_check = slam.get_robot_pose() # Use current estimate
            current_goal = controller.get_current_goal()
            
            if current_goal:
                dist = math.hypot(current_goal[0] - est_pose_check[0], 
                                  current_goal[1] - est_pose_check[1])
                
                if dist < controller.goal_tolerance:
                    print(f"✓ Goal G{controller.current_goal_idx + 1} reached (MANUAL)")
                    controller.current_goal_idx += 1
                    
                    if controller.finished():
                        print("\nMISSION COMPLETE (MANUAL)!\n")
        
        elif mode == "AUTONOMOUS":
            est_pose = slam.get_robot_pose()
            P = slam.get_covariance()
            
            nearby_obstacles = env.get_nearby_obstacles(true_pose, search_radius=150.0)
            
            current_goal = controller.get_current_goal()
            if current_goal:
                replan_counter += 1
                
                if replan_counter >= 15 or not planned_path:
                    replan_counter = 0
                    
                    if path_planner.is_path_blocked(est_pose[:2], current_goal, nearby_obstacles):
                        intermediate = path_planner.find_clear_waypoint(
                            est_pose[:2], current_goal, nearby_obstacles
                        )
                        
                        if intermediate:
                            planned_path = [est_pose[:2], intermediate, current_goal]
                            controller.set_intermediate_waypoint(intermediate)
                        else:
                            planned_path = [est_pose[:2], current_goal]
                    else:
                        planned_path = [est_pose[:2], current_goal]
                        controller.clear_intermediate_waypoint()
            
            v, w = controller.compute_control_smooth(
                est_pose=est_pose,
                covariance=P[0:3, 0:3],
                nearby_obstacles=nearby_obstacles,
                dt=DT
            )
            
            if controller.finished():
                print("\nMISSION COMPLETE!\n")
                mode = "MANUAL"
                v, w = 0.0, 0.0
                planned_path = []
        
        # Motion update for true robot
        nx = true_pose[0] + v * math.cos(true_pose[2]) * DT
        ny = true_pose[1] + v * math.sin(true_pose[2]) * DT
        ntheta = wrap_to_pi(true_pose[2] + w * DT)
        
        if not env.check_collision((nx, ny), ROBOT_RADIUS):
            true_pose = np.array([nx, ny, ntheta])
        else:
            v, w = 0.0, 0.0
            if mode == "AUTONOMOUS":
                replan_counter = 0
        
        # Odometry update (dead reckoning with noise accumulation)
        odom_noise_v = v + np.random.normal(0, 0.1 * abs(v) + 0.05)
        odom_noise_w = w + np.random.normal(0, 0.05 * abs(w) + 0.02)
        
        odom_pose[0] += odom_noise_v * math.cos(odom_pose[2]) * DT
        odom_pose[1] += odom_noise_v * math.sin(odom_pose[2]) * DT
        odom_pose[2] = wrap_to_pi(odom_pose[2] + odom_noise_w * DT)
        
        # EKF-SLAM updates
        slam.predict(control=(v, w), dt=DT)
        obs = env.get_sensor_observations(true_pose, add_noise=True)
        if obs:
            slam.update(obs)
        
        # Track paths
        est_pose = slam.get_robot_pose()
        est_path.append((est_pose[0], est_pose[1]))
        true_path.append((true_pose[0], true_pose[1]))
        odom_path.append((odom_pose[0], odom_pose[1]))
        
        if len(est_path) > 100000:
            est_path.pop(0)
        if len(true_path) > 100000:
            true_path.pop(0)
        if len(odom_path) > 100000:
            odom_path.pop(0)
        
        # Record data for plotting
        sim_data.record(sim_time, true_pose, est_pose, odom_pose, v, w, slam, true_landmarks)
        
        pose_error = np.linalg.norm(est_pose[:2] - true_pose[:2])
        pose_error_history.append(pose_error)
        
        # Drawing
        WIN.fill((40, 45, 52))
        draw_environment_header(WIN)
        draw_environment_panel(env, true_pose, slam, odom_pose, obs, 
                               GOALS_PIXELS, controller, planned_path)
        draw_mini_map(WIN, slam, controller)
        draw_legend(WIN)
        draw_control_panel(WIN, mode, buttons)
        draw_status_panel(WIN, slam, true_pose, odom_pose, controller, pose_error_history)
        
        pygame.display.flip()
    
    pygame.quit()
    
    # Final plots
    if len(sim_data.timestamps) > 0:
        plot_results(sim_data, true_landmarks)


if __name__ == "__main__":
    main()