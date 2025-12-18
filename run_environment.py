import pygame
import math
import numpy as np
from environment import Environment

# ---------------------------
# World display settings
# ---------------------------
WORLD_SCALE = 80  # pixels per meter (adjust zoom)
MARGIN = 50       # pixels for window border

ROOM_COLOR = (245, 240, 230)
TEXT_COLOR = (20, 20, 20)
ROBOT_COLOR = (255, 50, 50)
GRID_COLOR = (220, 220, 220)

# ---------------------------
# Helper functions
# ---------------------------
def world_to_screen(pos, world_size, screen_size):
    """Convert world coordinates (meters) to screen pixels."""
    wx, wy = pos
    sx = int(wx * WORLD_SCALE + MARGIN)
    sy = int(screen_size[1] - (wy * WORLD_SCALE + MARGIN))
    return (sx, sy)

def draw_grid(screen, env, screen_size):
    """Draw grid lines for the room floor."""
    cell_size_px = WORLD_SCALE
    for x in range(MARGIN, int(env.world_size[0]*WORLD_SCALE + MARGIN + 1), cell_size_px):
        pygame.draw.line(screen, GRID_COLOR, (x, MARGIN),
                         (x, int(env.world_size[1]*WORLD_SCALE + MARGIN)))
    for y in range(MARGIN, int(env.world_size[1]*WORLD_SCALE + MARGIN + 1), cell_size_px):
        pygame.draw.line(screen, GRID_COLOR, (MARGIN, y),
                         (int(env.world_size[0]*WORLD_SCALE + MARGIN), y))

def draw_robot(screen, robot_state, env, screen_size):
    """Draw robot as a triangle with FOV cone."""
    x, y, theta = robot_state
    pos = world_to_screen((x, y), env.world_size, screen_size)

    # Robot triangle
    size_px = 10
    pts = []
    for angle_offset in [0, 140, -140]:
        a = math.radians(angle_offset) + theta
        px = pos[0] + size_px * math.cos(a)
        py = pos[1] - size_px * math.sin(a)
        pts.append((px, py))
    pygame.draw.polygon(screen, ROBOT_COLOR, pts)

    # FOV cone
    fov = env.fov
    cone_length = env.max_range * WORLD_SCALE
    left_angle = theta + fov / 2
    right_angle = theta - fov / 2
    left = (pos[0] + cone_length * math.cos(left_angle),
            pos[1] - cone_length * math.sin(left_angle))
    right = (pos[0] + cone_length * math.cos(right_angle),
             pos[1] - cone_length * math.sin(right_angle))
    pygame.draw.polygon(screen, (255, 150, 150, 60), [pos, left, right], width=1)

# ---------------------------
# Main visualization
# ---------------------------
def visualize_environment():
    pygame.init()

    env = Environment(world_size=(10.0, 8.0), camera_only=False)
    env.define_living_room()

    screen_size = (int(env.world_size[0]*WORLD_SCALE + 2*MARGIN),
                   int(env.world_size[1]*WORLD_SCALE + 2*MARGIN))
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("🏠 Living Room Environment (Icons)")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 16)
    robot_state = np.array([5.0, 3.0, math.radians(45)])

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Draw background
        screen.fill(ROOM_COLOR)
        draw_grid(screen, env, screen_size)

        # Room border
        rect = pygame.Rect(MARGIN, MARGIN,
                           env.world_size[0]*WORLD_SCALE,
                           env.world_size[1]*WORLD_SCALE)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2)

        # Draw landmarks (icons)
        for lm in env.landmarks:
            pos = world_to_screen(lm['pos'], env.world_size, screen_size)
            if lm['icon'] is not None:
                screen.blit(lm['icon'], (pos[0]-25, pos[1]-25))
            else:
                pygame.draw.circle(screen, (30, 144, 255), pos, 6)
            text = font.render(lm['name'], True, TEXT_COLOR)
            screen.blit(text, (pos[0]+30, pos[1]-20))

        # Draw robot
        draw_robot(screen, robot_state, env, screen_size)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

# ---------------------------
# Run visualization
# ---------------------------
if __name__ == "__main__":
    visualize_environment()
