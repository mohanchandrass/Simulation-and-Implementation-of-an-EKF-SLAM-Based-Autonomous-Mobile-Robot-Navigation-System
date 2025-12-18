"""
environment.py - Enhanced Visual Environment with Dynamic Obstacles
Beautiful landmark icons, real-time obstacle placement, modern aesthetics
"""
import pygame
import math
import numpy as np

# Robot parameters
ROBOT_RADIUS = 12

# SCALE: 100 pixels = 1 meter
PIXELS_PER_METER = 80

# Room dimensions (10m × 8m)
ROOM_WIDTH_M = 10
ROOM_HEIGHT_M = 8
ROOM_WIDTH_PX = ROOM_WIDTH_M * PIXELS_PER_METER  # 1000px
ROOM_HEIGHT_PX = ROOM_HEIGHT_M * PIXELS_PER_METER  # 800px

# Margin for walls
MARGIN = 40

# Enhanced colors
LANDMARK_COLORS = {
    'A': (255, 75, 75),    # Red - Wall Corner
    'B': (75, 150, 255),   # Blue - Opposite Wall
    'C': (255, 180, 50),   # Orange - Pillar
    'D': (150, 75, 255),   # Purple - Table
    'E': (75, 255, 150)    # Green - Cabinet
}

def meters_to_pixels(x_m, y_m):
    """Convert from meters to pixels. Y-axis is inverted (screen coords)"""
    x_px = MARGIN + x_m * PIXELS_PER_METER
    y_px = ROOM_HEIGHT_PX + MARGIN - y_m * PIXELS_PER_METER
    return x_px, y_px

def pixels_to_meters(x_px, y_px):
    """Convert from pixels to meters"""
    x_m = (x_px - MARGIN) / PIXELS_PER_METER
    y_m = (ROOM_HEIGHT_PX + MARGIN - y_px) / PIXELS_PER_METER
    return x_m, y_m

# LANDMARKS with icons
WALL_LANDMARK_EPS = 0.15  # meters (inside wall)

LANDMARKS_METERS = [
    # Wall corners moved INSIDE the room
    (0 + WALL_LANDMARK_EPS, 0 + WALL_LANDMARK_EPS, "A", "Wall Corner", "🔴"),
    (10 - WALL_LANDMARK_EPS, 0 + WALL_LANDMARK_EPS, "B", "Opposite Wall", "🔵"),

    # Internal landmarks unchanged
    (4, 3, "C", "Pillar", "🟠"),
    (8, 5, "D", "Table Corner", "🟣"),
    (2, 7, "E", "Cabinet Edge", "🟢")
]


LANDMARKS_PIXELS = [meters_to_pixels(x, y) for x, y, _, _, _ in LANDMARKS_METERS]

# GOALS
GOALS_METERS = [(2, 2), (8, 4), (5, 7)]
GOALS_PIXELS = [meters_to_pixels(x, y) for x, y in GOALS_METERS]
# Update the PHYSICAL_OBJECTS with correct alignment and circular pillar
PHYSICAL_OBJECTS = [
    # Cabinet (1.5m wide × 0.8m deep at position (2, 7) - landmark E at right edge)
    {
        'type': 'cabinet',
        'pos_m': (0.5, 6.3),  # Cabinet starts at (0.5, 6.2) so right edge is at (2, 7)
        'width_m': 1.5,
        'height_m': 0.8,
        'color': (180, 200, 220),  # Professional light blue-gray
        'border_color': (100, 130, 160),
        'landmark_idx': 4,  # Index of landmark E
        'landmark_pos': 'right_edge'
    },
    # Pillar (circular, 0.4m radius at position (4, 3) - landmark C at center)
    {
        'type': 'pillar',
        'center_m': (4, 3),  # Center is at (4, 3) - landmark C
        'radius_m': 0.4,
        'color': (220, 220, 220),  # Light gray
        'border_color': (160, 160, 160),
        'landmark_idx': 2,  # Index of landmark C
        'landmark_pos': 'center'
    },
    # Table (rectangular, 1.0m × 0.6m at position (8, 5) - landmark D at front-left corner)
    {
        'type': 'table',
        'pos_m': (8, 5),  # Position of front-left corner (landmark D)
        'width_m': 1.0,
        'height_m': 0.6,
        'color': (210, 195, 180),  # Warm gray/beige
        'border_color': (170, 155, 140),
        'landmark_idx': 3,  # Index of landmark D
        'landmark_pos': 'corner'
    }
]


# Update map_data to include physical objects
# Update map_data to use new wall dimensions
map_data = {
    "walls": [
        [MARGIN - 15, MARGIN - 15, 15, ROOM_HEIGHT_PX + 30],  # Left wall
        [MARGIN - 15, MARGIN - 15, ROOM_WIDTH_PX + 30, 15],   # Top wall
        [MARGIN + ROOM_WIDTH_PX, MARGIN - 15, 15, ROOM_HEIGHT_PX + 30],  # Right wall
        [MARGIN - 15, MARGIN + ROOM_HEIGHT_PX, ROOM_WIDTH_PX + 30, 15],  # Bottom wall
    ],
    "dynamic_obstacles": [],  # User-placed obstacles
    "landmarks": LANDMARKS_PIXELS,
    "robot_start": None,
    "physical_objects": PHYSICAL_OBJECTS  # Add physical objects
}



class Environment:
    """Enhanced Environment with dynamic obstacle support"""
    
    def __init__(self, map_dict):
        # Parse map data
        self.walls = [pygame.Rect(x, y, w, h) for x, y, w, h in map_dict["walls"]]
        self.landmarks = map_dict["landmarks"]
        self.robot_start = map_dict.get("robot_start")
        self.physical_objects = map_dict.get("physical_objects", [])
        
        # Convert physical objects to pixel coordinates
        self.physical_objects_px = []
        for obj in self.physical_objects:
            if obj['type'] == 'pillar':
                cx_px, cy_px = meters_to_pixels(obj['center_m'][0], obj['center_m'][1])
                radius_px = obj['radius_m'] * PIXELS_PER_METER
                obj_px = {
                    'type': obj['type'],
                    'center': (cx_px, cy_px),
                    'radius': radius_px,
                    'color': obj['color'],
                    'border_color': obj.get('border_color', obj['color']),
                    'landmark_idx': obj['landmark_idx'],
                    'landmark_pos': obj['landmark_pos']
                }
                self.physical_objects_px.append(obj_px)
            else:
                x_px, y_px = meters_to_pixels(obj['pos_m'][0], obj['pos_m'][1])
                width_px = obj['width_m'] * PIXELS_PER_METER
                height_px = obj['height_m'] * PIXELS_PER_METER
                obj_px = {
                    'type': obj['type'],
                    'rect': pygame.Rect(x_px, y_px, width_px, height_px),
                    'color': obj['color'],
                    'border_color': obj.get('border_color', obj['color']),
                    'landmark_idx': obj['landmark_idx'],
                    'landmark_pos': obj['landmark_pos']
                }
                self.physical_objects_px.append(obj_px)
        
        self.dynamic_obstacles = []
        self.obstacle_hover = None
        
        if self.robot_start is None:
            self.robot_start = self._generate_random_start()
        
        self.max_range = 400.0
        self.fov = math.radians(90)
        self.sigma_r = 4.0
        self.sigma_phi = math.radians(3.0)

    def _generate_random_start(self):
        safe_x_min = MARGIN + 80
        safe_x_max = MARGIN + ROOM_WIDTH_PX - 80
        safe_y_min = MARGIN + 80
        safe_y_max = MARGIN + ROOM_HEIGHT_PX - 80
        
        for _ in range(100):
            x = np.random.uniform(safe_x_min, safe_x_max)
            y = np.random.uniform(safe_y_min, safe_y_max)
            if not self.check_collision((x, y), ROBOT_RADIUS + 10):
                return [x, y]
        return [MARGIN + ROOM_WIDTH_PX / 2, MARGIN + ROOM_HEIGHT_PX / 2]

    def reset_robot_start(self):
        """
        Public wrapper to reset robot start position.
        """
        self.robot_start = self._generate_random_start()
        return self.robot_start


    def draw(self, surface):
        """Draw enhanced environment with beautiful visuals"""

        room_rect = pygame.Rect(MARGIN, MARGIN, ROOM_WIDTH_PX, ROOM_HEIGHT_PX)
        pygame.draw.rect(surface, (245, 245, 245), room_rect)

        grid_color = (235, 235, 235)
        for i in range(1, 10):
            x = MARGIN + i * (ROOM_WIDTH_PX // 10)
            pygame.draw.line(surface, grid_color, (x, MARGIN), (x, MARGIN + ROOM_HEIGHT_PX), 1)
        for i in range(1, 8):
            y = MARGIN + i * (ROOM_HEIGHT_PX // 8)
            pygame.draw.line(surface, grid_color, (MARGIN, y), (MARGIN + ROOM_WIDTH_PX, y), 1)

        pygame.draw.rect(surface, (200, 200, 200), room_rect, 2)

        font_obj = pygame.font.SysFont("Arial", 14, bold=True)
        font_small = pygame.font.SysFont("Arial", 11)

        for obj in self.physical_objects_px:
            if obj['type'] == 'pillar':
                cx, cy = obj['center']
                r = obj['radius']

                shadow = pygame.Surface((r * 2 + 6, r * 2 + 6), pygame.SRCALPHA)
                pygame.draw.circle(shadow, (180, 180, 180, 100), (r + 3, r + 3), r + 3)
                surface.blit(shadow, (int(cx - r - 3), int(cy - r - 3)))

                pygame.draw.circle(surface, obj['color'], (int(cx), int(cy)), int(r))
                pygame.draw.circle(surface, obj['border_color'], (int(cx), int(cy)), int(r), 2)

                label_text = font_obj.render("PILLAR", True, (100, 100, 100))
                type_text = font_small.render("Structural", True, (120, 120, 120))
                LABEL_OFFSET_Y = 0
                center_x = cx                
                center_y = cy - LABEL_OFFSET_Y 
            else:
                rect = obj['rect']

                shadow = rect.copy()
                shadow.x += 3
                shadow.y += 3
                pygame.draw.rect(surface, (180, 180, 180, 100), shadow)

                pygame.draw.rect(surface, obj['color'], rect)
                pygame.draw.rect(surface, obj['border_color'], rect, 2)

                if obj['type'] == 'cabinet':

                    lm_x, lm_y = self.landmarks[obj['landmark_idx']]
                    rect.right = lm_x   
                    rect.top = lm_y 
                    
                    door_w = rect.width // 2
                    pygame.draw.rect(surface, (160, 180, 200),
                                     pygame.Rect(rect.left, rect.top, door_w, rect.height))
                    pygame.draw.line(surface, (140, 160, 180),
                                     (rect.left + door_w, rect.top),
                                     (rect.left + door_w, rect.bottom), 2)

                    label_text = font_obj.render("CABINET", True, (100, 120, 140))
                    type_text = font_small.render("Storage", True, (120, 140, 160))
                else:
                    label_text = font_obj.render("TABLE", True, (140, 130, 120))
                    type_text = font_small.render("Workspace", True, (160, 150, 140))

                center_x, center_y = rect.centerx, rect.centery

            text_bg = pygame.Surface(
                (max(label_text.get_width(), type_text.get_width()) + 20, 40),
                pygame.SRCALPHA
            )
            pygame.draw.rect(text_bg, (255, 255, 255, 180),
                             text_bg.get_rect(), border_radius=4)
            text_bg.blit(label_text, (10, 8))
            text_bg.blit(type_text, (10, 24))
            surface.blit(
                text_bg,
                (center_x - text_bg.get_width() // 2,
                 center_y - text_bg.get_height() // 2)
            )

        # ---------------- FIXED LANDMARK DRAW ----------------
        font_large = pygame.font.SysFont("Arial", 18, bold=True)
        font_small = pygame.font.SysFont("Arial", 10)
        LANDMARK_VISUAL_Y_OFFSET = 10  # <<< FIX

        for i, (lx, ly_raw) in enumerate(self.landmarks):
            ly = ly_raw + LANDMARK_VISUAL_Y_OFFSET

            label, desc, _ = LANDMARKS_METERS[i][2], LANDMARKS_METERS[i][3], LANDMARKS_METERS[i][4]
            color = LANDMARK_COLORS[label]

            for j in range(2):
                alpha = 150 - j * 75
                glow = 12 + j * 3
                s = pygame.Surface((glow * 2, glow * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*color, alpha), (glow, glow), glow)
                surface.blit(s, (int(lx - glow), int(ly - glow)))

            pygame.draw.circle(surface, color, (int(lx), int(ly)), 10)
            pygame.draw.circle(surface, (255, 255, 255), (int(lx), int(ly)), 8)
            pygame.draw.circle(surface, color, (int(lx), int(ly)), 8, 2)

            label_text = font_large.render(label, True, color)
            surface.blit(label_text, label_text.get_rect(center=(int(lx), int(ly))))

            desc_text = font_small.render(desc, True, (80, 80, 80))
            desc_bg = pygame.Surface(
                (desc_text.get_width() + 12, desc_text.get_height() + 6),
                pygame.SRCALPHA
            )
            pygame.draw.rect(desc_bg, (255, 255, 255, 230),
                             desc_bg.get_rect(), border_radius=3, width=1)
            desc_bg.blit(desc_text, (6, 3))
            surface.blit(desc_bg, (int(lx) + 15, int(ly) - 10))

    def check_collision(self, position, radius):
        x, y = position
        robot_rect = pygame.Rect(x - radius, y - radius, radius * 2, radius * 2)

        for wall in self.walls:
            if robot_rect.colliderect(wall):
                return True

        for obj in self.physical_objects_px:
            if obj['type'] == 'pillar':
                if math.hypot(x - obj['center'][0], y - obj['center'][1]) < radius + obj['radius']:
                    return True
            else:
                if robot_rect.colliderect(obj['rect']):
                    return True

        for obs in self.dynamic_obstacles:
            if math.hypot(x - obs['x'], y - obs['y']) < radius + obs['radius']:
                return True

        return False

    def line_intersects_rect(self, x1, y1, x2, y2, rect):
        """Check if line segment intersects with rectangle"""
        if rect.collidepoint(x1, y1) or rect.collidepoint(x2, y2):
            return True
        
        left, right, top, bottom = rect.left, rect.right, rect.top, rect.bottom
        
        edges = [
            (left, top, left, bottom),
            (right, top, right, bottom),
            (left, top, right, top),
            (left, bottom, right, bottom)
        ]
        
        for x3, y3, x4, y4 in edges:
            if self._line_segments_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
                return True
        
        return False
    
    def _line_segments_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4):
        """Check if two line segments intersect"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def has_line_of_sight(self, x1, y1, x2, y2):
        """Check line of sight considering walls, physical objects, and dynamic obstacles"""
        # Check walls
        for wall in self.walls:
            if self.line_intersects_rect(x1, y1, x2, y2, wall):
                return False
        
        # Check physical objects - but allow line of sight to landmarks ON the objects
        for obj in self.physical_objects_px:
            # Skip collision check if the line endpoint is exactly at the landmark
            # This allows the robot to "see" landmarks that are on objects
            if obj['type'] == 'pillar':
                # For pillar, check if endpoint is at the center (landmark C)
                if (abs(x2 - obj['center'][0]) < 1 and abs(y2 - obj['center'][1]) < 1):
                    continue  # Skip collision check for landmark C
                if self._line_intersects_circle(x1, y1, x2, y2, 
                                               obj['center'][0], obj['center'][1], 
                                               obj['radius']):
                    return False
            else:
                # For rectangular objects, check if endpoint is at the landmark position
                rect = obj['rect']
                landmark_pos = self.landmarks[obj['landmark_idx']]
                if (abs(x2 - landmark_pos[0]) < 1 and abs(y2 - landmark_pos[1]) < 1):
                    continue  # Skip collision check for landmark on this object
                if self.line_intersects_rect(x1, y1, x2, y2, rect):
                    return False
        
        # Check dynamic obstacles
        for obs in self.dynamic_obstacles:
            if self._line_intersects_circle(x1, y1, x2, y2, obs['x'], obs['y'], obs['radius']):
                return False
        
        return True
    
    def _line_intersects_circle(self, x1, y1, x2, y2, cx, cy, r):
        """Check if line segment intersects circle (excluding the endpoints)"""
        # First check if either endpoint is inside the circle (with tolerance)
        dist1 = math.hypot(x1 - cx, y1 - cy)
        dist2 = math.hypot(x2 - cx, y2 - cy)
        
        # If either endpoint is very close to the center (landmark point), don't count as intersection
        if dist1 < 5 or dist2 < 5:
            return False
        
        # Check if line segment intersects circle (excluding endpoints)
        dx, dy = x2 - x1, y2 - y1
        fx, fy = x1 - cx, y1 - cy
        
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - r*r
        
        discriminant = b*b - 4*a*c
        
        if discriminant < 0:
            return False
        
        discriminant = math.sqrt(discriminant)
        t1 = (-b - discriminant) / (2*a)
        t2 = (-b + discriminant) / (2*a)
        
        # Check if intersection occurs between the endpoints (0 < t < 1)
        return (0 < t1 < 1) or (0 < t2 < 1)
    
    def is_in_fov(self, robot_pose, landmark_pos):
        """Check if landmark is in FOV with line of sight"""
        rx, ry, rtheta = robot_pose
        lx, ly = landmark_pos
        
        dx = lx - rx
        dy = ly - ry
        distance = math.hypot(dx, dy)
        
        if distance > self.max_range:
            return False
        
        angle_to_landmark = math.atan2(dy, dx)
        angle_diff = angle_to_landmark - rtheta
        
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        if abs(angle_diff) > self.fov / 2:
            return False
        
        return self.has_line_of_sight(rx, ry, lx, ly)
    
    def get_sensor_observations(self, robot_pose, add_noise=True):
        """Get sensor observations of visible landmarks"""
        observations = []
        rx, ry, rtheta = robot_pose
        
        for lm_id, (lx, ly) in enumerate(self.landmarks):
            if self.is_in_fov(robot_pose, [lx, ly]):
                dx = lx - rx
                dy = ly - ry
                true_range = math.hypot(dx, dy)
                true_bearing = math.atan2(dy, dx) - rtheta
                
                while true_bearing > math.pi:
                    true_bearing -= 2 * math.pi
                while true_bearing < -math.pi:
                    true_bearing += 2 * math.pi
                
                if add_noise:
                    range_noise = np.random.normal(0, self.sigma_r)
                    bearing_noise = np.random.normal(0, self.sigma_phi)
                    measured_range = true_range + range_noise
                    measured_bearing = true_bearing + bearing_noise
                else:
                    measured_range = true_range
                    measured_bearing = true_bearing
                
                observations.append((measured_range, measured_bearing, lm_id))
        
        return observations
    
    def get_nearby_obstacles(self, robot_pose, search_radius=100.0):
        """Get nearby obstacle positions for avoidance"""
        rx, ry, _ = robot_pose
        obstacles = []
        
        # Wall corners
        for wall in self.walls:
            corners = [
                (wall.left, wall.top),
                (wall.right, wall.top),
                (wall.left, wall.bottom),
                (wall.right, wall.bottom)
            ]
            
            for cx, cy in corners:
                dist = math.hypot(cx - rx, cy - ry)
                if dist < search_radius:
                    obstacles.append((cx, cy))
        
        # Physical objects
        for obj in self.physical_objects_px:
            if obj['type'] == 'pillar':
                # Sample points around pillar perimeter
                center_x, center_y = obj['center']
                radius = obj['radius']
                dist_to_center = math.hypot(center_x - rx, center_y - ry)
                if dist_to_center < search_radius + radius:
                    for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
                        ox = center_x + radius * math.cos(angle)
                        oy = center_y + radius * math.sin(angle)
                        obstacles.append((ox, oy))
            else:
                # Add corners of the object
                rect = obj['rect']
                corners = [
                    (rect.left, rect.top),
                    (rect.right, rect.top),
                    (rect.left, rect.bottom),
                    (rect.right, rect.bottom),
                    (rect.centerx, rect.top),
                    (rect.centerx, rect.bottom),
                    (rect.left, rect.centery),
                    (rect.right, rect.centery)
                ]
                for cx, cy in corners:
                    dist = math.hypot(cx - rx, cy - ry)
                    if dist < search_radius:
                        obstacles.append((cx, cy))
        
        # Dynamic obstacles - sample points around perimeter
        for obs in self.dynamic_obstacles:
            dist_to_center = math.hypot(obs['x'] - rx, obs['y'] - ry)
            if dist_to_center < search_radius + obs['radius']:
                # Add multiple points around obstacle perimeter
                for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
                    ox = obs['x'] + obs['radius'] * math.cos(angle)
                    oy = obs['y'] + obs['radius'] * math.sin(angle)
                    obstacles.append((ox, oy))
        
        return obstacles


def draw_robot(surface, pose, color=(60, 140, 220), radius=ROBOT_RADIUS):
    """Draw professional robot"""
    x, y, theta = pose
    
    # Professional shadow effect
    for i in range(2):
        alpha = 40 - i * 20
        glow_r = radius + (2 - i) * 4
        s = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, alpha), (glow_r, glow_r), glow_r)
        surface.blit(s, (int(x - glow_r), int(y - glow_r)))
    
    # Main body with professional colors
    pygame.draw.circle(surface, color, (int(x), int(y)), radius)
    pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)), radius, 2)
    
    # Inner circle for depth
    pygame.draw.circle(surface, (80, 160, 240), (int(x), int(y)), radius // 2)
    
    # Professional direction indicator
    end_x = x + (radius + 10) * math.cos(theta)
    end_y = y + (radius + 10) * math.sin(theta)
    pygame.draw.line(surface, (255, 255, 255), (int(x), int(y)), 
                    (int(end_x), int(end_y)), 3)
    
    # Direction arrow head
    arrow_size = 6
    angle1 = theta + math.pi * 0.8
    angle2 = theta - math.pi * 0.8
    arrow_x1 = end_x + arrow_size * math.cos(angle1)
    arrow_y1 = end_y + arrow_size * math.sin(angle1)
    arrow_x2 = end_x + arrow_size * math.cos(angle2)
    arrow_y2 = end_y + arrow_size * math.sin(angle2)
    
    pygame.draw.polygon(surface, (255, 255, 255), 
                       [(int(end_x), int(end_y)), 
                        (int(arrow_x1), int(arrow_y1)),
                        (int(arrow_x2), int(arrow_y2))])


def draw_fov_cone(surface, pose, fov, max_range, env, color):
    """
    Draw FOV cone clipped by walls and physical obstacles
    (ray-cast based, physically correct)
    """

    x, y, theta = pose

    NUM_RAYS = 18          # smoothness of cone edge
    STEP = 6               # ray marching step (pixels)

    points = [(int(x), int(y))]

    for i in range(NUM_RAYS + 1):
        angle = theta - fov / 2 + i * fov / NUM_RAYS

        hit_x, hit_y = x, y

        for d in range(0, int(max_range), STEP):
            test_x = x + d * math.cos(angle)
            test_y = y + d * math.sin(angle)

            # Stop if ray is blocked
            if not env.has_line_of_sight(x, y, test_x, test_y):
                break

            hit_x, hit_y = test_x, test_y

        points.append((int(hit_x), int(hit_y)))

    # Semi-transparent, eye-friendly fill
    fov_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(fov_surface, color, points)
    surface.blit(fov_surface, (0, 0))


def draw_measurement_lines(surface, robot_pose, observations, landmarks, color):
    rx, ry, _ = robot_pose
    for _, _, lm_id in observations:
        lx, ly = landmarks[lm_id]
        pygame.draw.line(surface, color, (int(rx), int(ry)), (int(lx), int(ly)), 2)
