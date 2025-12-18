import pygame, json, math, os
from pygame.locals import *

pygame.init()
WIDTH, HEIGHT = 900, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("SLAM Map Editor")

# Colors
BG = (245, 245, 245)
GRID = (220, 220, 220)
WALL_COLOR = (0, 0, 0)
FURN_COLOR = (160, 32, 240)   # purple
LAND_COLOR = (0, 200, 0)      # green
ROBOT_COLOR = (0, 120, 255)
PREVIEW = (100, 100, 100)

FONT = pygame.font.SysFont(None, 20)
clock = pygame.time.Clock()

GRID_SIZE = 20
WALL_THICKNESS = 24
DEFAULT_FURN = (80, 50)

# data lists
walls = []        # each wall stored as rect tuple (x,y,w,h)
furnitures = []   # rects
landmarks = []    # point tuples (x,y)
robot_start = None

# editor state
mode = "wall"  # "wall", "furn", "land", "robot", "erase"
drawing = False
start_pos = (0,0)
preview_rect = None

# helper funcs
def draw_grid():
    for x in range(0, WIDTH, GRID_SIZE):
        pygame.draw.line(WIN, GRID, (x,0), (x,HEIGHT))
    for y in range(0, HEIGHT, GRID_SIZE):
        pygame.draw.line(WIN, GRID, (0,y), (WIDTH,y))

def snap(p):
    return ( (p[0]//GRID_SIZE)*GRID_SIZE, (p[1]//GRID_SIZE)*GRID_SIZE )

def rect_from_points(a,b, thickness=WALL_THICKNESS):
    x1,y1 = a; x2,y2 = b
    left = min(x1,x2); top = min(y1,y2)
    w = abs(x2-x1); h = abs(y2-y1)
    # if very thin in one dimension, make a thick wall rectangle along the long axis
    if w == 0 and h == 0:
        w = thickness; h = thickness
    if w < thickness and h < thickness:
        w = max(w, thickness); h = max(h, thickness)
    return (left, top, w, h)

def point_in_rect(pt, rect):
    rx,ry,rw,rh = rect
    return rx <= pt[0] <= rx+rw and ry <= pt[1] <= ry+rh

def save_map(filename="map_saved.json"):
    data = {
        "walls": walls,
        "furnitures": furnitures,
        "landmarks": landmarks,
        "robot_start": robot_start
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print("Map saved to", filename)

def load_map(filename="map_saved.json"):
    global walls, furnitures, landmarks, robot_start
    if not os.path.exists(filename):
        print("No saved map found.")
        return
    with open(filename, "r") as f:
        data = json.load(f)
    walls = data.get("walls", [])
    furnitures = data.get("furnitures", [])
    landmarks = data.get("landmarks", [])
    robot_start = data.get("robot_start", None)
    print("Map loaded from", filename)

# UI hint text
def draw_hud():
    lines = [
        f"Mode: {mode.upper()}   (press W-wall, F-furniture, L-landmark, R-robot, E-erase)",
        "Draw walls: click-drag. Furniture: click to place default size (shift+click to drag-size). Landmark: click.",
        "Save(S), Load(O), Clear(C). Arrow keys to pan (not implemented).",
        "Left-click = add / draw / set. Right-click = cancel drawing."
    ]
    for i,l in enumerate(lines):
        WIN.blit(FONT.render(l, True, (40,40,40)), (8, 6 + i*18))

# main loop
running = True
while running:
    clock.tick(60)
    WIN.fill(BG)
    draw_grid()

    mx,my = pygame.mouse.get_pos()
    for ev in pygame.event.get():
        if ev.type == QUIT:
            running = False
        elif ev.type == KEYDOWN:
            if ev.key == K_w: mode = "wall"
            elif ev.key == K_f: mode = "furn"
            elif ev.key == K_l: mode = "land"
            elif ev.key == K_r: mode = "robot"
            elif ev.key == K_e: mode = "erase"
            elif ev.key == K_s: save_map()
            elif ev.key == K_o: load_map()
            elif ev.key == K_c:
                walls=[]; furnitures=[]; landmarks=[]; robot_start=None
                print("Cleared map")
        elif ev.type == MOUSEBUTTONDOWN:
            if ev.button == 1:  # left click
                if mode == "wall":
                    drawing = True
                    start_pos = snap((mx,my))
                elif mode == "furn":
                    # if shift held, enter drag mode
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        drawing = True
                        start_pos = snap((mx,my))
                    else:
                        sx,sy = snap((mx,my))
                        w,h = DEFAULT_FURN
                        furnitures.append((sx,sy,w,h))
                elif mode == "land":
                    lx,ly = snap((mx+GRID_SIZE//2,my+GRID_SIZE//2))
                    landmarks.append((lx,ly))
                elif mode == "robot":
                    robot_start = snap((mx,my))
                elif mode == "erase":
                    p = (mx,my)
                    # remove walls/furn/land near point
                    removed = False
                    # top priority remove furn
                    for i,rect in enumerate(furnitures):
                        if point_in_rect(p, rect):
                            furnitures.pop(i); removed=True; break
                    if not removed:
                        for i,rect in enumerate(walls):
                            if point_in_rect(p, rect):
                                walls.pop(i); removed=True; break
                    if not removed:
                        # landmarks
                        for i,pt in enumerate(landmarks):
                            dx = pt[0]-p[0]; dy = pt[1]-p[1]
                            if dx*dx+dy*dy < 12*12:
                                landmarks.pop(i); removed=True; break
                # end left click
            elif ev.button == 3:  # right click cancels drawing
                drawing = False
                preview_rect = None
        elif ev.type == MOUSEBUTTONUP:
            if ev.button == 1 and drawing:
                if mode == "wall":
                    end = snap((mx,my))
                    rect = rect_from_points(start_pos, end)
                    # ensure thickness: convert thin line to thick long rect
                    x,y,wid,hei = rect
                    if wid < 10 or hei < 10:
                        # make a corridor-like wall with thickness on the longer axis
                        if abs(end[0]-start_pos[0]) > abs(end[1]-start_pos[1]):
                            # horizontal
                            ry = start_pos[1] - WALL_THICKNESS//2
                            rect = (min(start_pos[0],end[0]), ry, abs(end[0]-start_pos[0]), WALL_THICKNESS)
                        else:
                            rx = start_pos[0] - WALL_THICKNESS//2
                            rect = (rx, min(start_pos[1],end[1]), WALL_THICKNESS, abs(end[1]-start_pos[1]))
                    walls.append(rect)
                    preview_rect = None
                elif mode == "furn":
                    end = snap((mx,my))
                    rect = rect_from_points(start_pos, end, thickness=0)
                    furnitures.append(rect)
                    preview_rect = None
                drawing = False

    # update preview if drawing
    if drawing:
        if mode == "wall":
            cur = snap((mx,my))
            preview_rect = rect_from_points(start_pos, cur)
        elif mode == "furn":
            cur = snap((mx,my))
            preview_rect = rect_from_points(start_pos, cur, thickness=0)
    else:
        preview_rect = None

    # draw walls
    for rect in walls:
        pygame.draw.rect(WIN, WALL_COLOR, rect)

    # draw furnitures (purple)
    for rect in furnitures:
        pygame.draw.rect(WIN, FURN_COLOR, rect)

    # draw landmarks as small green circles
    for pt in landmarks:
        pygame.draw.circle(WIN, LAND_COLOR, (int(pt[0]), int(pt[1])), 5)

    # draw robot start marker
    if robot_start:
        pygame.draw.circle(WIN, ROBOT_COLOR, (robot_start[0], robot_start[1]), 10, 2)
        WIN.blit(FONT.render("ROBOT", True, (30,30,30)), (robot_start[0]+12, robot_start[1]-6))

    # draw preview rect if any
    if preview_rect:
        pygame.draw.rect(WIN, PREVIEW, preview_rect, 2)

    draw_hud()
    pygame.display.flip()

pygame.quit()
