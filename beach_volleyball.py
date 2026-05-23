"""
╔══════════════════════════════════════════════════════════╗
║      🏖️  FACE VOLLEY  —  Beach Volleyball Face Game      ║
║         2-Player • Face-Tracked • Physics-Based          ║
╚══════════════════════════════════════════════════════════╝

Controls:
  - Move your face LEFT/RIGHT to control your player
  - Move your face UP QUICKLY to JUMP
  - Player 1 = Left half of webcam
  - Player 2 = Right half of webcam

Requirements:
  pip install pygame opencv-python mediapipe numpy
"""

import pygame
import cv2
import mediapipe as mp
import numpy as np
import math
import time
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ─────────────────────────────────────────────
#  CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────

WIDTH, HEIGHT     = 1280, 720
FPS               = 60
GROUND_Y          = HEIGHT - 100        # Sand surface
NET_X             = WIDTH // 2
NET_HEIGHT        = 200
NET_TOP_Y         = GROUND_Y - NET_HEIGHT
NET_HALF_W        = 8                   # physics collision half-width of net post

GRAVITY           = 0.55
BALL_RADIUS       = 90
BOUNCE_DAMPENING  = 0.6
BALL_SPEED_SCALE  = 1.10               # multiplied on each head hit

JUMP_VELOCITY     = -14.0              # px/frame (negative = up)
JUMP_THRESHOLD    = 12                 # face pixel movement to trigger jump
JUMP_COOLDOWN     = 0.45              # seconds
FACE_SMOOTH       = 0.35              # lower = more smoothing

PLAYER_RADIUS     = 42
PLAYER_GROUND_Y   = GROUND_Y - PLAYER_RADIUS

WIN_SCORE         = 7                  # first to 7 wins

RALLY_TIME_LIMIT  = 80.0              # seconds per side before opponent scores

# ── Palette ─────────────────────────────────
SKY_TOP        = (100, 185, 255)
SKY_BOT        = (180, 225, 255)
OCEAN_TOP      = (35,  120, 190)
OCEAN_BOT      = (20,   80, 150)
SAND_TOP       = (245, 210, 130)
SAND_BOT       = (220, 185, 100)
SUN_COLOR      = (255, 235,  80)
CLOUD_COLOR    = (255, 255, 255)
NET_COLOR      = (255, 255, 255)
SCORE_BG       = (0,   0,   0, 160)
P1_COLOR       = (255, 100,  80)      # coral-red
P2_COLOR       = ( 80, 170, 255)      # ocean-blue
P1_DARK        = (200,  50,  30)
P2_DARK        = ( 30, 110, 210)
BALL_WHITE     = (255, 255, 255)
BALL_LINE      = (210, 210, 210)
TEXT_LIGHT     = (255, 255, 255)
TEXT_SHADOW    = ( 20,  20,  40)
BTN_COLOR      = (255, 180,  60)
BTN_HOVER      = (255, 210, 100)
BTN_TEXT       = ( 50,  30,   0)


# ─────────────────────────────────────────────
#  FACE TRACKER  (MediaPipe)
# ─────────────────────────────────────────────

class FaceTracker:
    """Wraps MediaPipe FaceDetection to track 2 faces.
    Also stores face crops (as RGB numpy arrays) for rendering on player circles.
    """

    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.detector = self.mp_face.FaceDetection(
            model_selection=0, min_detection_confidence=0.55
        )
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Smoothed positions for each player [x, y] in webcam coords (0-1)
        self.positions: List[Optional[Tuple[float, float]]] = [None, None]
        self.raw_prev:  List[Optional[Tuple[float, float]]] = [None, None]

        # Face crops (RGB numpy arrays, size PLAYER_RADIUS*2 x PLAYER_RADIUS*2)
        # Updated each frame when a face is detected.
        self.face_crops: List[Optional[np.ndarray]] = [None, None]

        # Last full frame (RGB, flipped) for webcam preview
        self.last_frame: Optional[np.ndarray] = None

    def update(self) -> List[Optional[Tuple[float, float]]]:
        ret, frame = self.cap.read()
        if not ret:
            return self.positions

        frame = cv2.flip(frame, 1)   # mirror so movement is intuitive
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.last_frame = rgb

        result = self.detector.process(rgb)

        fh, fw = rgb.shape[:2]

        faces_left:  List[Tuple[float, float, float, object]] = []   # (cx, cy, area, det)
        faces_right: List[Tuple[float, float, float, object]] = []

        if result.detections:
            for det in result.detections:
                bb  = det.location_data.relative_bounding_box
                cx  = bb.xmin + bb.width  / 2
                cy  = bb.ymin + bb.height / 2
                area = bb.width * bb.height
                if cx < 0.5:
                    faces_left.append((cx, cy, area, bb))
                else:
                    faces_right.append((cx, cy, area, bb))

        def best(lst):
            return max(lst, key=lambda f: f[2]) if lst else None

        raw_left  = best(faces_left)
        raw_right = best(faces_right)
        raw = [raw_left, raw_right]

        for i in range(2):
            if raw[i]:
                rx, ry, _, bb = raw[i]
                if self.positions[i] is None:
                    self.positions[i] = (rx, ry)
                else:
                    ox, oy = self.positions[i]
                    self.positions[i] = (
                        ox + FACE_SMOOTH * (rx - ox),
                        oy + FACE_SMOOTH * (ry - oy),
                    )

                # ── Extract face crop ──
                # Use the bounding box to crop the face from the RGB frame,
                # with a small margin so we get a bit of neck/hair.
                margin = 0.10
                x1 = max(0, int((bb.xmin - margin * bb.width) * fw))
                y1 = max(0, int((bb.ymin - margin * bb.height) * fh))
                x2 = min(fw, int((bb.xmin + (1 + margin) * bb.width) * fw))
                y2 = min(fh, int((bb.ymin + (1 + margin) * bb.height) * fh))
                crop = rgb[y1:y2, x1:x2]
                if crop.size > 0:
                    diameter = PLAYER_RADIUS * 2
                    self.face_crops[i] = cv2.resize(crop, (diameter, diameter))
            # If face not detected this frame, keep last known position & crop

        return self.positions

    def release(self):
        self.cap.release()
        self.detector.close()


# ─────────────────────────────────────────────
#  PHYSICS — BALL
# ─────────────────────────────────────────────

@dataclass
class Ball:
    x: float = WIDTH  / 2
    y: float = HEIGHT / 3
    vx: float = 3.5
    vy: float = 0.0
    radius: int = BALL_RADIUS
    rotation: float = 0.0    # visual spin angle

    def update(self):
        self.vy += GRAVITY
        self.x  += self.vx
        self.y  += self.vy
        self.rotation += self.vx * 1.5   # spin based on horizontal speed

    def bounce_off_head(self, px: float, py: float, player_side: int):
        """Bounce ball away from player."""
        dx = self.x - px
        dy = self.y - py
        dist = math.hypot(dx, dy) or 1
        nx, ny = dx / dist, dy / dist

        # Reflect velocity
        dot = self.vx * nx + self.vy * ny
        self.vx = (self.vx - 2 * dot * nx) * BALL_SPEED_SCALE
        self.vy = (self.vy - 2 * dot * ny) * BALL_SPEED_SCALE

        # Always give upward kick
        if self.vy > -4:
            self.vy = -9.5

        # Push ball outside overlap zone — no net clamping
        min_dist = self.radius + PLAYER_RADIUS + 2
        if dist < min_dist:
            self.x = px + nx * min_dist
            self.y = py + ny * min_dist

        # Cap speed
        speed = math.hypot(self.vx, self.vy)
        MAX = 14.5
        if speed > MAX:
            self.vx = self.vx / speed * MAX
            self.vy = self.vy / speed * MAX

    def reset_for_serve(self, server_side: int):
        """Place ball at the center of the server's court half, at a jumpable height.
        The ball floats still — player must jump up and hit it to start the rally.
        """
        quarter = WIDTH // 4
        # Center of left half = WIDTH/4, center of right half = 3*WIDTH/4
        self.x  = quarter if server_side == 1 else WIDTH - quarter
        # Sit the ball just above jump-reach: player jumps ~180px, ball at PLAYER_GROUND_Y - 160
        self.y  = PLAYER_GROUND_Y - PLAYER_RADIUS - self.radius - 60
        self.vx = 0.0
        self.vy = 0.0
        self.rotation = 0.0

    def reset(self, toward_player: int = 1):
        direction = 1 if toward_player == 2 else -1
        self.x  = WIDTH / 2
        self.y  = NET_TOP_Y - self.radius - 20
        self.vx = 4.5 * direction
        self.vy = -2.0
        self.rotation = 0.0


# ─────────────────────────────────────────────
#  PHYSICS — PLAYER
# ─────────────────────────────────────────────

@dataclass
class Player:
    side: int          # 1 = left, 2 = right
    x: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    on_ground: bool = True
    last_face_y: Optional[float] = None
    jump_cooldown_until: float = 0.0
    score: int = 0
    color: Tuple = P1_COLOR
    dark:  Tuple = P1_DARK
    name: str = "P1"
    emoji: str = "🔴"

    def __post_init__(self):
        half = WIDTH // 4
        self.x = half if self.side == 1 else WIDTH - half
        self.y = PLAYER_GROUND_Y
        if self.side == 2:
            self.color = P2_COLOR
            self.dark  = P2_DARK
            self.name  = "P2"
            self.emoji = "🔵"

    def apply_face(self, face_pos: Optional[Tuple[float, float]], now: float):
        """Map face position → game position, detect jump gesture."""
        if face_pos is None:
            return

        face_x, face_y = face_pos

        # ── Horizontal mapping ──
        half_w = WIDTH // 2
        if self.side == 1:
            target_x = np.interp(face_x, [0, 0.5], [PLAYER_RADIUS + 10, half_w - PLAYER_RADIUS - 10])
        else:
            target_x = np.interp(face_x, [0.5, 1.0], [half_w + PLAYER_RADIUS + 10, WIDTH - PLAYER_RADIUS - 10])

        self.x += 0.25 * (target_x - self.x)   # smooth horizontal approach

        # ── Jump detection: upward face flick triggers a jump ──
        if self.last_face_y is not None and now >= self.jump_cooldown_until:
            delta_y = (self.last_face_y - face_y) * HEIGHT   # upward = positive
            if delta_y > JUMP_THRESHOLD and self.on_ground:
                self.vy = JUMP_VELOCITY
                self.on_ground = False
                self.jump_cooldown_until = now + JUMP_COOLDOWN

        self.last_face_y = face_y

    def update_physics(self):
        if not self.on_ground:
            self.vy += GRAVITY * 1.1
            self.y  += self.vy
            if self.y >= PLAYER_GROUND_Y:
                self.y        = PLAYER_GROUND_Y
                self.vy       = 0.0
                self.on_ground = True
        else:
            self.y  = PLAYER_GROUND_Y
            self.vy = 0.0

    def collision_with_ball(self, ball: Ball) -> bool:
        dist = math.hypot(ball.x - self.x, ball.y - self.y)
        return dist < self.radius + ball.radius

    @property
    def radius(self):
        return PLAYER_RADIUS


# ─────────────────────────────────────────────
#  PARTICLE SYSTEM  (score celebration)
# ─────────────────────────────────────────────

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple
    life: float = 1.0
    size: float = 6.0

def spawn_particles(px, py, color, count=28) -> List[Particle]:
    parts = []
    for _ in range(count):
        angle = np.random.uniform(0, 2 * math.pi)
        speed = np.random.uniform(3, 10)
        parts.append(Particle(
            x=px, y=py,
            vx=math.cos(angle) * speed,
            vy=math.sin(angle) * speed - 3,
            color=color,
            life=1.0,
            size=np.random.uniform(4, 9)
        ))
    return parts

def update_particles(parts: List[Particle]) -> List[Particle]:
    alive = []
    for p in parts:
        p.x    += p.vx
        p.y    += p.vy
        p.vy   += 0.3
        p.life -= 0.025
        p.size *= 0.97
        if p.life > 0:
            alive.append(p)
    return alive


# ─────────────────────────────────────────────
#  RENDERER  (all drawing)
# ─────────────────────────────────────────────

# Cache for circular-masked face textures (pygame Surface)
# Key: player index (0 or 1)
_face_surf_cache: List[Optional[pygame.Surface]] = [None, None]

def make_circular_face_surf(crop_rgb: np.ndarray, diameter: int) -> pygame.Surface:
    """Convert an RGB numpy face crop into a circular pygame Surface."""
    resized = cv2.resize(crop_rgb, (diameter, diameter))
    surf = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    # Draw the image pixel by pixel via surfarray (fast path)
    img_surf = pygame.surfarray.make_surface(np.transpose(resized, (1, 0, 2)))
    surf.blit(img_surf, (0, 0))
    # Mask to circle using alpha
    r = diameter // 2
    mask = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    mask.fill((0, 0, 0, 0))
    pygame.draw.circle(mask, (255, 255, 255, 255), (r, r), r)
    # Apply mask: only keep pixels inside the circle
    for x in range(diameter):
        for y in range(diameter):
            if mask.get_at((x, y))[3] == 0:
                surf.set_at((x, y), (0, 0, 0, 0))
    return surf

def make_circular_face_surf_fast(crop_rgb: np.ndarray, diameter: int) -> pygame.Surface:
    """Faster circular mask via numpy alpha channel."""
    resized = cv2.resize(crop_rgb, (diameter, diameter))
    # Add alpha channel
    rgba = np.dstack([resized, np.full((diameter, diameter), 255, dtype=np.uint8)])
    # Build circle mask
    r = diameter / 2
    y_idx, x_idx = np.ogrid[:diameter, :diameter]
    dist2 = (x_idx - r + 0.5)**2 + (y_idx - r + 0.5)**2
    outside = dist2 > r**2
    rgba[outside, 3] = 0
    # pygame surfarray expects (width, height, channels) — transpose
    surf = pygame.surfarray.make_surface(np.transpose(rgba[:, :, :3], (1, 0, 2)))
    surf.set_colorkey(None)
    surf_alpha = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    surf_alpha.blit(surf, (0, 0))
    # Apply alpha mask
    alpha_arr = np.transpose(rgba[:, :, 3], (1, 0))  # (w, h)
    pygame.surfarray.pixels_alpha(surf_alpha)[:] = alpha_arr
    return surf_alpha


class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.font_big   = pygame.font.SysFont("Arial Rounded MT Bold", 72, bold=True)
        self.font_mid   = pygame.font.SysFont("Arial Rounded MT Bold", 36, bold=True)
        self.font_small = pygame.font.SysFont("Arial Rounded MT Bold", 24)
        self.font_tiny  = pygame.font.SysFont("Arial Rounded MT Bold", 18)

        # Pre-render static layers
        self._bg = self._make_bg()
        self._clouds = self._gen_clouds()
        self._cloud_offset = 0.0

        # Face surface cache per player (updated when crop changes)
        self._face_surfs: List[Optional[pygame.Surface]] = [None, None]
        self._face_crop_ids: List[int] = [-1, -1]   # id() of last crop used

    # ── Background ──────────────────────────

    def _make_bg(self) -> pygame.Surface:
        surf = pygame.Surface((WIDTH, HEIGHT))
        for y in range(HEIGHT):
            t = y / HEIGHT
            r = int(SKY_TOP[0] + (SKY_BOT[0] - SKY_TOP[0]) * t)
            g = int(SKY_TOP[1] + (SKY_BOT[1] - SKY_TOP[1]) * t)
            b = int(SKY_TOP[2] + (SKY_BOT[2] - SKY_TOP[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (WIDTH, y))
        ocean_top_y = GROUND_Y - 160
        for y in range(ocean_top_y, GROUND_Y):
            t = (y - ocean_top_y) / (GROUND_Y - ocean_top_y)
            r = int(OCEAN_TOP[0] + (OCEAN_BOT[0] - OCEAN_TOP[0]) * t)
            g = int(OCEAN_TOP[1] + (OCEAN_BOT[1] - OCEAN_TOP[1]) * t)
            b = int(OCEAN_TOP[2] + (OCEAN_BOT[2] - OCEAN_TOP[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (WIDTH, y))
        for y in range(GROUND_Y, HEIGHT):
            t = (y - GROUND_Y) / (HEIGHT - GROUND_Y)
            r = int(SAND_TOP[0] + (SAND_BOT[0] - SAND_TOP[0]) * t)
            g = int(SAND_TOP[1] + (SAND_BOT[1] - SAND_TOP[1]) * t)
            b = int(SAND_TOP[2] + (SAND_BOT[2] - SAND_TOP[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, y), (WIDTH, y))
        pygame.draw.circle(surf, SUN_COLOR, (120, 90), 55)
        pygame.draw.circle(surf, (255, 255, 200), (120, 90), 40)
        for i in range(12):
            angle = math.radians(i * 30)
            x1 = 120 + math.cos(angle) * 62
            y1 =  90 + math.sin(angle) * 62
            x2 = 120 + math.cos(angle) * 80
            y2 =  90 + math.sin(angle) * 80
            pygame.draw.line(surf, SUN_COLOR, (int(x1), int(y1)), (int(x2), int(y2)), 3)
        return surf

    def _gen_clouds(self):
        clouds = []
        positions = [(200, 80), (500, 50), (850, 95), (1100, 65), (1350, 80)]
        for (cx, cy) in positions:
            blobs = [
                (cx, cy, 40), (cx+50, cy-10, 55), (cx+110, cy, 42),
                (cx+60, cy+15, 35), (cx+20, cy+12, 30)
            ]
            clouds.append(blobs)
        return clouds

    def draw_background(self):
        self.screen.blit(self._bg, (0, 0))
        self._cloud_offset = (self._cloud_offset + 0.2) % (WIDTH + 300)
        for blobs in self._clouds:
            for (cx, cy, r) in blobs:
                draw_x = int((cx + self._cloud_offset) % (WIDTH + 300)) - 150
                pygame.draw.circle(self.screen, CLOUD_COLOR, (draw_x, cy), r)
        t = time.time()
        for i in range(6):
            y  = GROUND_Y - 140 + i * 22
            ox = int(math.sin(t * 1.5 + i) * 30)
            alpha_surf = pygame.Surface((WIDTH, 3), pygame.SRCALPHA)
            alpha_surf.fill((255, 255, 255, 30))
            self.screen.blit(alpha_surf, (ox, y))

    # ── Net ─────────────────────────────────
    # FIX: Removed the wide horizontal top-border line that created a
    # visually confusing horizontal bar and matched the invisible collision wall.
    # The net is now drawn as a proper vertical grid with post and top rope
    # only at the net post width — no wide horizontal line across the screen.

    def draw_net(self):
        # Only the center post — no net grid, no collision wall
        pygame.draw.rect(self.screen, (200, 170, 100),
                         (NET_X - 5, NET_TOP_Y - 10, 10, NET_HEIGHT + 10))

    # ── Ball ────────────────────────────────

    def draw_ball(self, ball: Ball):
        cx, cy = int(ball.x), int(ball.y)
        r = ball.radius
        shadow = pygame.Surface((r * 4, r * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow, (0, 0, 0, 60), (0, 0, r * 4, r * 2))
        self.screen.blit(shadow, (cx - r * 2, GROUND_Y + 2))
        pygame.draw.circle(self.screen, BALL_WHITE, (cx, cy), r)
        for i in range(3):
            angle = math.radians(ball.rotation + i * 60)
            x1 = cx + math.cos(angle) * r
            y1 = cy + math.sin(angle) * r
            x2 = cx - math.cos(angle) * r
            y2 = cy - math.sin(angle) * r
            pygame.draw.line(self.screen, BALL_LINE, (int(x1), int(y1)), (int(x2), int(y2)), 2)
        pygame.draw.circle(self.screen, (180, 180, 180), (cx, cy), r, 2)

    # ── Player ──────────────────────────────

    def _get_face_surf(self, player_idx: int, tracker: "FaceTracker") -> Optional[pygame.Surface]:
        """Return a circular face surface for the player, cached until crop changes."""
        crop = tracker.face_crops[player_idx]
        if crop is None:
            return None
        crop_id = id(crop)
        if crop_id != self._face_crop_ids[player_idx]:
            diameter = PLAYER_RADIUS * 2
            try:
                self._face_surfs[player_idx] = make_circular_face_surf_fast(crop, diameter)
            except Exception:
                self._face_surfs[player_idx] = None
            self._face_crop_ids[player_idx] = crop_id
        return self._face_surfs[player_idx]

    def draw_player(self, p: Player, tracker: Optional["FaceTracker"] = None):
        cx, cy = int(p.x), int(p.y)
        r = p.radius

        # Shadow on ground
        shadow_y = PLAYER_GROUND_Y + r + 4
        shadow_w = max(10, int(r * 2 * (1 - (PLAYER_GROUND_Y - cy) / (HEIGHT * 0.6))))
        shadow_s = pygame.Surface((shadow_w * 2, 14), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_s, (0, 0, 0, 60), (0, 0, shadow_w * 2, 14))
        self.screen.blit(shadow_s, (cx - shadow_w, shadow_y))

        # Body circle (always drawn — face is layered on top)
        pygame.draw.circle(self.screen, p.color, (cx, cy), r)
        pygame.draw.circle(self.screen, p.dark,  (cx, cy), r, 4)

        # ── Webcam face overlay ──
        # If we have a face crop from the tracker, stamp it inside the circle.
        # Otherwise fall back to the drawn emoji face.
        face_drawn = False
        if tracker is not None:
            player_idx = p.side - 1  # side 1 → index 0, side 2 → index 1
            face_surf = self._get_face_surf(player_idx, tracker)
            if face_surf is not None:
                self.screen.blit(face_surf, (cx - r, cy - r))
                # Re-draw the ring border on top of the face
                pygame.draw.circle(self.screen, p.dark, (cx, cy), r, 4)
                face_drawn = True

        if not face_drawn:
            # Fallback: drawn eyes + smile
            eye_r  = 5
            eye_ox = 12
            eye_y  = cy - 8
            pygame.draw.circle(self.screen, (30, 10, 10), (cx - eye_ox, eye_y), eye_r)
            pygame.draw.circle(self.screen, (30, 10, 10), (cx + eye_ox, eye_y), eye_r)
            smile_rect = pygame.Rect(cx - 14, cy + 2, 28, 20)
            pygame.draw.arc(self.screen, (30, 10, 10), smile_rect,
                            math.radians(200), math.radians(340), 3)

        # Name tag
        tag = self.font_tiny.render(p.name, True, TEXT_LIGHT)
        self.screen.blit(tag, (cx - tag.get_width() // 2, cy + r + 4))

    # ── Score HUD ───────────────────────────

    CAM_BOTTOM_Y = 210

    def draw_hud(self, p1: Player, p2: Player, paused: bool,
                 side_with_ball: int = 0, time_left: float = RALLY_TIME_LIMIT):
        """Draw score + rally timer."""
        side_w = WIDTH // 2 - 345
        bar_l  = pygame.Surface((side_w, 70), pygame.SRCALPHA)
        bar_l.fill((10, 20, 60, 170))
        bar_r  = pygame.Surface((side_w, 70), pygame.SRCALPHA)
        bar_r.fill((10, 20, 60, 170))
        self.screen.blit(bar_l, (0, 0))
        self.screen.blit(bar_r, (WIDTH - side_w, 0))

        s1 = self.font_big.render(str(p1.score), True, P1_COLOR)
        self.screen.blit(s1, (side_w // 2 - s1.get_width() // 2, 0))

        s2 = self.font_big.render(str(p2.score), True, P2_COLOR)
        rx = WIDTH - side_w + side_w // 2 - s2.get_width() // 2
        self.screen.blit(s2, (rx, 0))

        n1 = self.font_small.render("● " + p1.name, True, P1_COLOR)
        n2 = self.font_small.render(p2.name + " ●", True, P2_COLOR)
        self.screen.blit(n1, (8, 52))
        self.screen.blit(n2, (WIDTH - 8 - n2.get_width(), 52))

        # ── Rally timer ──
        if side_with_ball in (1, 2):
            pct = max(0.0, time_left / RALLY_TIME_LIMIT)
            bar_w_max = 260
            bar_w = int(bar_w_max * pct)
            bar_y = HEIGHT - 28
            bar_h = 14

            if side_with_ball == 1:
                bx = 20
                tc = P1_COLOR
            else:
                bx = WIDTH - 20 - bar_w_max
                tc = P2_COLOR

            # Background track
            pygame.draw.rect(self.screen, (50, 50, 80),
                             (bx, bar_y, bar_w_max, bar_h), border_radius=7)
            # Filled bar — turns red when < 15 seconds
            fill_color = (255, 60, 60) if time_left < 15 else tc
            if bar_w > 0:
                pygame.draw.rect(self.screen, fill_color,
                                 (bx, bar_y, bar_w, bar_h), border_radius=7)
            pygame.draw.rect(self.screen, (200, 200, 200),
                             (bx, bar_y, bar_w_max, bar_h), 2, border_radius=7)

            # Timer text
            timer_txt = self.font_tiny.render(f"{int(time_left)}s", True, TEXT_LIGHT)
            self.screen.blit(timer_txt, (bx + bar_w_max // 2 - timer_txt.get_width() // 2,
                                         bar_y - 2))

        if paused:
            pause_txt = self.font_mid.render("⏸  PAUSED", True, (255, 230, 80))
            self.screen.blit(pause_txt, (WIDTH // 2 - pause_txt.get_width() // 2, HEIGHT // 2 - 20))

    # ── Particles ───────────────────────────

    def draw_particles(self, parts: List[Particle]):
        for p in parts:
            alpha = int(p.life * 255)
            size  = max(2, int(p.size))
            surf  = pygame.Surface((size * 2, size * 2), pygame.SRCALPHA)
            col   = (*p.color, alpha)
            pygame.draw.circle(surf, col, (size, size), size)
            self.screen.blit(surf, (int(p.x) - size, int(p.y) - size))

    # ── Overlay Buttons ─────────────────────

    def draw_button(self, text: str, rect: pygame.Rect, hovered: bool) -> None:
        color = BTN_HOVER if hovered else BTN_COLOR
        pygame.draw.rect(self.screen, color, rect, border_radius=16)
        pygame.draw.rect(self.screen, (200, 140, 20), rect, 3, border_radius=16)
        label = self.font_mid.render(text, True, BTN_TEXT)
        self.screen.blit(label, (
            rect.centerx - label.get_width() // 2,
            rect.centery - label.get_height() // 2
        ))

    def draw_start_screen(self, btn_rect: pygame.Rect, mouse_pos):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 20, 60, 180))
        self.screen.blit(overlay, (0, 0))

        title = self.font_big.render("🏖️  FACE VOLLEY", True, (255, 235, 80))
        sub   = self.font_mid.render("Move your face to play  •  Jump by looking UP fast!", True, TEXT_LIGHT)
        p1l   = self.font_small.render("🔴 Player 1 = Left side of webcam", True, P1_COLOR)
        p2l   = self.font_small.render("🔵 Player 2 = Right side of webcam", True, P2_COLOR)
        timer_info = self.font_small.render(f"⏱  {int(RALLY_TIME_LIMIT)}s rally timer — hit it before time runs out!", True, (255, 220, 80))

        self.screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 180))
        self.screen.blit(sub,   (WIDTH // 2 - sub.get_width() // 2,   270))
        self.screen.blit(p1l,   (WIDTH // 2 - p1l.get_width() // 2,   325))
        self.screen.blit(p2l,   (WIDTH // 2 - p2l.get_width() // 2,   365))
        self.screen.blit(timer_info, (WIDTH // 2 - timer_info.get_width() // 2, 410))

        hovered = btn_rect.collidepoint(mouse_pos)
        self.draw_button("▶  START GAME", btn_rect, hovered)

    def draw_win_screen(self, winner: Player, btn_rect: pygame.Rect, mouse_pos):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 10, 40, 200))
        self.screen.blit(overlay, (0, 0))

        win_txt = self.font_big.render(f"🏆  {winner.name} WINS!", True, winner.color)
        self.screen.blit(win_txt, (WIDTH // 2 - win_txt.get_width() // 2, 240))

        hovered = btn_rect.collidepoint(mouse_pos)
        self.draw_button("🔄  PLAY AGAIN", btn_rect, hovered)

    def draw_webcam_preview(self, face_tracker: "FaceTracker"):
        """Split webcam: P1 (left half) | P2 (right half), centered at top."""
        if face_tracker.last_frame is None:
            return
        frame = face_tracker.last_frame   # already flipped & RGB from update()

        cam_w, cam_h = 320, 200
        gap          = 6
        border       = 3
        total_w      = cam_w * 2 + gap
        start_x      = WIDTH // 2 - total_w // 2
        start_y      = 4

        fh, fw = frame.shape[:2]
        left_half  = frame[:, :fw // 2, :]
        right_half = frame[:, fw // 2:, :]

        left_resized  = cv2.resize(left_half,  (cam_w, cam_h))
        right_resized = cv2.resize(right_half, (cam_w, cam_h))

        def to_surf(arr):
            return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))

        surf_l = to_surf(left_resized)
        surf_r = to_surf(right_resized)

        x_left  = start_x
        x_right = start_x + cam_w + gap

        pill = pygame.Surface((total_w + 16, cam_h + 28), pygame.SRCALPHA)
        pill.fill((0, 10, 40, 200))
        pygame.draw.rect(pill, (0, 10, 40, 200),
                         pill.get_rect(), border_radius=14)
        self.screen.blit(pill, (start_x - 8, start_y - 6))

        for surf, px, color, label_txt in [
            (surf_l, x_left,  P1_COLOR, "🔴 P1"),
            (surf_r, x_right, P2_COLOR, "🔵 P2"),
        ]:
            pygame.draw.rect(self.screen, color,
                             (px - border, start_y - border,
                              cam_w + border * 2, cam_h + border * 2),
                             border_radius=6)
            self.screen.blit(surf, (px, start_y))

            lbl = self.font_tiny.render(label_txt, True, TEXT_LIGHT)
            lbl_bg = pygame.Surface((lbl.get_width() + 10, lbl.get_height() + 4), pygame.SRCALPHA)
            lbl_bg.fill((*color, 180))
            self.screen.blit(lbl_bg, (px + 4, start_y + cam_h - lbl.get_height() - 6))
            self.screen.blit(lbl,    (px + 9, start_y + cam_h - lbl.get_height() - 4))

        div_x = x_left + cam_w + gap // 2
        pygame.draw.line(self.screen, (255, 255, 255),
                         (div_x, start_y), (div_x, start_y + cam_h), 2)

    def draw_pause_button(self, rect: pygame.Rect, paused: bool, mouse_pos):
        hovered = rect.collidepoint(mouse_pos)
        label   = "▶ Resume" if paused else "⏸ Pause"
        self.draw_button(label, rect, hovered)

    def draw_restart_button(self, rect: pygame.Rect, mouse_pos):
        hovered = rect.collidepoint(mouse_pos)
        self.draw_button("🔄 Restart", rect, hovered)

    def draw_serve_hint(self, server: Player, ball: "Ball"):
        """Arrow + text above the floating serve ball."""
        hint = self.font_small.render("Jump & Hit to Serve!", True, (255, 235, 80))
        hx = int(ball.x) - hint.get_width() // 2
        hy = int(ball.y) - ball.radius - hint.get_height() - 12
        bg = pygame.Surface((hint.get_width() + 14, hint.get_height() + 8), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 130))
        self.screen.blit(bg, (hx - 7, hy - 4))
        self.screen.blit(hint, (hx, hy))
        # Small downward arrow pointing at the ball
        ax = int(ball.x)
        ay = int(ball.y) - ball.radius - 6
        pygame.draw.polygon(self.screen, (255, 235, 80),
                            [(ax, ay), (ax - 10, ay - 16), (ax + 10, ay - 16)])


# ─────────────────────────────────────────────
#  GAME MANAGER
# ─────────────────────────────────────────────

class GameState:
    START   = "start"
    SERVING = "serving"   # server has ball floating above them
    PLAYING = "playing"
    PAUSED  = "paused"
    WIN     = "win"


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("🏖️  Face Volley")

        self.screen   = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock    = pygame.time.Clock()
        self.renderer = Renderer(self.screen)

        self.tracker  = FaceTracker()
        self.ball     = Ball()
        self.p1       = Player(side=1)
        self.p2       = Player(side=2)

        self.particles: List[Particle] = []
        self.state     = GameState.START

        # UI button rects
        self.btn_start   = pygame.Rect(WIDTH // 2 - 130, 460, 260, 60)
        self.btn_pause   = pygame.Rect(20,  HEIGHT - 60, 140, 44)
        self.btn_restart = pygame.Rect(175, HEIGHT - 60, 150, 44)

        # Score flash timer
        self._flash_msg      = ""
        self._flash_until    = 0.0

        # ── Rally timer state ──
        # side_with_ball: 1 or 2 — whose side the ball is currently on
        self._side_with_ball: int = 1
        self._side_timer_start: float = 0.0   # time.time() when ball entered this side

        # ── Serve state ──
        # After scoring, the scoring player gets the ball above them to serve.
        self._serving_player: Optional[Player] = None  # player who is serving

    def _ball_side(self) -> int:
        """Return 1 if ball is on P1's side, 2 if on P2's side."""
        return 1 if self.ball.x < NET_X else 2

    def reset(self):
        self.p1.score = 0
        self.p2.score = 0
        self.p1.y = PLAYER_GROUND_Y
        self.p2.y = PLAYER_GROUND_Y
        # P1 serves first
        self._start_serve(self.p1)

    def _start_serve(self, server: Player):
        """Put game into SERVING state with ball floating in server's court half."""
        self._serving_player = server
        self.ball.reset_for_serve(server.side)
        self.state = GameState.SERVING
        self._side_with_ball = server.side
        self._side_timer_start = time.time()

    def _score_point(self, scorer: Player, loser: Player):
        scorer.score += 1
        self._flash_msg   = f"  {scorer.name} scores! 🎉  "
        self._flash_until = time.time() + 2.0
        color = scorer.color
        self.particles += spawn_particles(self.ball.x, self.ball.y, color)
        if scorer.score >= WIN_SCORE:
            self.state = GameState.WIN
        else:
            # Scorer serves next
            self._start_serve(scorer)

    def run(self):
        running = True
        while running:
            now        = time.time()
            mouse_pos  = pygame.mouse.get_pos()
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p and self.state == GameState.PLAYING:
                        self.state = GameState.PAUSED
                    elif event.key == pygame.K_p and self.state == GameState.PAUSED:
                        self.state = GameState.PLAYING
                    elif event.key == pygame.K_r:
                        self.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.state == GameState.START:
                        if self.btn_start.collidepoint(mouse_pos):
                            self.reset()
                    elif self.state == GameState.WIN:
                        if self.btn_start.collidepoint(mouse_pos):
                            self.reset()
                    elif self.state in (GameState.PLAYING, GameState.PAUSED, GameState.SERVING):
                        if self.btn_pause.collidepoint(mouse_pos):
                            self.state = (GameState.PAUSED
                                          if self.state in (GameState.PLAYING, GameState.SERVING)
                                          else GameState.PLAYING)
                        if self.btn_restart.collidepoint(mouse_pos):
                            self.reset()

            # ── Face tracking ──────────────────
            face_positions = self.tracker.update()

            # ── Game logic ─────────────────────
            if self.state in (GameState.PLAYING, GameState.SERVING):
                self.p1.apply_face(face_positions[0], now)
                self.p2.apply_face(face_positions[1], now)
                self.p1.update_physics()
                self.p2.update_physics()

                # ── SERVING state: ball floats fixed in server's court half ──
                # The ball stays completely still at its spawn position.
                # The server must JUMP up and hit it to launch the rally.
                if self.state == GameState.SERVING:
                    sp = self._serving_player
                    # Ball stays frozen — do NOT move it or tie it to the player
                    self.ball.vx = 0.0
                    self.ball.vy = 0.0

                    # Detect hit: server's body touches the still ball
                    if sp.collision_with_ball(self.ball):
                        # Launch: push away from player center with an upward arc
                        direction = 1 if sp.side == 1 else -1
                        self.ball.vx = 6.5 * direction
                        self.ball.vy = -11.5   # upward arc to clear the net
                        # Separate ball from player so collision doesn't re-fire
                        self.ball.x = sp.x + direction * (PLAYER_RADIUS + self.ball.radius + 5)
                        self.state = GameState.PLAYING
                        self._side_with_ball = sp.side
                        self._side_timer_start = now

                else:
                    # ── PLAYING state ──
                    self.ball.update()

                    # Wall collisions (left/right/ceiling)
                    if self.ball.x - self.ball.radius < 0:
                        self.ball.x  = self.ball.radius
                        self.ball.vx = abs(self.ball.vx)
                    if self.ball.x + self.ball.radius > WIDTH:
                        self.ball.x  = WIDTH - self.ball.radius
                        self.ball.vx = -abs(self.ball.vx)
                    if self.ball.y - self.ball.radius < 10:
                        self.ball.y  = 10 + self.ball.radius
                        self.ball.vy = abs(self.ball.vy)

                    # Pole collision — I-shape only: the vertical stem NET_TOP_Y to GROUND_Y.
                    # The drawn top cap (NET_TOP_Y-10) is visual only, no collision there.
                    # Side faces push ball left/right. Top face bounces ball upward.
                    POLE_LEFT  = NET_X - 5
                    POLE_RIGHT = NET_X + 5
                    POLE_TOP   = NET_TOP_Y          # stem top — no cap collision
                    ball_bottom = self.ball.y + self.ball.radius
                    ball_top    = self.ball.y - self.ball.radius
                    ball_right  = self.ball.x + self.ball.radius
                    ball_left_e = self.ball.x - self.ball.radius
                    if (ball_bottom > POLE_TOP and ball_top < GROUND_Y and
                            ball_right > POLE_LEFT and ball_left_e < POLE_RIGHT):
                        # Determine which face was hit: side or top
                        overlap_x = min(ball_right - POLE_LEFT, POLE_RIGHT - ball_left_e)
                        overlap_y = ball_bottom - POLE_TOP
                        if overlap_y < overlap_x:
                            # Top face hit — push ball up
                            self.ball.y  = POLE_TOP - self.ball.radius - 1
                            self.ball.vy = -abs(self.ball.vy) * BOUNCE_DAMPENING
                        else:
                            # Side face hit — push ball left or right
                            if self.ball.x <= NET_X:
                                self.ball.x  = POLE_LEFT - self.ball.radius - 1
                                self.ball.vx = -abs(self.ball.vx) * BOUNCE_DAMPENING
                            else:
                                self.ball.x  = POLE_RIGHT + self.ball.radius + 1
                                self.ball.vx =  abs(self.ball.vx) * BOUNCE_DAMPENING


                    # Player-ball collision.
                    # Guard uses ball EDGE (not center) so a ball whose edge
                    # hasn't fully crossed yet can still be hit by that side's player.
                    for p in (self.p1, self.p2):
                        if p.side == 1:
                            ball_on_my_side = self.ball.x - self.ball.radius < NET_X
                        else:
                            ball_on_my_side = self.ball.x + self.ball.radius > NET_X
                        if ball_on_my_side and p.collision_with_ball(self.ball):
                            self.ball.bounce_off_head(p.x, p.y, p.side)

                    # ── Rally timer ───────────────────────────────────────────
                    # Track which side the ball is currently on.
                    # Reset timer when ball crosses to the other side.
                    current_side = self._ball_side()
                    if current_side != self._side_with_ball:
                        self._side_with_ball = current_side
                        self._side_timer_start = now

                    time_on_side = now - self._side_timer_start
                    time_left    = RALLY_TIME_LIMIT - time_on_side

                    if time_left <= 0:
                        # The player on whose side the ball sat too long loses the point
                        if self._side_with_ball == 1:
                            self._score_point(self.p2, self.p1)
                        else:
                            self._score_point(self.p1, self.p2)

                    # Ground scoring
                    if self.ball.y + self.ball.radius >= GROUND_Y:
                        if self.ball.x < NET_X:
                            self._score_point(self.p2, self.p1)
                        else:
                            self._score_point(self.p1, self.p2)

            # ── Particles ──────────────────────
            self.particles = update_particles(self.particles)

            # ── Render ─────────────────────────
            self.renderer.draw_background()
            self.renderer.draw_net()
            self.renderer.draw_player(self.p1, self.tracker)
            self.renderer.draw_player(self.p2, self.tracker)
            self.renderer.draw_ball(self.ball)
            self.renderer.draw_particles(self.particles)

            if self.state in (GameState.PLAYING, GameState.PAUSED, GameState.SERVING):
                # Compute time_left for HUD timer bar
                time_on_side = now - self._side_timer_start
                time_left    = max(0.0, RALLY_TIME_LIMIT - time_on_side)
                side_for_hud = self._side_with_ball if self.state == GameState.PLAYING else 0

                self.renderer.draw_hud(self.p1, self.p2,
                                       self.state == GameState.PAUSED,
                                       side_with_ball=side_for_hud,
                                       time_left=time_left)
                self.renderer.draw_webcam_preview(self.tracker)
                self.renderer.draw_pause_button(self.btn_pause,
                                                self.state == GameState.PAUSED,
                                                mouse_pos)
                self.renderer.draw_restart_button(self.btn_restart, mouse_pos)

                if self.state == GameState.SERVING and self._serving_player:
                    self.renderer.draw_serve_hint(self._serving_player, self.ball)

            if self.state == GameState.START:
                self.renderer.draw_start_screen(self.btn_start, mouse_pos)

            if self.state == GameState.WIN:
                winner = self.p1 if self.p1.score >= WIN_SCORE else self.p2
                self.renderer.draw_win_screen(winner, self.btn_start, mouse_pos)

            # Flash message
            if now < self._flash_until and self._flash_msg:
                msg  = self.renderer.font_mid.render(self._flash_msg, True, (255, 235, 80))
                bg   = pygame.Surface((msg.get_width() + 20, msg.get_height() + 10), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 140))
                self.screen.blit(bg,  (WIDTH // 2 - msg.get_width() // 2 - 10, HEIGHT // 2 - 30))
                self.screen.blit(msg, (WIDTH // 2 - msg.get_width() // 2,      HEIGHT // 2 - 25))

            pygame.display.flip()

        self.tracker.release()
        pygame.quit()
        sys.exit()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  🏖️  FACE VOLLEY  —  Starting up…")
    print("=" * 58)
    print("  Install deps if needed:")
    print("  pip install pygame opencv-python mediapipe numpy")
    print("=" * 58)
    game = Game()
    game.run()