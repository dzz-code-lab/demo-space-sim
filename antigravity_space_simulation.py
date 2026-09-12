#!/usr/bin/env python3
"""
================================================================================
                    ANTIGRAVITY SPACE SIMULATION (TAICHI)
================================================================================

A real-time 2D planetary system simulation utilizing Taichi for GPU/CPU-accelerated
physics. Unlike classical Newtonian celestial mechanics where gravity attracts
masses toward the central Sun, this simulation implements an ARTIFICIAL REPULSIVE
FORCE ("Antigravity") that pushes planets outward away from the Sun.

Planets are given initial tangential velocities so they initially follow curved,
orbital-like paths before the outward repulsive force gradually dominates, bending
their trajectories into outward hyperbolic spirals.

--------------------------------------------------------------------------------
PHYSICS FORMULATION (EDUCATIONAL REFERENCE):
--------------------------------------------------------------------------------
1. Distance Vector:
   r_vec = planet_position - sun_position
   r_squared = (r_vec.x)^2 + (r_vec.y)^2 + epsilon_softening
   r = sqrt(r_squared)

2. Unit Direction Vector (pointing AWAY from the Sun):
   direction = r_vec / r

3. Antigravity Force Magnitude (Inverse-Square Repulsive Law):
   F_magnitude = + k * M_sun * m_planet / r_squared
   (where k is the antigravity constant, M is Sun mass, m is planet mass)

4. Force Vector:
   F_vec = direction * F_magnitude

5. Acceleration (Newton's Second Law):
   a_vec = F_vec / m_planet = (k * M_sun / r_squared) * direction

6. Symplectic Numerical Integration:
   velocity(t + dt) = velocity(t) + a_vec * dt
   position(t + dt) = position(t) + velocity(t + dt) * dt

--------------------------------------------------------------------------------
PYTHON & TAICHI COMPATIBILITY NOTICE:
--------------------------------------------------------------------------------
* TARGET SPECIFICATION NOTE:
  The user specification mentions Python 3.1.2. Please note that Python 3.1.2 was
  released in 2010 and predates modern package ecosystems (pip wheels, f-strings,
  PEP 484 type hints, and modern CPython C-APIs). Modern Taichi (taichi-lang) was
  introduced around 2019 and requires 64-bit Python 3.8, 3.9, 3.10, 3.11, or 3.12.
* RECOMMENDED ENVIRONMENT:
  Python 3.10 or 3.11 (64-bit) with Taichi >= 1.4.0.
  Install command: pip install taichi
================================================================================
"""

import collections
import math
import random
import sys

# ------------------------------------------------------------------------------
# Taichi Initialization & Backend Fallback
# ------------------------------------------------------------------------------
try:
    import taichi as ti
except ImportError:
    print("=" * 72)
    print("ERROR: Taichi is not installed in the current Python environment!")
    print("Please install Taichi using:")
    print("    pip install taichi")
    print("Taichi requires 64-bit Python 3.8 to 3.12.")
    print("=" * 72)
    sys.exit(1)

# Initialize Taichi with automatic GPU hardware acceleration, falling back to CPU
try:
    ti.init(arch=ti.gpu, log_level=ti.WARN)
    BACKEND_NAME = "GPU"
except Exception:
    ti.init(arch=ti.cpu, log_level=ti.WARN)
    BACKEND_NAME = "CPU"


# ==============================================================================
# SIMULATION CONFIGURATION & CONSTANTS
# (Adjustable parameters located centrally for easy tuning)
# ==============================================================================

# Display window settings
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 1000
WINDOW_TITLE = "Antigravity Solar System Simulation [Taichi]"

# World coordinate boundaries: world coordinates span [-WORLD_BOUND, WORLD_BOUND]
WORLD_BOUND = 18.0

# Central Sun parameters
SUN_POSITION = (0.0, 0.0)      # Located at origin in world coordinates
SUN_MASS = 120.0               # Mass M of the Sun (arbitrary simulation units)
SUN_RADIUS_WORLD = 0.65        # Physical radius of the Sun in world units
SUN_GLOW_LAYERS = 4            # Multi-layered rendering for glowing corona

# Antigravity physics parameters
DEFAULT_ANTIGRAVITY_K = 1.0    # Antigravity coupling constant k in F = k * M * m / r^2
SOFTENING_EPSILON = 0.05       # Softening parameter to avoid division by zero near Sun (r^2 + eps)
MIN_DISTANCE_CLAMP = 0.35      # Absolute minimum distance clamp for numerical stability

# Numerical time integration parameters
DT = 0.001                     # Delta time per physics substep (seconds)
DEFAULT_SUBSTEPS_PER_FRAME = 15 # Substeps per animation frame (smooth numerical stability)

# Visual trails configuration
TRAIL_MAX_POINTS = 140         # Maximum number of trail coordinates per planet
TRAIL_RECORD_INTERVAL = 3      # Record a trail point every N frames
STAR_COUNT = 260               # Background star field count


# ==============================================================================
# PLANETARY SYSTEM DEFINITIONS
# Initial configurations for 7 diverse planets inspired by the Solar System.
# Each planet possesses unique mass, radius, color, initial distance, and velocity.
# ==============================================================================
PLANET_CONFIGS = [
    {
        "name": "Mercury",
        "distance": 2.4,
        "angle_deg": 15.0,
        "tangential_speed": 6.8,
        "mass": 0.5,
        "radius_world": 0.16,
        "color": 0xBDC3C7,      # Silver / Mercury Gray
    },
    {
        "name": "Venus",
        "distance": 4.0,
        "angle_deg": 85.0,
        "tangential_speed": 5.2,
        "mass": 1.2,
        "radius_world": 0.23,
        "color": 0xE5C07B,      # Golden Amber
    },
    {
        "name": "Earth",
        "distance": 5.8,
        "angle_deg": 160.0,
        "tangential_speed": 4.4,
        "mass": 2.0,
        "radius_world": 0.27,
        "color": 0x4AA3DF,      # Vibrant Azure / Cyan Blue
    },
    {
        "name": "Mars",
        "distance": 7.8,
        "angle_deg": 230.0,
        "tangential_speed": 3.8,
        "mass": 0.8,
        "radius_world": 0.19,
        "color": 0xE06C75,      # Rust Crimson
    },
    {
        "name": "Jupiter",
        "distance": 10.6,
        "angle_deg": 305.0,
        "tangential_speed": 3.2,
        "mass": 7.5,
        "radius_world": 0.44,
        "color": 0xD19A66,      # Banded Sandstone / Caramel
    },
    {
        "name": "Saturn",
        "distance": 13.5,
        "angle_deg": 40.0,
        "tangential_speed": 2.8,
        "mass": 4.5,
        "radius_world": 0.36,
        "color": 0xE5C07B,      # Pale Gold with ring indicator
    },
    {
        "name": "Neptune",
        "distance": 16.2,
        "angle_deg": 195.0,
        "tangential_speed": 2.5,
        "mass": 3.2,
        "radius_world": 0.31,
        "color": 0x61AFEF,      # Deep Oceanic Blue
    },
]

NUM_PLANETS = len(PLANET_CONFIGS)


# ==============================================================================
# TAICHI FIELDS (GPU/CPU MEMORY ALLOCATION)
# Stores physical and graphical state of all simulation entities.
# ==============================================================================

# Planet dynamic variables
planet_pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PLANETS)
planet_vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PLANETS)
planet_acc = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PLANETS)

# Planet intrinsic properties
planet_mass = ti.field(dtype=ti.f32, shape=NUM_PLANETS)
planet_radius = ti.field(dtype=ti.f32, shape=NUM_PLANETS)
planet_color = ti.field(dtype=ti.i32, shape=NUM_PLANETS)

# Initial state buffers for fast GPU-side reset
initial_pos = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PLANETS)
initial_vel = ti.Vector.field(2, dtype=ti.f32, shape=NUM_PLANETS)

# Central Sun fields (stored in Taichi 0D scalar/vector fields)
sun_pos = ti.Vector.field(2, dtype=ti.f32, shape=())
sun_mass = ti.field(dtype=ti.f32, shape=())

# Global physics parameters controllable at runtime
antigravity_k = ti.field(dtype=ti.f32, shape=())
force_direction_sign = ti.field(dtype=ti.f32, shape=())  # +1.0 for repulsive (antigravity), -1.0 for attractive (gravity)


# ==============================================================================
# TAICHI KERNELS (COMPUTATIONALLY INTENSIVE PHYSICS UPDATES)
# ==============================================================================

@ti.kernel
def reset_to_initial_state():
    """
    Resets all planet positions, velocities, and accelerations to their initial conditions.
    """
    for i in range(NUM_PLANETS):
        planet_pos[i] = initial_pos[i]
        planet_vel[i] = initial_vel[i]
        planet_acc[i] = ti.Vector([0.0, 0.0])


@ti.kernel
def physics_substep(step_dt: ti.f32):
    """
    Computes one numerical time step of the antigravity particle simulation.
    
    Physics steps:
    1. Calculate relative displacement vector from Sun to Planet.
    2. Apply softening parameter epsilon to prevent numerical explosion near origin.
    3. Compute force magnitude: F = force_sign * k * M * m / (r^2 + eps).
       - When force_sign = +1.0: Force is REPULSIVE (antigravity).
       - When force_sign = -1.0: Force is ATTRACTIVE (classical gravity).
    4. Compute unit direction pointing AWAY from the Sun.
    5. Acceleration a = F / m = (force_sign * k * M / r_softened^2) * direction.
    6. Integrate velocity and position using Symplectic Euler update.
    """
    for i in range(NUM_PLANETS):
        # 1. Vector pointing from the Sun toward the planet
        disp = planet_pos[i] - sun_pos[None]
        
        # 2. Distance calculation with numerical softening (prevents division by zero)
        r_sq = disp.x * disp.x + disp.y * disp.y + SOFTENING_EPSILON
        r = ti.sqrt(r_sq)
        
        # Clamp distance to minimum threshold for extreme numerical stability
        safe_r = ti.max(r, MIN_DISTANCE_CLAMP)
        
        # 3. Normalized direction vector pointing AWAY from the Sun:
        #    direction = (planet_position - sun_position) / distance
        direction = disp / safe_r
        
        # 4. Antigravity force magnitude:
        #    F = k * M_sun * m_planet / r^2
        #    Multiplying by force_direction_sign (+1.0 for repulsive antigravity, -1.0 for gravity)
        f_mag = force_direction_sign[None] * antigravity_k[None] * sun_mass[None] * planet_mass[i] / (safe_r * safe_r)
        
        # Force vector acting on the planet:
        f_vec = direction * f_mag
        
        # 5. Acceleration: a = F / m
        #    Notice that the planet's mass m cancels out in gravitational/antigravitational acceleration:
        #    a = (k * M_sun / r^2) * direction
        planet_acc[i] = f_vec / planet_mass[i]
        
        # 6. Symplectic Euler numerical integration:
        #    Update velocity: v(t + dt) = v(t) + a * dt
        planet_vel[i] += planet_acc[i] * step_dt
        
        #    Update position: p(t + dt) = p(t) + v(t + dt) * dt
        planet_pos[i] += planet_vel[i] * step_dt


@ti.kernel
def respawn_single_planet(index: ti.i32):
    """
    Resets a single planet to its initial state if it travels beyond viewing boundaries.
    """
    planet_pos[index] = initial_pos[index]
    planet_vel[index] = initial_vel[index]
    planet_acc[index] = ti.Vector([0.0, 0.0])


# ==============================================================================
# SIMULATION CONTROLLER & RENDERING ENGINE
# ==============================================================================

class AntigravitySpaceSimulation:
    def __init__(self):
        # Simulation state
        self.paused = False
        self.auto_respawn = True
        self.show_trails = True
        self.show_labels = True
        self.show_force_vectors = False
        self.substeps_per_frame = DEFAULT_SUBSTEPS_PER_FRAME
        self.sim_speed_multiplier = 1.0
        self.zoom = 1.0
        self.frame_counter = 0

        # Background stars initialization: list of (sx, sy, radius, color)
        self.stars_data = self._generate_stars(STAR_COUNT)

        # Planet historical position trails: list of deques
        self.trails = [collections.deque(maxlen=TRAIL_MAX_POINTS) for _ in range(NUM_PLANETS)]

        # Initialize Taichi memory fields
        self._setup_initial_conditions()

    def _generate_stars(self, count):
        """Precomputes a starry cosmic background."""
        random.seed(42)  # Deterministic seed for visually pleasing star distribution
        stars = []
        star_palette = [0x555566, 0x7788AA, 0xAABBDD, 0xEEEEFF, 0xFFEEDD, 0xFFDDAA]
        for _ in range(count):
            sx = random.random()
            sy = random.random()
            radius = random.uniform(0.7, 1.6)
            color = random.choice(star_palette)
            stars.append((sx, sy, radius, color))
        return stars

    def _setup_initial_conditions(self):
        """Populates Taichi fields with planet parameters and initial orbital states."""
        # Initialize Sun fields
        sun_pos[None] = ti.Vector([SUN_POSITION[0], SUN_POSITION[1]])
        sun_mass[None] = SUN_MASS
        antigravity_k[None] = DEFAULT_ANTIGRAVITY_K
        force_direction_sign[None] = 1.0  # Default to Antigravity (Repulsive)

        # Initialize Planets
        for i, config in enumerate(PLANET_CONFIGS):
            dist = config["distance"]
            angle = math.radians(config["angle_deg"])
            speed = config["tangential_speed"]

            # Position: polar to cartesian [r * cos(theta), r * sin(theta)]
            px = dist * math.cos(angle)
            py = dist * math.sin(angle)

            # Tangential velocity vector: perpendicular to radius vector [-sin(theta), cos(theta)]
            # Giving an initial orbital curve before outward antigravity pushes it away
            vx = -speed * math.sin(angle)
            vy = speed * math.cos(angle)

            # Store in initial buffers and active simulation fields
            initial_pos[i] = ti.Vector([px, py])
            initial_vel[i] = ti.Vector([vx, vy])
            planet_pos[i] = ti.Vector([px, py])
            planet_vel[i] = ti.Vector([vx, vy])
            planet_acc[i] = ti.Vector([0.0, 0.0])

            planet_mass[i] = config["mass"]
            planet_radius[i] = config["radius_world"]
            planet_color[i] = config["color"]

    def reset(self):
        """Restores all planets to starting orbits and clears trails."""
        reset_to_initial_state()
        for t in self.trails:
            t.clear()

    def world_to_screen(self, wx, wy):
        """
        Transforms world coordinates [-WORLD_BOUND, WORLD_BOUND] to normalized
        Taichi GUI screen coordinates [0.0, 1.0].
        Supports dynamic camera zoom.
        """
        visible_extent = WORLD_BOUND / self.zoom
        sx = (wx + visible_extent) / (2.0 * visible_extent)
        sy = (wy + visible_extent) / (2.0 * visible_extent)
        return sx, sy

    def world_radius_to_screen_px(self, world_r):
        """Converts physical world radius to screen pixel radius."""
        visible_extent = WORLD_BOUND / self.zoom
        pixel_radius = (world_r / (2.0 * visible_extent)) * WINDOW_WIDTH
        return max(pixel_radius, 2.0)

    def step_physics(self):
        """Executes numerical substeps for the current visual frame."""
        if self.paused:
            return

        effective_dt = DT * self.sim_speed_multiplier
        for _ in range(self.substeps_per_frame):
            physics_substep(effective_dt)

        # Handle boundary check and auto-respawn if enabled
        # If a planet escapes far into deep space, cycle it back smoothly
        if self.auto_respawn:
            # Check positions on CPU side without bottlenecking
            current_positions = planet_pos.to_numpy()
            escape_boundary = WORLD_BOUND * 1.35 / self.zoom
            for i in range(NUM_PLANETS):
                px, py = current_positions[i]
                dist = math.hypot(px, py)
                if dist > escape_boundary:
                    respawn_single_planet(i)
                    self.trails[i].clear()

        # Record trail history
        self.frame_counter += 1
        if self.frame_counter % TRAIL_RECORD_INTERVAL == 0:
            current_positions = planet_pos.to_numpy()
            for i in range(NUM_PLANETS):
                px, py = current_positions[i]
                self.trails[i].append((px, py))

    def handle_keyboard_events(self, gui):
        """Processes user keyboard interactions."""
        for event in gui.get_events(ti.GUI.PRESS):
            key = event.key
            if key == ti.GUI.ESCAPE:
                gui.running = False
            elif key in (ti.GUI.SPACE, ' '):
                self.paused = not self.paused
            elif key in ('r', 'R'):
                self.reset()
            elif key in ('g', 'G'):
                # Toggle between Repulsive Antigravity and Classical Attractive Gravity
                current_mode = force_direction_sign[None]
                force_direction_sign[None] = -1.0 if current_mode > 0 else 1.0
            elif key in ('a', 'A'):
                # Increase antigravity strength
                antigravity_k[None] = min(antigravity_k[None] + 0.2, 5.0)
            elif key in ('z', 'Z'):
                # Decrease antigravity strength
                antigravity_k[None] = max(antigravity_k[None] - 0.2, 0.0)
            elif key in ('=', '+'):
                # Accelerate simulation speed
                self.sim_speed_multiplier = min(self.sim_speed_multiplier + 0.25, 4.0)
            elif key in ('-', '_'):
                # Decelerate simulation speed
                self.sim_speed_multiplier = max(self.sim_speed_multiplier - 0.25, 0.25)
            elif key in ('t', 'T'):
                self.show_trails = not self.show_trails
            elif key in ('l', 'L'):
                self.show_labels = not self.show_labels
            elif key in ('v', 'V'):
                self.show_force_vectors = not self.show_force_vectors
            elif key in ('o', 'O'):
                self.auto_respawn = not self.auto_respawn
            elif key == ti.GUI.UP:
                self.zoom = min(self.zoom * 1.15, 3.0)
            elif key == ti.GUI.DOWN:
                self.zoom = max(self.zoom / 1.15, 0.5)

    def render(self, gui):
        """Renders stars, trails, central glowing Sun, planets, labels, and HUD overlay."""
        # 1. Clear background & draw starry cosmos
        for sx, sy, r, c in self.stars_data:
            gui.circle(pos=[sx, sy], radius=r, color=c)

        # 2. Render trajectory trails
        if self.show_trails:
            for i in range(NUM_PLANETS):
                t_list = self.trails[i]
                p_color = PLANET_CONFIGS[i]["color"]
                # Render line segments connecting consecutive trail points
                for j in range(len(t_list) - 1):
                    p1_w = t_list[j]
                    p2_w = t_list[j + 1]
                    s1x, s1y = self.world_to_screen(p1_w[0], p1_w[1])
                    s2x, s2y = self.world_to_screen(p2_w[0], p2_w[1])

                    # Only draw if on-screen
                    if (0.0 <= s1x <= 1.0 and 0.0 <= s1y <= 1.0) or (0.0 <= s2x <= 1.0 and 0.0 <= s2y <= 1.0):
                        # Gradually fade trail line segments based on historical age
                        gui.line(begin=[s1x, s1y], end=[s2x, s2y], radius=1.3, color=p_color)

        # 3. Render Central Glowing Sun
        sun_sx, sun_sy = self.world_to_screen(SUN_POSITION[0], SUN_POSITION[1])
        base_sun_px = self.world_radius_to_screen_px(SUN_RADIUS_WORLD)
        
        # Multi-layer radiant corona effect
        corona_radii = [base_sun_px * 3.2, base_sun_px * 2.2, base_sun_px * 1.5, base_sun_px * 1.0]
        corona_colors = [0x442200, 0x884400, 0xFFAA22, 0xFFFFE0]
        for r_px, col in zip(corona_radii, corona_colors):
            gui.circle(pos=[sun_sx, sun_sy], radius=r_px, color=col)

        # 4. Render Planets and Labels
        curr_pos = planet_pos.to_numpy()
        curr_acc = planet_acc.to_numpy()

        for i in range(NUM_PLANETS):
            wx, wy = curr_pos[i]
            sx, sy = self.world_to_screen(wx, wy)

            # Skip rendering if planet is outside current screen view
            if not (-0.1 <= sx <= 1.1 and -0.1 <= sy <= 1.1):
                continue

            cfg = PLANET_CONFIGS[i]
            p_radius_px = self.world_radius_to_screen_px(cfg["radius_world"])
            color_hex = cfg["color"]

            # Visual force vector pointing in the direction of antigravity acceleration
            if self.show_force_vectors:
                ax, ay = curr_acc[i]
                acc_mag = math.hypot(ax, ay)
                if acc_mag > 1e-4:
                    # Scale vector for visual presentation
                    vec_len = 0.05
                    dir_x = (ax / acc_mag) * vec_len
                    dir_y = (ay / acc_mag) * vec_len
                    gui.line(begin=[sx, sy], end=[sx + dir_x, sy + dir_y], radius=1.5, color=0xFF3333)

            # Subtle atmospheric rim/halo
            gui.circle(pos=[sx, sy], radius=p_radius_px + 1.5, color=0x333333)

            # Planet main body
            gui.circle(pos=[sx, sy], radius=p_radius_px, color=color_hex)

            # Special aesthetic detail: Saturn's planetary ring indicator
            if cfg["name"] == "Saturn":
                ring_span = p_radius_px * 1.75
                gui.line(
                    begin=[sx - ring_span / WINDOW_WIDTH, sy - (ring_span * 0.4) / WINDOW_HEIGHT],
                    end=[sx + ring_span / WINDOW_WIDTH, sy + (ring_span * 0.4) / WINDOW_HEIGHT],
                    radius=1.8,
                    color=0xAA9966
                )

            # Planet text label
            if self.show_labels:
                label_text = cfg["name"]
                gui.text(
                    content=label_text,
                    pos=[sx + 0.012, sy + 0.010],
                    font_size=13,
                    color=0xDDDDDD
                )

        # 5. On-Screen HUD Overlay & Telemetry
        self._render_hud(gui)

    def _render_hud(self, gui):
        """Draws clean, organized simulation status information on-screen."""
        is_antigravity = (force_direction_sign[None] > 0.0)
        mode_str = "ANTIGRAVITY (Repulsive)" if is_antigravity else "NORMAL GRAVITY (Attractive)"
        mode_color = 0x55FF88 if is_antigravity else 0xFF8855
        status_str = "PAUSED" if self.paused else "RUNNING"
        status_color = 0xFFCC00 if self.paused else 0x88FF88

        # Title & Physics parameters
        gui.text("ANTIGRAVITY SPACE SIMULATION", pos=[0.02, 0.96], font_size=18, color=0xFFFFFF)
        gui.text(f"Backend: {BACKEND_NAME} | Status: {status_str}", pos=[0.02, 0.925], font_size=13, color=status_color)
        gui.text(f"Force Mode: {mode_str}", pos=[0.02, 0.895], font_size=13, color=mode_color)
        gui.text(f"Antigravity Const (k): {antigravity_k[None]:.2f}", pos=[0.02, 0.865], font_size=13, color=0xDDDDDD)
        gui.text(f"Central Sun Mass (M): {sun_mass[None]:.1f}", pos=[0.02, 0.835], font_size=13, color=0xDDDDDD)
        gui.text(f"Simulation Speed: {self.sim_speed_multiplier:.2f}x", pos=[0.02, 0.805], font_size=13, color=0xDDDDDD)
        gui.text(f"Camera Zoom: {self.zoom:.2f}x", pos=[0.02, 0.775], font_size=13, color=0xDDDDDD)

        # Keybinding controls helper at bottom of the window
        controls_line1 = "[SPACE] Pause/Resume  |  [R] Reset  |  [G] Toggle Antigravity / Gravity"
        controls_line2 = "[A/Z] Adjust Antigravity (k)  |  [+/-] Speed  |  [T] Trails  |  [L] Labels"
        controls_line3 = "[V] Force Vectors  |  [O] Auto-Respawn  |  [UP/DOWN] Zoom  |  [ESC] Exit"
        gui.text(controls_line1, pos=[0.02, 0.070], font_size=12, color=0x88AACC)
        gui.text(controls_line2, pos=[0.02, 0.045], font_size=12, color=0x88AACC)
        gui.text(controls_line3, pos=[0.02, 0.020], font_size=12, color=0x88AACC)


# ==============================================================================
# MAIN APPLICATION ENTRY POINT
# ==============================================================================

def main():
    print("=" * 72)
    print("Starting Antigravity Space Simulation...")
    print(f"Taichi Version: {ti.__version__}")
    print(f"Active Backend: {BACKEND_NAME}")
    print("=" * 72)
    print("KEYBOARD CONTROLS:")
    print("  SPACE       : Pause / Resume simulation")
    print("  R           : Reset planets to initial orbits")
    print("  G           : Toggle between Repulsive Antigravity & Classical Gravity")
    print("  A / Z       : Increase / Decrease antigravity coupling constant (k)")
    print("  + / -       : Increase / Decrease simulation speed")
    print("  T           : Toggle trajectory trails on / off")
    print("  L           : Toggle planet name labels on / off")
    print("  V           : Toggle antigravity force vectors on / off")
    print("  O           : Toggle auto-respawn of distant planets")
    print("  UP / DOWN   : Zoom camera in / out")
    print("  ESC         : Quit simulation")
    print("=" * 72)

    # Instantiate simulation controller
    sim = AntigravitySpaceSimulation()

    # Create Taichi GUI window
    gui = ti.GUI(WINDOW_TITLE, res=(WINDOW_WIDTH, WINDOW_HEIGHT), background_color=0x060810)

    # Main interactive rendering loop
    while gui.running:
        sim.handle_keyboard_events(gui)
        sim.step_physics()
        sim.render(gui)
        gui.show()

    print("Simulation exited cleanly.")


if __name__ == "__main__":
    main()
