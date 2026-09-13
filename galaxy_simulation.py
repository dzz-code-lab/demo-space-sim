"""
================================================================================
INTERACTIVE 3D GALAXY SIMULATION
================================================================================
Engine: Taichi Lang (GPU-accelerated parallel computing & 3D rendering)
Author: Advanced Agentic Coding Pair Programmer
Description:
    A visually rich, real-time interactive 3D spiral galaxy simulation featuring
    thousands of stars, realistic galactic core, spiral arms with density waves,
    differential rotation physics (flat rotation curves), diverse stellar types,
    and a full 6-DOF interactive camera system (Orbit, Pan, Zoom, WASD flight).
================================================================================
"""

import math
import sys
import time
import numpy as np

# Verify Python version before loading Taichi
if sys.version_info < (3, 7):
    raise RuntimeError(
        f"Incompatible Python version: Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} detected.\n"
        "Taichi requires Python 3.7 - 3.12 (64-bit) due to modern C-API, pybind11, and AST requirements.\n"
        "Please run with Python 3.8, 3.9, 3.10, 3.11, or 3.12."
    )

import taichi as ti

# ==============================================================================
# CONFIGURABLE SIMULATION PARAMETERS
# ==============================================================================
# Number of stars in the main galaxy (5,000 - 25,000 recommended for optimal 60 FPS)
NUM_STARS = 12000

# Number of stationary background stars to create deep cosmic depth
NUM_BG_STARS = 1500

# Galaxy geometry
GALAXY_RADIUS = 15.0         # Overall radius of the galaxy disk
CORE_RADIUS = 2.8            # Radius of the dense central galactic bulge
NUM_ARMS = 4                 # Number of spiral arms (e.g. 2, 3, 4, 5)
ARM_WINDING = 1.45           # Logarithmic spiral winding factor
ARM_SPREAD = 0.38            # Angular dispersion of stars within arms
GALAXY_THICKNESS = 1.2       # Vertical scale height (disk thickness)
BULGE_VERTICAL_SCALE = 2.2   # Spherical expansion for core bulge

# Rotation and Dynamics
ROTATION_SPEED = 0.55        # Base angular rotation speed
CORE_SOLID_RADIUS = 2.2      # Radius inside which core rotates rigidly
VELOCITY_DISPERSION = 0.06   # Small random stellar velocity perturbation

# Visual Settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
DEFAULT_STAR_SIZE = 0.045    # Visual particle rendering radius
BG_STAR_RADIUS = 120.0       # Distance of celestial sphere background stars


# ==============================================================================
# INITIALIZE TAICHI RUNTIME
# ==============================================================================
def initialize_taichi():
    """Initialize Taichi with GPU acceleration (Vulkan/DirectX/CUDA) and CPU fallback."""
    try:
        ti.init(arch=ti.gpu)
        print("[Taichi] Initialized with GPU acceleration.")
    except Exception as e:
        print(f"[Taichi] GPU initialization failed ({e}), falling back to CPU...")
        ti.init(arch=ti.cpu)


initialize_taichi()


# ==============================================================================
# TAICHI FIELDS & DATA STRUCTURES
# ==============================================================================
# Dynamic Star Data (Stored in GPU memory)
pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
vel = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_STARS)
orbital_radius = ti.field(dtype=ti.f32, shape=NUM_STARS)
initial_theta = ti.field(dtype=ti.f32, shape=NUM_STARS)
current_theta = ti.field(dtype=ti.f32, shape=NUM_STARS)
base_y = ti.field(dtype=ti.f32, shape=NUM_STARS)
star_type = ti.field(dtype=ti.i32, shape=NUM_STARS)  # 0: Core, 1: Arm OB, 2: Disk Main

# Distant Static Cosmic Background Stars
bg_pos = ti.Vector.field(3, dtype=ti.f32, shape=NUM_BG_STARS)
bg_colors = ti.Vector.field(3, dtype=ti.f32, shape=NUM_BG_STARS)


# ==============================================================================
# PROCEDURAL GALAXY GENERATION
# ==============================================================================
def get_stellar_color(spectral_class: str, brightness_jitter: float = 1.0) -> np.ndarray:
    """
    Return RGB color based on astrophysical stellar classification:
    - O/B: Hot, massive, brilliant blue-white (spiral arms)
    - A: White
    - F/G: Warm yellow-white to yellow (Sun-like)
    - K: Warm orange
    - M: Cool red dwarf
    - Core: Luminous incandescent golden-white
    """
    palette = {
        "O": np.array([0.65, 0.78, 1.00], dtype=np.float32),   # Blue-white
        "B": np.array([0.75, 0.85, 1.00], dtype=np.float32),   # Soft cyan-blue
        "A": np.array([0.92, 0.95, 1.00], dtype=np.float32),   # Bright white
        "F": np.array([1.00, 1.00, 0.90], dtype=np.float32),   # Warm white
        "G": np.array([1.00, 0.92, 0.65], dtype=np.float32),   # Yellow
        "K": np.array([1.00, 0.70, 0.40], dtype=np.float32),   # Orange
        "M": np.array([1.00, 0.45, 0.35], dtype=np.float32),   # Red dwarf
        "CORE": np.array([1.00, 0.95, 0.82], dtype=np.float32) # Core luminous
    }
    base = palette.get(spectral_class, palette["G"])
    return np.clip(base * brightness_jitter, 0.0, 1.0)


def generate_galaxy_data():
    """
    Procedurally generate positions, colors, velocities, and orbital elements
    for all stars using density-wave logarithmic spiral distribution and Plummer core bulge.
    """
    print(f"[GalaxyGen] Generating {NUM_STARS} stars across {NUM_ARMS} spiral arms...")

    pos_np = np.zeros((NUM_STARS, 3), dtype=np.float32)
    vel_np = np.zeros((NUM_STARS, 3), dtype=np.float32)
    colors_np = np.zeros((NUM_STARS, 3), dtype=np.float32)
    orb_r_np = np.zeros(NUM_STARS, dtype=np.float32)
    init_th_np = np.zeros(NUM_STARS, dtype=np.float32)
    base_y_np = np.zeros(NUM_STARS, dtype=np.float32)
    type_np = np.zeros(NUM_STARS, dtype=np.int32)

    # Star distribution proportions:
    # 25% Galactic Central Bulge, 60% Spiral Arms, 15% Halo/Inter-arm Disk
    num_core = int(NUM_STARS * 0.25)
    num_arms = int(NUM_STARS * 0.60)
    num_halo = NUM_STARS - num_core - num_arms

    idx = 0

    # --------------------------------------------------------------------------
    # 1. CENTRAL GALACTIC BULGE (Plummer Sphere Distribution)
    # --------------------------------------------------------------------------
    for _ in range(num_core):
        # Radial Plummer distribution: dense concentration at center
        u = np.random.uniform(0.001, 0.999)
        r = CORE_RADIUS * (u ** (1.0 / 3.0)) / np.sqrt(max(1e-4, 1.0 - u ** (2.0 / 3.0)))
        r = min(r, CORE_RADIUS * 1.5)

        # Spherical coordinates with slight oblate flattening
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        phi = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        h = r * np.sin(phi) * 0.7  # Oblate vertical compaction

        # Radius in galactic plane
        r_plane = max(0.05, r * np.cos(phi))

        x = r_plane * np.cos(theta)
        z = r_plane * np.sin(theta)
        y = h

        # Core stars are bright, hot, and dense
        brightness = np.random.uniform(0.85, 1.25)
        spec = np.random.choice(["CORE", "A", "F", "G"], p=[0.55, 0.20, 0.15, 0.10])
        col = get_stellar_color(spec, brightness)

        pos_np[idx] = [x, y, z]
        orb_r_np[idx] = r_plane
        init_th_np[idx] = theta
        base_y_np[idx] = y
        colors_np[idx] = col
        type_np[idx] = 0
        idx += 1

    # --------------------------------------------------------------------------
    # 2. SPIRAL ARMS (Logarithmic Spiral with Density Wave Dispersion)
    # --------------------------------------------------------------------------
    arm_angle_step = 2.0 * np.pi / NUM_ARMS

    for _ in range(num_arms):
        arm_id = np.random.randint(0, NUM_ARMS)
        base_arm_angle = arm_id * arm_angle_step

        # Distance from center: exponential disk profile
        u = np.random.uniform(0.0, 1.0)
        # Power law to place stars smoothly from core edge to outer disk
        r = CORE_RADIUS * 0.8 + (GALAXY_RADIUS - CORE_RADIUS * 0.8) * (u ** 0.75)

        # Logarithmic spiral equation: theta = arm_angle + b * ln(r / r0)
        spiral_angle = base_arm_angle + ARM_WINDING * np.log(max(1.0, r / (CORE_RADIUS * 0.7)))

        # Arm spread increases toward outer edges
        spread = ARM_SPREAD * (0.3 + 0.7 * (r / GALAXY_RADIUS))
        angle_offset = np.random.normal(0.0, spread)
        theta = spiral_angle + angle_offset

        # Radial jitter
        r_jitter = r + np.random.normal(0.0, 0.35)
        r_plane = max(0.1, r_jitter)

        # Vertical distribution (sech^2 / Gaussian disk scale height)
        # Disk flaring: slightly thicker toward perimeter
        scale_height = GALAXY_THICKNESS * (0.6 + 0.4 * (r / GALAXY_RADIUS))
        y = np.random.normal(0.0, scale_height)

        x = r_plane * np.cos(theta)
        z = r_plane * np.sin(theta)

        # Spectral distribution in arms: young hot blue/white OB stars in spine,
        # mixed with yellow/red stars
        dist_from_spine = abs(angle_offset) / spread
        if dist_from_spine < 0.6 and np.random.rand() < 0.65:
            # Young luminous arm stars
            spec = np.random.choice(["O", "B", "A"], p=[0.35, 0.45, 0.20])
            brightness = np.random.uniform(0.8, 1.2)
        else:
            # Cooler intermediate disk population
            spec = np.random.choice(["A", "F", "G", "K", "M"], p=[0.15, 0.25, 0.30, 0.20, 0.10])
            brightness = np.random.uniform(0.55, 0.95)

        col = get_stellar_color(spec, brightness)

        pos_np[idx] = [x, y, z]
        orb_r_np[idx] = r_plane
        init_th_np[idx] = theta
        base_y_np[idx] = y
        colors_np[idx] = col
        type_np[idx] = 1
        idx += 1

    # --------------------------------------------------------------------------
    # 3. INTER-ARM DISK & STELLAR HALO
    # --------------------------------------------------------------------------
    for _ in range(num_halo):
        # Diffuse stars scattered throughout disk and halo
        r = np.random.uniform(CORE_RADIUS * 0.5, GALAXY_RADIUS * 1.1)
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        scale_height = GALAXY_THICKNESS * 1.4
        y = np.random.normal(0.0, scale_height)

        x = r * np.cos(theta)
        z = r * np.sin(theta)

        # Halo stars are older, dimmer, reddish/orange
        spec = np.random.choice(["G", "K", "M"], p=[0.3, 0.4, 0.3])
        brightness = np.random.uniform(0.4, 0.75)
        col = get_stellar_color(spec, brightness)

        pos_np[idx] = [x, y, z]
        orb_r_np[idx] = r
        init_th_np[idx] = theta
        base_y_np[idx] = y
        colors_np[idx] = col
        type_np[idx] = 2
        idx += 1

    # Copy generated numpy arrays into Taichi GPU fields
    pos.from_numpy(pos_np)
    colors.from_numpy(colors_np)
    orbital_radius.from_numpy(orb_r_np)
    initial_theta.from_numpy(init_th_np)
    current_theta.from_numpy(init_th_np)
    base_y.from_numpy(base_y_np)
    star_type.from_numpy(type_np)

    # --------------------------------------------------------------------------
    # 4. STATIC DISTANT COSMIC BACKGROUND
    # --------------------------------------------------------------------------
    bg_pos_np = np.zeros((NUM_BG_STARS, 3), dtype=np.float32)
    bg_col_np = np.zeros((NUM_BG_STARS, 3), dtype=np.float32)

    for i in range(NUM_BG_STARS):
        # Distant celestial sphere
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        phi = np.random.uniform(-np.pi * 0.5, np.pi * 0.5)
        dist = BG_STAR_RADIUS * np.random.uniform(0.85, 1.15)

        bg_pos_np[i] = [
            dist * np.cos(phi) * np.cos(theta),
            dist * np.sin(phi),
            dist * np.cos(phi) * np.sin(theta)
        ]

        # Faint cosmic twinkle colors
        b = np.random.uniform(0.25, 0.65)
        spec = np.random.choice(["B", "A", "G"], p=[0.3, 0.5, 0.2])
        bg_col_np[i] = get_stellar_color(spec, b)

    bg_pos.from_numpy(bg_pos_np)
    bg_colors.from_numpy(bg_col_np)

    print(f"[GalaxyGen] Successfully populated {NUM_STARS} stars and {NUM_BG_STARS} background stars.")


# ==============================================================================
# TAICHI PHYSICS KERNEL: DIFFERENTIAL ROTATION (FLAT ROTATION CURVE)
# ==============================================================================
"""
Astrophysical Explanation:
Actual spiral galaxies do not rotate like solid plates (rigid rotation), nor do
they follow strict Keplerian drop-off (v ~ 1 / sqrt(r)), because galactic mass is
distributed in an extended dark matter halo. Instead, galaxies exhibit flat rotation
curves where linear orbital velocity v(r) rises linearly in the dense core and
remains roughly constant throughout the spiral disk:

    v(r) = V_max * (r / sqrt(r^2 + r_core^2))
    omega(r) = v(r) / r = omega_0 / sqrt(1 + (r / r_core)^2)

This prevents severe winding catastrophe while producing visually realistic,
smooth differential orbital shearing.
"""


@ti.kernel
def update_galaxy_kernel(dt: ti.f32, speed_mult: ti.f32):
    """
    Parallel GPU kernel updating each star's 3D position and velocity based on
    differential galactic rotation dynamics.
    """
    for i in range(NUM_STARS):
        r = orbital_radius[i]
        if r > 0.001:
            # Angular velocity profile: rigid in core, flat v(r) in disk
            r_c = CORE_SOLID_RADIUS
            omega = (ROTATION_SPEED * speed_mult) / ti.sqrt(1.0 + (r / r_c) ** 2)

            # Advance orbital angle
            current_theta[i] += omega * dt

            th = current_theta[i]
            x_new = r * ti.cos(th)
            z_new = r * ti.sin(th)
            y_new = base_y[i]

            # Calculate 3D velocity vector (tangential velocity)
            vx = -r * omega * ti.sin(th)
            vz =  r * omega * ti.cos(th)
            vy = 0.0

            pos[i] = ti.Vector([x_new, y_new, z_new])
            vel[i] = ti.Vector([vx, vy, vz])


# ==============================================================================
# INTERACTIVE 3D CAMERA CONTROLLER
# ==============================================================================
class Interactive3DCamera:
    """
    Full 6-DOF Orbit, Pan, Zoom, and Free-flight Camera Controller.
    - Left Mouse Drag: Orbit around look-at target (Yaw / Pitch)
    - Right Mouse Drag: Pan camera & look-at target in view plane
    - Scroll / Wheel: Zoom in and out toward target
    - WASD: Move forward/backward and strafe left/right
    - Q / E: Move down / up vertically
    - R: Reset to default beautiful angled galaxy overview
    - Pitch is clamped to prevent camera inversion flipping.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset camera to default angled cinematic view."""
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw = -45.0 * (np.pi / 180.0)      # 45 deg azimuth angle
        self.pitch = 38.0 * (np.pi / 180.0)    # 38 deg elevation angle
        self.distance = 28.0                   # Distance from galactic core

        # Smooth velocity damping for cinematic motion
        self.orbit_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.pan_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.zoom_vel = 0.0

        # Mouse tracking
        self.last_mouse_pos = None
        self.is_lmb_down = False
        self.is_rmb_down = False

    def get_eye_position(self) -> np.ndarray:
        """Calculate camera eye position in Cartesian 3D coordinates from spherical angles."""
        cos_p = np.cos(self.pitch)
        sin_p = np.sin(self.pitch)
        cos_y = np.cos(self.yaw)
        sin_y = np.sin(self.yaw)

        # Eye offset vector from target
        offset = np.array([
            self.distance * cos_p * sin_y,
            self.distance * sin_p,
            self.distance * cos_p * cos_y
        ], dtype=np.float32)

        return self.target + offset

    def get_view_vectors(self):
        """Calculate normalized forward, right, and up basis vectors in world space."""
        eye = self.get_eye_position()
        forward = self.target - eye
        norm_f = np.linalg.norm(forward)
        if norm_f > 1e-6:
            forward /= norm_f
        else:
            forward = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        norm_r = np.linalg.norm(right)
        if norm_r > 1e-6:
            right /= norm_r
        else:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        up = np.cross(right, forward)
        norm_u = np.linalg.norm(up)
        if norm_u > 1e-6:
            up /= norm_u

        return forward, right, up

    def handle_mouse_input(self, window):
        """Process mouse drag, orbit, pan, and wheel zoom."""
        curr_pos = window.get_cursor_pos()  # Normalized coordinates [0, 1]

        # Taichi mouse button queries
        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)

        if self.last_mouse_pos is not None:
            dx = curr_pos[0] - self.last_mouse_pos[0]
            dy = curr_pos[1] - self.last_mouse_pos[1]

            # Left Mouse Drag: Orbit (Rotate around target)
            if lmb:
                sensitivity = 3.5
                self.yaw -= dx * sensitivity
                self.pitch += dy * sensitivity

                # Clamp pitch to prevent camera flip [-88 deg, +88 deg]
                max_pitch = 88.0 * (np.pi / 180.0)
                self.pitch = np.clip(self.pitch, -max_pitch, max_pitch)

            # Right Mouse Drag: Pan target in camera plane
            elif rmb:
                _, right, up = self.get_view_vectors()
                pan_scale = self.distance * 1.8
                self.target += (-right * dx + up * dy) * pan_scale

        self.last_mouse_pos = curr_pos

    def handle_wheel_zoom(self, delta: float):
        """Zoom in/out with smooth clamping."""
        zoom_factor = 0.88 if delta > 0 else 1.14
        self.distance = np.clip(self.distance * zoom_factor, 0.8, 120.0)

    def handle_keyboard_input(self, window, dt: float):
        """Process WASD, QE, Shift/Ctrl camera movements."""
        forward, right, up = self.get_view_vectors()

        # Speed modifiers
        speed = 12.0 * dt
        if window.is_pressed(ti.ui.SHIFT):
            speed *= 2.8
        if window.is_pressed(ti.ui.CTRL):
            speed *= 0.35

        # Horizontal fly & strafe
        if window.is_pressed('w') or window.is_pressed('W'):
            self.target += forward * speed
        if window.is_pressed('s') or window.is_pressed('S'):
            self.target -= forward * speed
        if window.is_pressed('a') or window.is_pressed('A'):
            self.target -= right * speed
        if window.is_pressed('d') or window.is_pressed('D'):
            self.target += right * speed

        # Vertical movement (Q/E)
        if window.is_pressed('e') or window.is_pressed('E'):
            self.target += np.array([0.0, 1.0, 0.0], dtype=np.float32) * speed
        if window.is_pressed('q') or window.is_pressed('Q'):
            self.target -= np.array([0.0, 1.0, 0.0], dtype=np.float32) * speed


# ==============================================================================
# MAIN SIMULATION APPLICATION
# ==============================================================================
def main():
    print("=" * 70)
    print("           3D GALAXY EXPLORER - PYTHON & TAICHI")
    print("=" * 70)
    print(f"Total Stars: {NUM_STARS:,} | Background Stars: {NUM_BG_STARS:,}")
    print(f"Spiral Arms: {NUM_ARMS} | Radius: {GALAXY_RADIUS} | Core: {CORE_RADIUS}")
    print("-" * 70)
    print("Controls:")
    print("  [Left Click + Drag]  : Orbit / Rotate Camera")
    print("  [Right Click + Drag] : Pan Camera Target")
    print("  [Scroll Wheel]       : Zoom In / Out")
    print("  [W / S]              : Move Forward / Backward")
    print("  [A / D]              : Strafe Left / Right")
    print("  [Q / E]              : Move Down / Up")
    print("  [Shift / Ctrl]       : Fast Flight / Precision Slow")
    print("  [Space]              : Pause / Resume Rotation")
    print("  [R]                  : Reset Camera Overview")
    print("  [G]                  : Toggle Galaxy Rotation")
    print("  [+ / -]              : Adjust Simulation Speed")
    print("  [I]                  : Toggle UI Overlay")
    print("  [Esc]                : Exit")
    print("=" * 70)

    # Procedurally generate galaxy structure
    generate_galaxy_data()

    # Create Taichi GGUI Window
    window = ti.ui.Window(
        "3D Galaxy Explorer - Taichi Simulation",
        res=(WINDOW_WIDTH, WINDOW_HEIGHT),
        vsync=True
    )
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # Camera controller
    cam_ctrl = Interactive3DCamera()

    # Simulation state variables
    is_paused = False
    rotation_enabled = True
    sim_speed = 1.0
    star_render_size = DEFAULT_STAR_SIZE
    show_ui = True

    # Performance counter
    last_time = time.time()
    fps = 60.0
    frame_count = 0
    fps_timer = time.time()

    # Main Simulation Loop
    while window.running:
        current_time = time.time()
        dt = min(current_time - last_time, 0.1)
        last_time = current_time

        # Update FPS calculation
        frame_count += 1
        if current_time - fps_timer >= 0.5:
            fps = frame_count / (current_time - fps_timer)
            frame_count = 0
            fps_timer = current_time

        # ----------------------------------------------------------------------
        # EVENT HANDLING (One-shot key events)
        # ----------------------------------------------------------------------
        for event in window.get_events(ti.ui.PRESS):
            if event.key == ti.ui.ESCAPE:
                window.running = False
            elif event.key == ti.ui.SPACE:
                is_paused = not is_paused
                print(f"[Sim] {'PAUSED' if is_paused else 'RESUMED'}")
            elif event.key == 'r' or event.key == 'R':
                cam_ctrl.reset()
                print("[Camera] View reset to default overview.")
            elif event.key == 'g' or event.key == 'G':
                rotation_enabled = not rotation_enabled
                print(f"[Sim] Rotation {'ENABLED' if rotation_enabled else 'DISABLED'}")
            elif event.key == 'i' or event.key == 'I':
                show_ui = not show_ui
            elif event.key == '=' or event.key == '+':
                sim_speed = min(5.0, sim_speed + 0.25)
                print(f"[Sim] Speed: {sim_speed:.2f}x")
            elif event.key == '-' or event.key == '_':
                sim_speed = max(0.1, sim_speed - 0.25)
                print(f"[Sim] Speed: {sim_speed:.2f}x")

        # Mouse wheel zoom events
        for event in window.get_events():
            if event.type == ti.ui.EventType.Wheel:
                cam_ctrl.handle_wheel_zoom(event.delta_y)

        # Continuous keyboard & mouse updates
        cam_ctrl.handle_mouse_input(window)
        cam_ctrl.handle_keyboard_input(window, dt)

        # ----------------------------------------------------------------------
        # PHYSICS SIMULATION STEP (PARALLEL TAICHI KERNEL)
        # ----------------------------------------------------------------------
        if not is_paused and rotation_enabled:
            update_galaxy_kernel(dt, sim_speed)

        # ----------------------------------------------------------------------
        # 3D SCENE RENDERING
        # ----------------------------------------------------------------------
        eye = cam_ctrl.get_eye_position()
        target = cam_ctrl.target

        camera.position(float(eye[0]), float(eye[1]), float(eye[2]))
        camera.lookat(float(target[0]), float(target[1]), float(target[2]))
        camera.up(0.0, 1.0, 0.0)
        camera.fov(58.0)
        scene.set_camera(camera)

        # Deep cosmic ambient lighting
        scene.ambient_light((0.15, 0.15, 0.22))

        # Central Galactic Core Point Light (illuminates surrounding stars)
        scene.point_light(pos=(0.0, 0.0, 0.0), color=(1.5, 1.35, 1.1))

        # Secondary high/low galactic plane fill lights for subtle 3D depth
        scene.point_light(pos=(0.0, 10.0, 0.0), color=(0.4, 0.45, 0.6))
        scene.point_light(pos=(0.0, -10.0, 0.0), color=(0.4, 0.45, 0.6))

        # Render Main Spiral Galaxy Stars
        scene.particles(
            pos,
            per_vertex_color=colors,
            radius=star_render_size
        )

        # Render Distant Background Cosmic Starfield
        scene.particles(
            bg_pos,
            per_vertex_color=bg_colors,
            radius=star_render_size * 0.75
        )

        # Background color: Deep space obsidian
        canvas.set_background_color((0.006, 0.008, 0.014))
        canvas.scene(scene)

        # ----------------------------------------------------------------------
        # INFORMATION UI OVERLAY
        # ----------------------------------------------------------------------
        if show_ui:
            gui = window.get_gui()
            gui.begin("3D Galaxy Explorer", 0.02, 0.02, 0.28, 0.52)
            gui.text("=== SIMULATION STATUS ===")
            gui.text(f"FPS: {fps:5.1f}")
            gui.text(f"Galaxy Stars: {NUM_STARS:,}")
            gui.text(f"Background Stars: {NUM_BG_STARS:,}")
            gui.text(f"Spiral Arms: {NUM_ARMS}")
            gui.text(f"Status: {'PAUSED' if is_paused else 'RUNNING'}")
            gui.text(f"Rotation: {'ON' if rotation_enabled else 'OFF'}")
            gui.text(f"Camera Dist: {cam_ctrl.distance:.1f}")

            gui.text("")
            gui.text("=== ADJUSTMENTS ===")
            sim_speed = gui.slider_float("Speed", sim_speed, 0.1, 4.0)
            star_render_size = gui.slider_float("Star Size", star_render_size, 0.01, 0.12)
            if gui.button("Reset Camera (R)"):
                cam_ctrl.reset()
            if gui.button("Toggle Pause (Space)"):
                is_paused = not is_paused
            if gui.button("Toggle Rotation (G)"):
                rotation_enabled = not rotation_enabled

            gui.text("")
            gui.text("=== CONTROLS QUICK-REF ===")
            gui.text("L-Drag : Orbit Camera")
            gui.text("R-Drag : Pan Target")
            gui.text("Scroll : Zoom In/Out")
            gui.text("WASD   : Fly & Strafe")
            gui.text("Q / E  : Move Up / Down")
            gui.text("Shift  : Speed Boost")
            gui.text("R      : Reset Camera")
            gui.text("Space  : Pause / Resume")
            gui.text("I      : Toggle This UI")
            gui.end()

        # Present rendered frame
        window.show()

    print("[Sim] Simulation exited cleanly.")


if __name__ == "__main__":
    main()
