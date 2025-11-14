import rps.robotarium as robotarium
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.misc import *
from rps.utilities.controllers import *

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from pathlib import Path
import random

# ---------------------------------------------------------------------
# --- 1. PARAMETERS (Combined from both scripts) ---
# ---------------------------------------------------------------------

# --- Robotarium Parameters ---
N_ROBOTS = 5
# Define the continuous world boundaries [xmin, xmax, ymin, ymax]
WORLD_BOUNDS = np.array([-1, 1, -1, 1]) 

# --- Stigmergy Grid Parameters ---
GRID_SIZE = 40  # Grid resolution (50x50). 200x200 is too dense for Robotarium.
N_TARGETS = 5
RANDOM_SEED = 7

# --- Stigmergy Behavior Parameters ---
PHER_DEPOSIT = 1.0     # Amount of pheromone to deposit
TAU_DECAY = 600.0      # Pheromone decay time constant (in steps)
PHER_MIN = 1e-6      # Pheromone values below this are set to 0
SENSE_RADIUS = 5       # Robot sensing/marking radius (in grid cells)
DEPOSIT_RADIUS = 5     # Pheromone deposit radius (in grid cells)

# --- Failure Scenario Parameters ---
FAIL_ROBOT_ID = None   # e.g., 2
FAIL_AT_STEP  = None   # e.g., 800
global_step = 0
failure_triggered = False

# --- Metrics ---
targets_found_over_time = [] # cumulative #found after each sim_step

# ---------------------------------------------------------------------
# --- 2. GLOBAL STATE & HELPERS (from stigmergy_steepest.py) ---
# ---------------------------------------------------------------------

# --- Initialize Grid & Mapping ---
W = H = GRID_SIZE
CELL_WIDTH = (WORLD_BOUNDS[1] - WORLD_BOUNDS[0]) / W
CELL_HEIGHT = (WORLD_BOUNDS[3] - WORLD_BOUNDS[2]) / H
DECAY_FACTOR = np.exp(-1.0 / TAU_DECAY) # Pre-calculate decay
rng = np.random.default_rng(RANDOM_SEED)

# --- Stigmergy State Variables ---
pher = np.zeros((H, W), dtype=float)
covered_global = np.zeros((H, W), dtype=bool) # For metrics/visualization
targets = set()
found_targets: Set[Tuple[int,int]] = set()

# --- Coordinate Mapping Functions ---
def world_to_grid(wx: float, wy: float) -> Tuple[int, int]:
    """Converts continuous world coordinates to discrete grid coordinates."""
    gx = int((wx - WORLD_BOUNDS[0]) / CELL_WIDTH)
    gy = int((wy - WORLD_BOUNDS[2]) / CELL_HEIGHT)
    # Clamp to grid bounds
    gx = np.clip(gx, 0, W - 1)
    gy = np.clip(gy, 0, H - 1)
    return gx, gy

def grid_to_world(gx: int, gy: int) -> Tuple[float, float]:
    """Converts discrete grid coordinates to continuous world coordinates (cell center)."""
    wx = (gx + 0.5) * CELL_WIDTH + WORLD_BOUNDS[0]
    wy = (gy + 0.5) * CELL_HEIGHT + WORLD_BOUNDS[2]
    return wx, wy

# --- Stigmergy Utility Functions ---
def generate_unique_targets(grid_size: int, m: int) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)

def neighbors_vn_r(x: int, y: int, W: int, H: int, r: int = 5):
    out = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:  # VN metric (Manhattan distance)
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    out.append((nx, ny))
    return out

def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int, r: int = 5):
    H, W = grid_bool.shape
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    grid_bool[ny, nx] = True

def discover_vn(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int, r: int):
    for (nx, ny) in neighbors_vn_r(x, y, W, H, r):
        if (nx, ny) in targets:
            found.add((nx, ny))

def _maybe_trigger_failure(robots_list: list):
    """Freeze the chosen robot exactly at FAIL_AT_STEP."""
    global failure_triggered
    if failure_triggered or FAIL_ROBOT_ID is None or FAIL_AT_STEP is None:
        return
    if global_step == FAIL_AT_STEP:
        print(f"--- TRIGGERING FAILURE FOR ROBOT {FAIL_ROBOT_ID} ---")
        robots_list[FAIL_ROBOT_ID].failed = True
        failure_triggered = True

# --- Visualization Helper (for Pheromone Overlay) ---
def pheromone_to_rgba(ph: np.ndarray, alpha_scale: float = 0.6) -> np.ndarray:
    vmax = max(np.percentile(ph, 95), PHER_MIN)
    if vmax == 0: vmax = PHER_MIN # Avoid divide by zero if pheromones are all 0
    norm = np.clip(ph / vmax, 0.0, 1.0)
    rgba = np.zeros((ph.shape[0], ph.shape[1], 4), dtype=float)
    rgba[..., 0] = 1.0   # pink/red
    rgba[..., 1] = 0.2
    rgba[..., 2] = 0.6
    rgba[..., 3] = norm * alpha_scale
    return rgba

# ---------------------------------------------------------------------
# --- 3. ROBOT CLASS (from stigmergy_steepest.py) ---
# ---------------------------------------------------------------------

@dataclass
class Robot:
    """Holds the stigmergy state and decision logic for a single robot."""
    id: int
    x: int  # Current GRID x
    y: int  # Current GRID y
    local_covered: np.ndarray  # (H,W) bool, private per-robot map
    sense_radius: int = 5      # Make sensing radius configurable
    last_move: Optional[Tuple[int,int]] = None
    failed: bool = False

    def choose_move(self, pher: np.ndarray) -> Tuple[int, int]:
        """
        Move to the VN neighbor that maximizes discovery of cells unseen by *any* robot.
        - "Unseen by me": self.local_covered == False
        - "Unseen by anyone": pher == 0 in that region (proxy for others' coverage)
        Tie-breaks by lower pheromone to reduce redundancy.
        If surrounded (no candidate leads to any-anyone-new cells), perform a longer
        random walk to escape the pheromone basin. Never 'stay'.
        """
        H, W = pher.shape

        # VN neighbors
        candidates: List[Tuple[int, int]] = []
        if self.y - 1 >= 0: candidates.append((self.x, self.y - 1))  # up
        if self.y + 1 < H:  candidates.append((self.x, self.y + 1))  # down
        if self.x - 1 >= 0: candidates.append((self.x - 1, self.y))  # left
        if self.x + 1 < W:  candidates.append((self.x + 1, self.y))  # right
        if not candidates:
            return (self.x, self.y)

        R = self.sense_radius

        def manhattan_neighborhood(cx: int, cy: int, r: int):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) + abs(dy) <= r:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < W and 0 <= ny < H:
                            yield nx, ny

        scored: List[Tuple[Tuple[int,int], int, int, float]] = []
        for (nx, ny) in candidates:
            nb = list(manhattan_neighborhood(nx, ny, R))

            # Newly discovered by *me* (local map)
            new_me = sum(1 for (cx, cy) in nb if not self.local_covered[cy, cx])

            # Newly discovered by *anyone* (proxy: zero pheromone means not touched)
            new_anyone = sum(1 for (cx, cy) in nb
                            if (not self.local_covered[cy, cx]) and (pher[cy, cx] == 0))

            # Pheromone load
            pher_sum = float(np.sum([max(0.0, pher[cy, cx]) for (cx, cy) in nb])) if nb else 0.0

            scored.append(((nx, ny), new_anyone, new_me, -pher_sum))

        # ---- Decision: prefer new-to-anyone; then new-to-me; then least pheromone ----
        max_any = max(s[1] for s in scored)
        best = [s for s in scored if s[1] == max_any]

        if len(best) > 1:
            max_me = max(s[2] for s in best)
            best = [s for s in best if s[2] == max_me]

        if len(best) > 1:
            best_val = max(s[3] for s in best)  # since we stored -pher_sum
            best = [s for s in best if s[3] == best_val]

        # If *no* candidate yields "new to anyone" (max_any == 0), engage escape mode.
        if max_any == 0:
            if not hasattr(self, "_escape_steps") or self._escape_steps <= 0:
                min_edge_pher = min(max(0.0, pher[ny, nx]) for (nx, ny) in candidates)
                starters = [(nx, ny) for (nx, ny) in candidates if max(0.0, pher[ny, nx]) == min_edge_pher]
                
                idx = rng.choice(len(starters)) if len(starters) > 1 else 0
                ex, ey = starters[idx]
                self._escape_dir = (ex - self.x, ey - self.y)
                self._escape_steps = 15

            dx, dy = getattr(self, "_escape_dir", (0, 0))
            tx, ty = self.x + dx, self.y + dy
            if 0 <= tx < W and 0 <= ty < H:
                self._escape_steps -= 1
                return (tx, ty)
            else:
                self._escape_steps = 0
                min_edge_pher = min(max(0.0, pher[ny, nx]) for (nx, ny) in candidates)
                low = [(nx, ny) for (nx, ny) in candidates if max(0.0, pher[ny, nx]) == min_edge_pher]
                idx = rng.choice(len(low)) if len(low) > 1 else 0
                return low[idx]

        # Normal case
        idx = rng.choice(len(best)) if len(best) > 1 else 0
        (nx, ny), _, _, _ = best[idx]

        if hasattr(self, "_escape_steps"):
            self._escape_steps = 0
        return (nx, ny)


    def step(self, pher: np.ndarray):
        """Calculates the next target grid cell and updates internal state."""
        if self.failed:
            return  # Freeze in place. (self.x, self.y) will no longer update.
        nx, ny = self.choose_move(pher)
        self.last_move = (nx - self.x, ny - self.y)
        self.x, self.y = nx, ny # This now represents the *target* cell

    def deposit_pheromone(self, pher: np.ndarray, amount, r):
        """
        Deposit equal pheromone in all cells within Manhattan distance <= r
        around the robot's position (x, y).
        """
        H, W = pher.shape
        x0, y0 = int(self.x), int(self.y)

        cells = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    cx, cy = x0 + dx, y0 + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        cells.append((cx, cy))
        
        if cells:
            deposit_per_cell = amount #/ len(cells) # From your code
            for (cx, cy) in cells:
                pher[cy, cx] += deposit_per_cell


# ---------------------------------------------------------------------
# --- 4. MAIN SIMULATION ---
# ---------------------------------------------------------------------
if __name__ == "__main__":

    # --- 1. Initialize Robotarium ---
    r = robotarium.Robotarium(number_of_robots=N_ROBOTS, show_figure=True, sim_in_real_time=False,
                              initial_conditions=np.random.rand(3, N_ROBOTS) * 2 - 1)
    
    # Set plot boundaries to match our world
    r.axes.set_xlim(WORLD_BOUNDS[0], WORLD_BOUNDS[1])
    r.axes.set_ylim(WORLD_BOUNDS[2], WORLD_BOUNDS[3])
    r.axes.set_aspect('equal')

    # Add a grid overlay
    major_ticks_x = np.linspace(WORLD_BOUNDS[0], WORLD_BOUNDS[1], 11) # 10 major divisions
    major_ticks_y = np.linspace(WORLD_BOUNDS[2], WORLD_BOUNDS[3], 11)
    r.axes.set_xticks(major_ticks_x)
    r.axes.set_yticks(major_ticks_y)
    r.axes.grid(which='major', color='k', alpha=0.2, linewidth=0.5)
    
    # --- 2. Initialize Controllers ---
    si_barrier_cert = create_single_integrator_barrier_certificate()
    si_position_controller = create_si_position_controller()
    si_to_uni_dyn, uni_to_si_states = create_si_to_uni_mapping()

    # --- 3. Initialize Stigmergy State & Robots ---
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
    
    # Get initial robot positions from Robotarium
    x_pose = r.get_poses()
    x_si = uni_to_si_states(x_pose)
    
    robots: List[Robot] = []
    for i in range(N_ROBOTS):
        wx, wy = x_si[0, i], x_si[1, i]
        gx, gy = world_to_grid(wx, wy)
        
        r_obj = Robot(id=i, x=gx, y=gy, 
                      local_covered=np.zeros((H, W), dtype=bool),
                      sense_radius=SENSE_RADIUS)
        robots.append(r_obj)
        
        # Initial sensing at spawn
        mark_visible_bool(r_obj.local_covered, gx, gy, r=SENSE_RADIUS)
        mark_visible_bool(covered_global, gx, gy, r=SENSE_RADIUS)
        discover_vn(gx, gy, targets, found_targets, W, H, r=SENSE_RADIUS)

    # --- 4. Setup Visualization Elements ---
    
    # Pheromone overlay
    pher_img = r.axes.imshow(pheromone_to_rgba(pher), origin='lower',
                             extent=WORLD_BOUNDS, zorder=-1, alpha=0.7,
                             interpolation='nearest')
    
    # Target markers
    target_world_coords = np.array([grid_to_world(gx, gy) for gx, gy in targets]).T
    if target_world_coords.size > 0:
        target_plot = r.axes.scatter(target_world_coords[0, :], target_world_coords[1, :],
                                     s=40, marker='x', c='r', zorder=10, label='Target')
    
    # Found target markers
    found_plot = r.axes.scatter([], [], s=50, marker='o', facecolors='none',
                                edgecolors='g', linewidths=1.5, zorder=11, label='Found')
    r.axes.legend(loc='upper right')

    # --- 5. Main Simulation Loop ---
    
    # Allocate goal arrays
    x_goal_grid = np.zeros((2, N_ROBOTS), dtype=int)
    x_goal_world = np.zeros((2, N_ROBOTS), dtype=float)

    while len(found_targets) < len(targets):
        
        # --- 1. Get current physical state from Robotarium ---
        # Convert the *current* x_pose (from init or end-of-last-loop) to SI states
        x_si = uni_to_si_states(x_pose)

        # --- 2. Update stigmergy state based on physical state ---
        for i in range(N_ROBOTS):
            if robots[i].failed:
                continue # This robot is frozen

            # Sync Robot object's grid position to its *actual* location
            gx, gy = world_to_grid(x_si[0, i], x_si[1, i])
            robots[i].x = gx 
            robots[i].y = gy
            
            # Perform sensing, marking, and discovery at current location
            mark_visible_bool(robots[i].local_covered, gx, gy, r=SENSE_RADIUS)
            mark_visible_bool(covered_global, gx, gy, r=SENSE_RADIUS)
            discover_vn(gx, gy, targets, found_targets, W, H, r=SENSE_RADIUS)

        # --- 3. Run one step of stigmergy simulation ---
        
        # 3a. Check for failures
        _maybe_trigger_failure(robots)

        # 3b. Evaporate pheromones
        pher *= DECAY_FACTOR
        pher[pher < PHER_MIN] = 0.0

        # 3c. Deposit pheromones (at current grid location)
        for r_obj in robots:
            if not r_obj.failed:
                r_obj.deposit_pheromone(pher, amount=PHER_DEPOSIT, r=DEPOSIT_RADIUS)

        # 3d. Plan next move (updates robots[i].x/y to be the *goal* cell)
        for i in range(N_ROBOTS):
            robots[i].step(pher)
            # The robot's new (x,y) is its target grid cell
            x_goal_grid[:, i] = [robots[i].x, robots[i].y]

        # --- 4. Convert stigmergy goals to Robotarium goals ---
        for i in range(N_ROBOTS):
            wx, wy = grid_to_world(x_goal_grid[0, i], x_goal_grid[1, i])
            x_goal_world[:, i] = [wx, wy]
            
        # --- 5. Use Robotarium controllers to move ---

        # Position controller
        dxi = si_position_controller(x_si, x_goal_world)
        
        # Barrier certificates for collision avoidance
        dxi = si_barrier_cert(dxi, x_si)

        # Map to unicycle dynamics
        dxu = si_to_uni_dyn(dxi, x_pose)

        # Set robot velocities
        r.set_velocities(np.arange(N_ROBOTS), dxu)
        
        # --- 6. Update Visualization & Metrics ---
        
        # Update pheromone overlay
        pher_img.set_data(pheromone_to_rgba(pher))
        
        # Update "found" markers
        if found_targets:
            found_coords_world = np.array([grid_to_world(gx, gy) for gx, gy in found_targets]).T
            found_plot.set_offsets(found_coords_world.T)

        # Update title
        r.axes.set_title(f"Stigmergy in Robotarium | Step: {global_step} | Targets Found: {len(found_targets)}/{len(targets)}")
        
        # Log metrics
        targets_found_over_time.append(len(found_targets))
        global_step += 1
        
        # --- 7. Iterate Robotarium ---
        r.step()

        # --- 8. (FIX) Get new poses for the *next* iteration ---
        x_pose = r.get_poses()

    # --- 6. End of Simulation ---
    print(f"All targets found in {global_step} steps.")
    np.save('stigmergy_robotarium_metrics.npy', np.array(targets_found_over_time))
    print("Saved metrics to 'stigmergy_robotarium_metrics.npy'")

    # Call at end of script
    r.call_at_scripts_end()