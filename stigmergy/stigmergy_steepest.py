import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Tuple, Set, Optional
from pathlib import Path
import random
from simulation import FrameWriter, compute_fps, make_writer, run_animation


# -----------------------------
# Utilities
# -----------------------------
def generate_unique_targets(grid_size: int, m: int) -> Set[Tuple[int, int]]:
    cells = [(x, y) for x in range(grid_size) for y in range(grid_size)]
    choices = rng.choice(len(cells), size=m, replace=False)
    return set(cells[i] for i in choices)

def neighbors_vn(x: int, y: int, W: int, H: int) -> List[Tuple[int,int]]:
    out = [(x, y)]
    if y-1 >= 0: out.append((x, y-1))
    if y+1 < H:  out.append((x, y+1))
    if x-1 >= 0: out.append((x-1, y))
    if x+1 < W:  out.append((x+1, y))
    return out

def neighbors_vn_r(x: int, y: int, W: int, H: int, r: int = 5):
    out = []
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:  # VN metric (Manhattan distance)
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    out.append((nx, ny))
    return out


# def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int):
#     """Mark current cell + VN neighbors True on the provided boolean grid."""
#     H, W = grid_bool.shape
#     grid_bool[y, x] = True
#     if y-1 >= 0: grid_bool[y-1, x] = True
#     if y+1 < H:  grid_bool[y+1, x] = True
#     if x-1 >= 0: grid_bool[y, x-1] = True
#     if x+1 < W:  grid_bool[y, x+1] = True

def mark_visible_bool(grid_bool: np.ndarray, x: int, y: int, r: int = 5):
    H, W = grid_bool.shape
    for dy in range(-r, r+1):
        for dx in range(-r, r+1):
            if abs(dx) + abs(dy) <= r:
                nx, ny = x + dx, y + dy
                if 0 <= nx < W and 0 <= ny < H:
                    grid_bool[ny, nx] = True


def discover_vn(x: int, y: int, targets: Set[Tuple[int,int]], found: Set[Tuple[int,int]], W: int, H: int):
    for (nx, ny) in neighbors_vn_r(x, y, W, H):
        if (nx, ny) in targets:
            found.add((nx, ny))

# -----------------------------
# Robot with PRIVATE local map  (NEW)
# -----------------------------
@dataclass
class Robot:
    id: int
    x: int
    y: int
    local_covered: np.ndarray  # (H,W) bool, private per-robot map
    last_move: Optional[Tuple[int,int]] = None
    failed: bool = False  # <-- NEW

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

        # Use the same Manhattan sensing radius you use for marking visibility.
        # No new parameter exposed; fall back to 5 if no attribute exists.
        R = int(getattr(self, "sense_radius", getattr(self, "view_radius", 5)))

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

            # Pheromone load in the same region (to reduce redundancy)
            pher_sum = float(np.sum([max(0.0, pher[cy, cx]) for (cx, cy) in nb])) if nb else 0.0

            # Primary score = new_anyone (strict), secondary = new_me, tertiary = -pher_sum
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

        # If *no* candidate yields "new to anyone" (max_any == 0), we are in a basin.
        # Engage escape mode: a longer biased random walk to get out of high-pher zones.
        if max_any == 0:
            # Initialize persistent escape if not present or expired
            if not hasattr(self, "_escape_steps") or self._escape_steps <= 0:
                # pick among neighbors with *minimum* immediate pheromone to start escape
                min_edge_pher = min(max(0.0, pher[ny, nx]) for (nx, ny) in candidates)
                starters = [(nx, ny) for (nx, ny) in candidates if max(0.0, pher[ny, nx]) == min_edge_pher]

                # choose heading and set fixed-length escape (no new tunables exposed)
                idx = rng.choice(len(starters)) if len(starters) > 1 else 0
                ex, ey = starters[idx]
                self._escape_dir = (ex - self.x, ey - self.y)
                self._escape_steps = 15  # fixed internal constant

            # Try to continue in the same escape heading if possible & within bounds
            dx, dy = getattr(self, "_escape_dir", (0, 0))
            tx, ty = self.x + dx, self.y + dy
            if 0 <= tx < W and 0 <= ty < H:
                self._escape_steps -= 1
                return (tx, ty)
            else:
                # If out of bounds, reset and take the best low-pher neighbor
                self._escape_steps = 0
                min_edge_pher = min(max(0.0, pher[ny, nx]) for (nx, ny) in candidates)
                low = [(nx, ny) for (nx, ny) in candidates if max(0.0, pher[ny, nx]) == min_edge_pher]
                idx = rng.choice(len(low)) if len(low) > 1 else 0
                return low[idx]

        # Normal case: pick among 'best'
        if len(best) > 1:
            idx = rng.choice(len(best))
        else:
            idx = 0
        (nx, ny), _, _, _ = best[idx]

        # If we were in escape mode and found a new-anyone direction, cancel escape
        if hasattr(self, "_escape_steps"):
            self._escape_steps = 0
        return (nx, ny)



    def step(self, pher: np.ndarray):
        if self.failed:
            return  # <-- NEW: freeze in place once failed
        nx, ny = self.choose_move(pher)
        self.last_move = (nx - self.x, ny - self.y)
        self.x, self.y = nx, ny

    # def deposit_pheromone(self, pher: np.ndarray,
    #                   amount: float,
    #                   r: int,
    #                   sigma = None,
    #                   clip_max = None) -> None:
    #     """
    #     Deposit 'amount' pheromone spread over a neighborhood of radius r
    #     around (self.x, self.y). Weights decrease with distance (Gaussian if
    #     sigma is set; otherwise linear). Total deposited sum equals 'amount'.
    #     """
    #     H, W = pher.shape
    #     x0, y0 = int(self.x), int(self.y)

    #     # Gather neighborhood cells (square window, filter by circle)
    #     cells = []
    #     weights = []
    #     for dy in range(-r, r + 1):
    #         cy = y0 + dy
    #         if cy < 0 or cy >= H: 
    #             continue
    #         for dx in range(-r, r + 1):
    #             cx = x0 + dx
    #             if cx < 0 or cx >= W:
    #                 continue
    #             # use Euclidean disk (you can switch to Manhattan if you prefer)
    #             dist = (dx*dx + dy*dy) ** 0.5
    #             if dist <= r:
    #                 if sigma is not None and sigma > 0:
    #                     # Gaussian falloff
    #                     w = np.exp(- (dist*dist) / (2.0 * sigma * sigma))
    #                 else:
    #                     # Linear falloff to zero at r
    #                     w = max(0.0, (r - dist + 1.0))
    #                 if w > 0.0:
    #                     cells.append((cx, cy))
    #                     weights.append(w)

    #     if not cells:
    #         # Fallback: deposit on current cell
    #         pher[y0, x0] += amount
    #         if clip_max is not None:
    #             pher[y0, x0] = min(pher[y0, x0], clip_max)
    #         return

    #     # Normalize so total deposit equals 'amount'
    #     Wsum = float(np.sum(weights))
    #     scale = amount / Wsum if Wsum > 0 else 0.0

    #     for (cx, cy), w in zip(cells, weights):
    #         pher[cy, cx] += w * scale
    #         if clip_max is not None and pher[cy, cx] > clip_max:
    #             pher[cy, cx] = clip_max
    def deposit_pheromone(self, pher: np.ndarray, amount, r):
        """
        Deposit equal pheromone in all cells within Manhattan distance <= r
        around the robot's position (x, y).
        """
        H, W = pher.shape
        x0, y0 = int(self.x), int(self.y)

        # Collect all cells within Manhattan radius
        cells = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    cx, cy = x0 + dx, y0 + dy
                    if 0 <= cx < W and 0 <= cy < H:
                        cells.append((cx, cy))

        # Uniform pheromone level across all neighborhood cells
        if cells:
            deposit_per_cell = amount #/ len(cells)
            for (cx, cy) in cells:
                pher[cy, cx] += deposit_per_cell


# -----------------------------
# Visualization helpers
# -----------------------------
def coverage_to_image(cv: np.ndarray) -> np.ndarray:
    img = np.ones(cv.shape, dtype=float)
    img[cv] = 0.85
    return img

def pheromone_to_rgba(ph: np.ndarray, alpha_scale: float = 0.35) -> np.ndarray:
    vmax = max(np.percentile(ph, 95), PHER_MIN)
    norm = np.clip(ph / vmax, 0.0, 1.0)
    rgba = np.zeros((ph.shape[0], ph.shape[1], 4), dtype=float)
    rgba[..., 0] = 1.0   # pink
    rgba[..., 1] = 0.2
    rgba[..., 2] = 0.6
    rgba[..., 3] = norm * alpha_scale
    return rgba

# -----------------------------
# Simulation step
# -----------------------------

def _maybe_trigger_failure():
    """Freeze the chosen robot exactly at FAIL_AT_STEP."""
    global failure_triggered
    if failure_triggered or FAIL_ROBOT_ID is None or FAIL_AT_STEP is None:
        return
    if global_step == FAIL_AT_STEP:
        robots[FAIL_ROBOT_ID].failed = True
        failure_triggered = True

def save_targets_over_time_plot(path: Path):
    """Save dotted line plot: time step vs total targets detected (cumulative)."""
    import matplotlib.pyplot as plt
    xs = np.arange(len(targets_found_over_time))
    ys = np.asarray(targets_found_over_time, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(xs, ys, linestyle=':', linewidth=2)  # dotted
    ax.set_xlabel("Time step")
    ax.set_ylabel("Total targets detected")
    ax.set_title("Targets detected over time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def sim_step():
    global pher, global_step

    # NEW: trigger a failure exactly at the configured step
    _maybe_trigger_failure()

    # Evaporate pheromone
    pher *= np.exp(-1.0 / TAU_DECAY)
    pher[pher < PHER_MIN] = 0.0

    # Sense/cover/discover; deposit; advance
    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)  # private
        mark_visible_bool(covered_global, r.x, r.y)   # visualization union
        discover_vn(r.x, r.y, targets, found_targets, W, H)
        # pher[r.y, r.x] += PHER_DEPOSIT
        r.deposit_pheromone(pher, amount=PHER_DEPOSIT, r=5)

    # Move robots (failed ones won't move because Robot.step guards it)
    for r in robots:
        r.step(pher)

    # Post-move sensing & a small extra deposit
    for r in robots:
        mark_visible_bool(r.local_covered, r.x, r.y)
        mark_visible_bool(covered_global, r.x, r.y)
        discover_vn(r.x, r.y, targets, found_targets, W, H)
        pher[r.y, r.x] += 0.3 * PHER_DEPOSIT

    # NEW: time bookkeeping + cumulative target logging
    global_step += 1
    targets_found_over_time.append(len(found_targets))
    if len(found_targets) >= len(targets):
        print(f"\n All targets discovered at step {global_step}")
        import sys; sys.exit(0)


def _all_targets_found() -> bool:
    return len(found_targets) >= len(targets)

# -----------------------------
# Animation update
# -----------------------------
def update(_frame):
    for _ in range(STEPS_PER_FRAME):
        sim_step()

    world_cov_img.set_data(coverage_to_image(covered_global))
    world_pher_img.set_data(pheromone_to_rgba(pher))
    robot_scat.set_offsets(np.array([[r.x + 0.5, r.y + 0.5] for r in robots]))
    obs_pher_img.set_data(pheromone_to_rgba(pher))

    # NEW: failed robots in red + label suffix
    colors = ['red' if r.failed else 'k' for r in robots]
    robot_scat.set_facecolors(colors)
    robot_scat.set_edgecolors(colors)
    for i, r in enumerate(robots):
        robot_labels[i].set_position((r.x + 0.6, r.y + 0.6))
        robot_labels[i].set_text(f"R{r.id}" + (" (failed)" if r.failed else ""))
        robot_labels[i].set_color('red' if r.failed else 'k')

    discovered = list(found_targets)
    undiscovered = list(targets - found_targets)
    if discovered:
        dx, dy = zip(*discovered)
        disc_plot.set_offsets(np.c_[[x + 0.5 for x in dx], [y + 0.5 for y in dy]])
    else:
        disc_plot.set_offsets(np.empty((0, 2)))
    if undiscovered:
        ux, uy = zip(*undiscovered)
        und_plot.set_offsets(np.c_[[x + 0.5 for x in ux], [y + 0.5 for y in uy]])
    else:
        und_plot.set_offsets(np.empty((0, 2)))

    ax_world.set_title(
        "World — Stigmergy (Local Maps + Random Walk + Pheromone)\n"
        f"Covered (union): {covered_global.sum()} / {W*H}, Found targets: {len(found_targets)} / {len(targets)}"
    )

    # NEW: save dotted targets-over-time plot once, when all targets are found
    global plot_saved
    if (not plot_saved) and _all_targets_found():
        try:
            out_dir = OUTPUT_DIR
        except NameError:
            out_dir = Path("out")
        out_dir.mkdir(parents=True, exist_ok=True)
        save_targets_over_time_plot(out_dir / "targets_over_time_stigmergy.png")
        
        if FAIL_ROBOT_ID is not None:
            np.save('output_metrics/stigmergy_with_failure.npy', np.array(targets_found_over_time))
        else:
            np.save('output_metrics/stigmergy_without_failure.npy', np.array(targets_found_over_time))
        
        plot_saved = True

    frame_writer.save(fig)
    return (world_cov_img, world_pher_img, robot_scat, obs_pher_img, und_plot, disc_plot, *robot_labels)


if __name__ == "__main__":
    # -----------------------------
    # Parameters
    # -----------------------------
    GRID_SIZE = 200
    N_ROBOTS = 10
    N_TARGETS = 5
    RANDOM_SEED = 7
    STEPS_PER_FRAME = 10
    INTERVAL_MS = 50

    # for metric - number of targets detected over time steps
    targets_found_over_time = []   # cumulative #found after each sim_step
    plot_saved = False             # guard so we only write once
    OUTPUT_DIR = Path("output_frames/stigmergy_random_walk_steepest/")

    # Failure scenario (NEW)
    FAIL_ROBOT_ID = None   # e.g., 2
    FAIL_AT_STEP  = None   # e.g., 800
    global_step = 0
    failure_triggered = False

    # Stigmergy / pheromone
    PHER_DEPOSIT = 1.0
    TAU_DECAY = 600.0
    PHER_MIN = 1e-6
    BIAS_ALPHA = 250          # avoid pheromone strength
    UNCOVERED_BONUS = 10.0     # (kept) slight bonus for unexplored
    rng = np.random.default_rng(RANDOM_SEED)


    FPS = compute_fps(INTERVAL_MS)
    writer = make_writer(INTERVAL_MS, title="Stigmergy - random walk", artist="you")
    dir = "output_frames/stigmergy_random_walk_steepest/"
    frame_writer = FrameWriter(dir)

    # -----------------------------
    # World setup
    # -----------------------------
    W = H = GRID_SIZE
    robot_starting_x = W // 2
    robot_starting_y = H // 2
    pts = rng.random((N_ROBOTS, 2)) * np.array([W, H])
    # Robots spawn  
    all_cells = [(x, y) for x in range(W) for y in range(H)]
    spawn_idx = rng.choice(len(all_cells), size=N_ROBOTS, replace=False)
    spawn_positions = [all_cells[i] for i in spawn_idx]
    # robots = [Robot(i, x, y, local_covered=np.zeros((H, W), dtype=bool)) for i, (x, y) in enumerate(spawn_positions)]
    print(pts[0,0], pts[0,1])
    robots = [Robot(i, int(pts[i, 0]), int(pts[i,1]), local_covered=np.zeros((H, W), dtype=bool)) for i in range(N_ROBOTS)]

    # Targets & state
    targets = generate_unique_targets(GRID_SIZE, N_TARGETS)
    found_targets: Set[Tuple[int,int]] = set()

    # Global (for visualization only — NOT shared by robots)
    covered_global = np.zeros((H, W), dtype=bool)
    pher = np.zeros((H, W), dtype=float)
    DECAY_FACTOR = np.exp(-1.0 / TAU_DECAY)

    # -----------------------------
    # Matplotlib layout
    # -----------------------------
    fig = plt.figure(figsize=(12.5, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.12)
    ax_world = fig.add_subplot(gs[0, 0])
    ax_obs   = fig.add_subplot(gs[0, 1])

    world_cov_img = ax_world.imshow(coverage_to_image(covered_global), origin='lower',
                                    extent=[0, W, 0, H], vmin=0.0, vmax=1.0, zorder=0)
    world_pher_img = ax_world.imshow(pheromone_to_rgba(pher), origin='lower',
                                    extent=[0, W, 0, H], zorder=1)

    # Faint grid
    ax_world.set_xticks(np.arange(0, W+1, 10)); ax_world.set_yticks(np.arange(0, H+1, 10))
    ax_world.set_xticks(np.arange(0, W+1, 1), minor=True); ax_world.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_world.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_world.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)

    robot_colors = ['k' for _ in robots]  # black initially
    robot_scat = ax_world.scatter([r.x + 0.5 for r in robots],
                                [r.y + 0.5 for r in robots],
                                s=40, marker='o', c=robot_colors, zorder=3)

    # NEW: text labels above robots
    robot_labels = []
    for r in robots:
        t = ax_world.text(r.x + 0.6, r.y + 0.6, f"R{r.id}",
                        fontsize=7, color='k', zorder=5)
        robot_labels.append(t)

    # Targets
    if targets:
        tx, ty = zip(*targets)
    else:
        tx, ty = [], []
    ax_world.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                    s=20, marker='x', c='r', alpha=0.9, zorder=4)

    ax_world.set_title("World — Stigmergy (Local Maps + Random Walk + Pheromone)")
    ax_world.set_xlim(0, W); ax_world.set_ylim(0, H); ax_world.set_aspect('equal', adjustable='box')

    # Observer view (not shared by robots)
    obs_pher_img = ax_obs.imshow(pheromone_to_rgba(pher), origin='lower',
                                extent=[0, W, 0, H], zorder=0)
    und_plot = ax_obs.scatter([x + 0.5 for x in tx], [y + 0.5 for y in ty],
                            s=18, marker='x', c='r', label='Undiscovered', zorder=2)
    disc_plot = ax_obs.scatter([], [], s=25, marker='o', facecolors='none',
                            edgecolors='g', linewidths=1.5, label='Discovered', zorder=2)

    ax_obs.set_xticks(np.arange(0, W+1, 10)); ax_obs.set_yticks(np.arange(0, H+1, 10))
    ax_obs.set_xticks(np.arange(0, W+1, 1), minor=True); ax_obs.set_yticks(np.arange(0, H+1, 1), minor=True)
    ax_obs.grid(which='major', color='k', alpha=0.15, linewidth=0.5)
    ax_obs.grid(which='minor', color='k', alpha=0.05, linewidth=0.2)
    ax_obs.set_title("Observer — Pheromone Field (Robots don't share maps)")
    ax_obs.set_xlim(0, W); ax_obs.set_ylim(0, H); ax_obs.set_aspect('equal', adjustable='box')
    ax_obs.legend(loc='upper right', fontsize=8, frameon=False)

    

    # -----------------------------
    # Run
    # -----------------------------
    anim = run_animation(fig, update, frames=2000, interval_ms=INTERVAL_MS, blit=False)
    plt.show()
