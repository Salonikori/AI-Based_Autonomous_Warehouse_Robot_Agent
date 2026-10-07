"""
AI-Based Autonomous Warehouse Robot Agent
-----------------------------------------
Domain      : Robotics
Application : Intelligent warehouse navigation and package delivery
Agent type  : Model-based + Goal-based + Utility-based agent (with replanning)

Run:
    python warehouse_robot_agent.py               # run and print event log + summary
    python warehouse_robot_agent.py --animate     # live ASCII animation in terminal
    python warehouse_robot_agent.py --plot        # also save warehouse_run.png (needs matplotlib)
    python warehouse_robot_agent.py --seed 5      # different random scenario
"""

import argparse
import heapq
import random
import time
from dataclasses import dataclass

from ai_policy import OrderDecisionModel

# ----------------------------------------------------------------------------
# 1. ENVIRONMENT  (the warehouse)
# ----------------------------------------------------------------------------
H, W = 12, 20
SHELF_ROWS = [(3, 4), (7, 8)]
SHELF_COLS = [(3, 5), (7, 9), (11, 13), (15, 17)]
CHARGER = (1, 1)
DISPATCH = (10, 18)
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def build_static_map():
    grid = [["." for _ in range(W)] for _ in range(H)]
    for c in range(W):
        grid[0][c] = grid[H - 1][c] = "#"
    for r in range(H):
        grid[r][0] = grid[r][W - 1] = "#"
    for r1, r2 in SHELF_ROWS:
        for c1, c2 in SHELF_COLS:
            for r in range(r1, r2 + 1):
                for c in range(c1, c2 + 1):
                    grid[r][c] = "#"
    return grid


@dataclass
class Task:
    tid: int
    pickup: tuple
    drop: tuple
    priority: int          # 1 = normal, 2 = urgent, 3 = express
    created: int = 0


class Warehouse:
    """Partially observable, dynamic, stochastic environment."""

    def __init__(self, seed=1, n_dynamic=3, n_tasks=10):
        self.seed = seed
        self.rng = random.Random(seed)
        self.grid = build_static_map()
        self.free = [(r, c) for r in range(H) for c in range(W) if self.grid[r][c] == "."]
        self.pickup_spots = [
            (r, c) for (r, c) in self.free
            if (r, c) not in (CHARGER, DISPATCH)
            and any(self.grid[r + dr][c + dc] == "#" and 0 < r + dr < H - 1 and 0 < c + dc < W - 1
                    for dr, dc in MOVES)
        ]
        # moving obstacles = humans / other robots
        self.dynamic = []
        while len(self.dynamic) < n_dynamic:
            p = self.rng.choice(self.free)
            if manhattan(p, CHARGER) > 4 and p not in self.dynamic:
                self.dynamic.append(p)
        self.total_tasks = n_tasks
        self.tasks_created = 0
        self.pending = []
        for _ in range(6):
            self._new_task(0)

    def _new_task(self, t):
        if self.tasks_created >= self.total_tasks:
            return
        self.tasks_created += 1
        self.pending.append(Task(self.tasks_created, self.rng.choice(self.pickup_spots),
                                 DISPATCH, self.rng.choice([1, 1, 2, 3]), t))

    # ---- what the sensors can see (partial observability) ----
    def sense(self, pos, radius=2):
        return {o for o in self.dynamic if manhattan(o, pos) <= radius}

    def is_free(self, cell):
        r, c = cell
        return 0 <= r < H and 0 <= c < W and self.grid[r][c] != "#" and cell not in self.dynamic

    # ---- environment dynamics (stochastic) ----
    def tick(self, t, robot_pos):
        new_positions = []
        for o in self.dynamic:
            if self.rng.random() < 0.6:
                dr, dc = self.rng.choice(MOVES)
                n = (o[0] + dr, o[1] + dc)
                if (self.grid[n[0]][n[1]] != "#" and n != robot_pos
                        and n not in new_positions and n not in self.dynamic):
                    o = n
            new_positions.append(o)
        self.dynamic = new_positions
        if self.rng.random() < 0.08:           # new order arrives at random
            self._new_task(t)


# ----------------------------------------------------------------------------
# 2. PATH PLANNING  (A* search)
# ----------------------------------------------------------------------------
def astar(grid, start, goal, blocked):
    if start == goal:
        return []
    open_set = [(manhattan(start, goal), 0, start)]
    came, g = {start: None}, {start: 0}
    while open_set:
        _, cost, cur = heapq.heappop(open_set)
        if cur == goal:
            path = []
            while cur != start:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        if cost > g[cur]:
            continue
        for dr, dc in MOVES:
            n = (cur[0] + dr, cur[1] + dc)
            if not (0 <= n[0] < H and 0 <= n[1] < W) or grid[n[0]][n[1]] == "#" or n in blocked:
                continue
            ng = cost + 1
            if ng < g.get(n, 10 ** 9):
                g[n] = ng
                came[n] = cur
                heapq.heappush(open_set, (ng + manhattan(n, goal), ng, n))
    return None


# ----------------------------------------------------------------------------
# 3. THE AGENT  (sense -> update model -> decide -> plan -> act)
# ----------------------------------------------------------------------------
class WarehouseRobotAgent:
    MAX_BATTERY = 100
    LOW_BATTERY = 25
    RESERVE = 10
    CHARGE_RATE = 10

    def __init__(self, env):
        self.env = env
        self.pos = CHARGER
        self.battery = 60
        self.carrying = None
        self.task = None
        self.goal = None
        self.goal_kind = None       # 'pickup' | 'dispatch' | 'charger'
        self.plan = []
        self.blocked = set()        # internal world model: currently seen obstacles
        self.stats = dict(done=0, steps=0, replans=0, charges=0, waits=0, delay=[])
        self.trail = [self.pos]
        self.log = []
        # ML decision layer: ranks feasible orders; A* and battery rules remain hard safety guards.
        self.ai = OrderDecisionModel()
        self.last_ai_predictions = []
        self.last_ai_choice = None

    def say(self, t, msg):
        line = f"[t={t:03d}] {msg}"
        self.log.append(line)
        print(line)

    # -- PERCEPTION + MODEL UPDATE -------------------------------------------
    def perceive(self):
        self.blocked = self.env.sense(self.pos)
        return self.blocked

    # -- DECISION: goal selection using a utility function --------------------
    def energy_needed(self, task):
        return (manhattan(self.pos, task.pickup) + manhattan(task.pickup, task.drop)
                + manhattan(task.drop, CHARGER) + self.RESERVE)

    def utility(self, task, t):
        distance = manhattan(self.pos, task.pickup) + manhattan(task.pickup, task.drop)
        waiting = t - task.created
        return (task.priority * 10 + 0.2 * waiting) / (distance + 1)

    def choose_goal(self, t):
        if self.carrying:                                   # finish current delivery
            return self.carrying.drop, "dispatch"
        feasible = [k for k in self.env.pending if self.energy_needed(k) <= self.battery]
        if self.battery < self.LOW_BATTERY or (self.env.pending and not feasible):
            self.last_ai_predictions = []
            self.last_ai_choice = "charger"
            return CHARGER, "charger"                       # battery management
        if feasible:
            # ML policy ranks feasible orders. It never overrides hard battery/path safety.
            ranked = self.ai.rank(feasible, self, t)
            self.last_ai_predictions = ranked
            best = next(k for k in feasible if k.tid == ranked[0].task_id)
            self.task = best
            self.last_ai_choice = best.tid
            self.say(t, f"ML selected task T{best.tid} (priority {best.priority}, "
                        f"AI score {ranked[0].score:.2f}, battery {self.battery}%)")
            return best.pickup, "pickup"
        return None, None                                    # nothing to do

    # -- PLANNING ---------------------------------------------------------------
    def replan(self, t):
        path = astar(self.env.grid, self.pos, self.goal, self.blocked)
        if path is None:
            self.plan = []
            return False
        self.plan = path
        return True

    # -- ONE FULL AGENT CYCLE ---------------------------------------------------
    def step(self, t):
        self.perceive()

        # arrived at a goal? -> perform the special action
        if self.goal is not None and self.pos == self.goal:
            if self.goal_kind == "pickup":
                self.carrying = self.task
                self.env.pending.remove(self.task)
                self.say(t, f"Picked up package T{self.task.tid} at {self.pos}")
            elif self.goal_kind == "dispatch":
                self.stats["done"] += 1
                self.stats["delay"].append(t - self.carrying.created)
                self.say(t, f"Delivered T{self.carrying.tid} to dispatch  "
                            f"(completed: {self.stats['done']})")
                self.carrying = self.task = None
            elif self.goal_kind == "charger":
                if self.battery < self.MAX_BATTERY:
                    self.battery = min(self.MAX_BATTERY, self.battery + self.CHARGE_RATE)
                    if self.battery == self.MAX_BATTERY:
                        self.stats["charges"] += 1
                        self.say(t, "Fully charged")
                    return
            self.goal = self.goal_kind = None
            self.plan = []

        # choose a goal when idle (or re-evaluate low battery en route to a pickup)
        if self.goal is None:
            self.goal, self.goal_kind = self.choose_goal(t)
            if self.goal is None:
                self.stats["waits"] += 1
                return
            self.replan(t)

        # next step blocked by a newly seen obstacle -> replan (reactive behaviour)
        if self.plan and self.plan[0] in self.blocked:
            self.stats["replans"] += 1
            self.say(t, f"Obstacle ahead at {self.plan[0]} -> replanning route")
            self.replan(t)

        if not self.plan:
            if self.pos != self.goal and not self.replan(t):
                self.stats["waits"] += 1                     # path blocked -> wait
            return                                           # (arrival is handled next cycle)

        # ACT: move one cell
        nxt = self.plan.pop(0)
        if self.env.is_free(nxt):
            self.pos = nxt
            self.battery -= 1
            self.stats["steps"] += 1
            self.trail.append(self.pos)
        else:
            self.stats["waits"] += 1
            self.plan.insert(0, nxt)


# ----------------------------------------------------------------------------
# 4. VISUALISATION
# ----------------------------------------------------------------------------
def render(env, agent):
    rows = []
    for r in range(H):
        line = ""
        for c in range(W):
            p = (r, c)
            if p == agent.pos:
                line += "R"
            elif p in env.dynamic:
                line += "X"
            elif p == CHARGER:
                line += "C"
            elif p == DISPATCH:
                line += "D"
            elif any(k.pickup == p for k in env.pending):
                line += "P"
            else:
                line += env.grid[r][c]
        rows.append(line)
    return "\n".join(rows)


def save_plot(env, agent, filename="warehouse_run.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plot (pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in range(H):
        for c in range(W):
            if env.grid[r][c] == "#":
                ax.add_patch(plt.Rectangle((c, H - 1 - r), 1, 1, color="#555"))
    ys = [H - 1 - r + 0.5 for r, c in agent.trail]
    xs = [c + 0.5 for r, c in agent.trail]
    ax.plot(xs, ys, color="tab:blue", linewidth=1.5, label="Robot path")
    ax.scatter(CHARGER[1] + .5, H - 1 - CHARGER[0] + .5, c="green", s=150, marker="s", label="Charger")
    ax.scatter(DISPATCH[1] + .5, H - 1 - DISPATCH[0] + .5, c="red", s=150, marker="s", label="Dispatch")
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal")
    ax.set_title("Autonomous warehouse robot - trajectory")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=3)
    fig.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")


# ----------------------------------------------------------------------------
# 5. MAIN SIMULATION LOOP
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=1500)
    args = ap.parse_args()

    env = Warehouse(seed=args.seed, n_tasks=args.tasks)
    agent = WarehouseRobotAgent(env)
    print(f"Warehouse {H}x{W} | tasks: {args.tasks} | moving obstacles: {len(env.dynamic)}\n"
          "Legend: R robot, X moving obstacle, # shelf/wall, C charger, D dispatch, P pending pickup\n")

    t = 0
    while t < args.max_ticks:
        t += 1
        agent.step(t)
        env.tick(t, agent.pos)
        if args.animate:
            print("\033[H\033[J" + render(env, agent) +
                  f"\n t={t}  battery={agent.battery}%  delivered={agent.stats['done']}/{args.tasks}")
            time.sleep(0.05)
        if agent.stats["done"] == args.tasks:
            break

    s = agent.stats
    avg = sum(s["delay"]) / len(s["delay"]) if s["delay"] else 0
    print("\n================ SUMMARY ================")
    print(f"Tasks completed        : {s['done']}/{args.tasks}")
    print(f"Total time (ticks)     : {t}")
    print(f"Distance travelled     : {s['steps']} cells")
    print(f"Replans (obstacles)    : {s['replans']}")
    print(f"Full charge cycles     : {s['charges']}")
    print(f"Idle/wait ticks        : {s['waits']}")
    print(f"Avg order->delivery    : {avg:.1f} ticks")
    print(f"Final battery          : {agent.battery}%")
    if args.plot:
        save_plot(env, agent)


if __name__ == "__main__":
    main()
