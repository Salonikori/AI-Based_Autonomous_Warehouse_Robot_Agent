"""
Flask backend for the Warehouse Robot Agent website.

The agent itself lives in warehouse_robot_agent.py (unchanged). This file only
wraps it so the browser can create a simulation, advance it tick by tick, and
run headless benchmarks.

Run:  python app.py      then open http://127.0.0.1:5000
"""
import contextlib
import io
import os
import threading
import uuid
from collections import OrderedDict
from functools import lru_cache

from flask import Flask, jsonify, render_template, request

from groq_agent import GroqAgent

import warehouse_robot_agent as wra
from ai_policy import model_info

app = Flask(__name__)
groq = GroqAgent()

MAX_TICKS = 1500
MAX_SESSIONS = 200
SESSIONS = OrderedDict()   # sid -> Sim  (one simulation per browser tab)
SESSION_LOCK = threading.RLock()


def clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


class Sim:
    """One running warehouse + robot agent."""

    def __init__(self, seed, tasks, obstacles):
        self.n_tasks = tasks
        self.env = wra.Warehouse(seed=seed, n_dynamic=obstacles, n_tasks=tasks)
        self.agent = wra.WarehouseRobotAgent(self.env)
        self.t = 0
        self.seen_log = 0
        self.done = False

    def advance(self, n):
        # the agent prints its event log; we read agent.log instead
        with contextlib.redirect_stdout(io.StringIO()):
            for _ in range(n):
                if self.done:
                    break
                self.t += 1
                self.agent.step(self.t)
                self.env.tick(self.t, self.agent.pos)
                if self.agent.stats["done"] == self.n_tasks or self.t >= MAX_TICKS:
                    self.done = True

    def summary(self):
        s = self.agent.stats
        delays = s["delay"]
        return {
            "completed": s["done"], "steps": s["steps"], "replans": s["replans"],
            "charges": s["charges"], "waits": s["waits"],
            "avg_delay": round(sum(delays) / len(delays), 1) if delays else 0,
        }

    def state(self):
        a, e = self.agent, self.env
        new_log = a.log[self.seen_log:]
        self.seen_log = len(a.log)
        return {
            "t": self.t, "done": self.done, "n_tasks": self.n_tasks,
            "config": {"seed": self.env.seed, "tasks": self.n_tasks, "obstacles": len(e.dynamic)},
            "robot": a.pos, "battery": a.battery,
            "carrying": a.carrying.tid if a.carrying else None,
            "goal": {"pos": a.goal, "kind": a.goal_kind},
            "plan": a.plan[:40], "plan_len": len(a.plan), "sensed": len(a.blocked),
            "obstacles": e.dynamic,
            "pending": [{"id": k.tid, "pos": k.pickup, "priority": k.priority,
                         "utility": round(a.utility(k, self.t), 2),
                         "ai_score": next((round(p.score, 2) for p in a.last_ai_predictions if p.task_id == k.tid), None),
                         "ai_confidence": next((round(p.confidence * 100, 1) for p in a.last_ai_predictions if p.task_id == k.tid), None)}
                        for k in e.pending],
            "ai": {
                **model_info(),
                "choice": a.last_ai_choice,
                "ranked": [
                    {"task_id": p.task_id, "score": round(p.score, 2),
                     "confidence": round(p.confidence * 100, 1), "reason": p.reason}
                    for p in a.last_ai_predictions[:8]
                ],
            },
            "trail": a.trail[-120:],
            "stats": self.summary(),
            "log": new_log,
        }


@app.get("/")
def index():
    return render_template("index.html", rows=wra.H, cols=wra.W)


@app.post("/api/reset")
def reset():
    d = request.get_json(silent=True) or {}
    sid = d.get("sid") or uuid.uuid4().hex
    sim = Sim(clamp(d.get("seed"), 0, 9999, 1),
              clamp(d.get("tasks"), 1, 20, 10),
              clamp(d.get("obstacles"), 0, 8, 3))
    with SESSION_LOCK:
        SESSIONS[sid] = sim
        SESSIONS.move_to_end(sid)
        while len(SESSIONS) > MAX_SESSIONS:
            SESSIONS.popitem(last=False)
    out = sim.state()
    out["sid"] = sid
    out["layout"] = {"grid": ["".join(row) for row in sim.env.grid],
                     "charger": wra.CHARGER, "dispatch": wra.DISPATCH}
    return jsonify(out)


@app.post("/api/step")
def step():
    d = request.get_json(silent=True) or {}
    with SESSION_LOCK:
        sim = SESSIONS.get(d.get("sid"))
        if sim is None:
            return jsonify(error="No simulation found. Press Reset to start one."), 404
        sim.advance(clamp(d.get("n"), 1, 25, 1))
        return jsonify(sim.state())


def get_sim(d):
    with SESSION_LOCK:
        return SESSIONS.get(d.get("sid"))


@app.post("/api/order")
def rush_order():
    """Add an express (priority 3) order to show how the utility function reacts."""
    d = request.get_json(silent=True) or {}
    with SESSION_LOCK:
        sim = SESSIONS.get(d.get("sid"))
        if sim is None or sim.done:
            return jsonify(error="Start or reset the simulation first."), 400
        env = sim.env
        env.total_tasks += 1
        sim.n_tasks += 1
        env.tasks_created += 1
        env.pending.append(wra.Task(env.tasks_created, env.rng.choice(env.pickup_spots),
                                    wra.DISPATCH, 3, sim.t))
        sim.agent.say(sim.t, "Rush order added by operator (priority 3)")
        return jsonify(sim.state())


@app.post("/api/obstacle")
def add_obstacle():
    """Drop a moving obstacle on a free cell (click the map)."""
    d = request.get_json(silent=True) or {}
    cell = (clamp(d.get("r"), 0, wra.H - 1, 0), clamp(d.get("c"), 0, wra.W - 1, 0))
    with SESSION_LOCK:
        sim = SESSIONS.get(d.get("sid"))
        if sim is None:
            return jsonify(error="Start or reset the simulation first."), 400
        env = sim.env
        if cell in env.dynamic:
            env.dynamic.remove(cell)                      # click again to remove
        elif (env.grid[cell[0]][cell[1]] == "."
              and cell != sim.agent.pos
              and cell not in (wra.CHARGER, wra.DISPATCH)
              and not any(task.pickup == cell for task in env.pending)
              and len(env.dynamic) < 12):
            env.dynamic.append(cell)
        else:
            return jsonify(error="That cell is not free."), 400
        return jsonify(sim.state())


@lru_cache(maxsize=8)
def run_benchmark(runs):
    rows = []
    for seed in range(1, runs + 1):
        sim = Sim(seed, 10, 3)
        sim.advance(MAX_TICKS)
        rows.append({"seed": seed, "ticks": sim.t, "battery": sim.agent.battery,
                     **sim.summary()})
    keys = ["ticks", "steps", "replans", "charges", "waits", "avg_delay"]
    avg = {k: round(sum(r[k] for r in rows) / len(rows), 1) for k in keys}
    return {"rows": rows, "average": avg}


@app.get("/api/benchmark")
def benchmark():
    return jsonify(run_benchmark(clamp(request.args.get("runs"), 1, 20, 10)))



@app.get("/api/llm-status")
def llm_status():
    return jsonify(groq.status())


@app.post("/api/groq")
def groq_chat():
    """Ask Groq about the current warehouse state using the live agent context."""
    d = request.get_json(silent=True) or {}
    sim = get_sim(d)
    message = str(d.get("message", "")).strip()
    if not message:
        return jsonify(error="Enter a message for Groq."), 400
    if not sim:
        return jsonify(error="Start or reset the simulation first."), 400
    if not groq.enabled:
        return jsonify(error="Groq is not configured. Add GROQ_API_KEY to .env and restart the server."), 503
    try:
        return jsonify(groq.chat(message, sim.state(), sim.agent, sim.env))
    except Exception as exc:
        return jsonify(error=f"Groq request failed: {exc}"), 502

@app.get("/api/health")
def health():
    return jsonify(status="ok", simulation="ready", groq=groq.status())


@app.get("/api/ai-model")
def ai_model():
    """Expose the active ML decision model and its input features."""
    return jsonify(model_info())


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=os.getenv("FLASK_DEBUG", "0") == "1")
