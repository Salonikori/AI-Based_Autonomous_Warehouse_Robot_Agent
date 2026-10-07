// Warehouse Robot Agent — front end
// Talks to the Flask API in app.py and draws the live simulation on <canvas id="floor">.

(() => {
  const $ = (id) => document.getElementById(id);

  const canvas = $("floor");
  const ctx = canvas.getContext("2d");

  const els = {
    pill: $("pill"), progfill: $("progfill"), proglabel: $("proglabel"),
    now: $("now"), overlay: $("overlay"), resTitle: $("res-title"), resStats: $("res-stats"),
    battery: $("battery"), batfill: $("batfill"), carry: $("carry"),
    sT: $("s-t"), sSteps: $("s-steps"), sReplans: $("s-replans"), sCharges: $("s-charges"),
    cSense: $("c-sense"), cModel: $("c-model"), cDecide: $("c-decide"), cPlan: $("c-plan"), cAct: $("c-act"),
    queue: $("queue"), log: $("log"),
    aiDot: $("ai-dot"), aiModel: $("ai-model"), aiMeta: $("ai-meta"), aiChoice: $("ai-choice"),
    groqDot: $("groq-dot"), groqStatus: $("groq-status"), groqChat: $("groq-chat"), groqInput: $("groq-input"), groqSend: $("groq-send"),
    play: $("play"), step: $("step"), finish: $("finish"), reset: $("reset"), again: $("again"),
    rush: $("rush"),
    seed: $("seed"), tasks: $("tasks"), obstacles: $("obstacles"),
    msg: $("msg"),
    scenarioStatus: $("scenario-status"),
  };

  let sid = null;
  let layout = null;   // {grid, charger, dispatch}
  let latest = null;   // last state() from server
  let timer = null;
  let speedMs = 160;
  let busy = false;
  let resetSeq = 0;

  const CELL = () => canvas.width / layout.cols;
  const CELLY = () => canvas.height / layout.rows;

  // ------------------------------------------------------- animation state
  let animRobotPx = null;     // [x,y] currently-rendered (interpolated) position
  let targetRobotPx = null;   // [x,y] the cell the robot should ease toward
  let prevRobotCell = null;   // last known [r,c], used to derive heading
  let robotHeading = 0;       // radians, 0 = facing +x (right)
  let particles = [];         // {x,y,age,life,color,kind}
  let rafId = null;

  function spawnBurst(x, y, color, kind) {
    particles.push({ x, y, age: 0, life: kind === "ring" ? 620 : 480, color, kind });
  }

  function tickAnimation() {
    if (layout) {
      if (targetRobotPx) {
        if (!animRobotPx) animRobotPx = targetRobotPx.slice();
        animRobotPx[0] += (targetRobotPx[0] - animRobotPx[0]) * 0.22;
        animRobotPx[1] += (targetRobotPx[1] - animRobotPx[1]) * 0.22;
      }
      particles.forEach((p) => { p.age += 16; });
      particles = particles.filter((p) => p.age < p.life);
      draw();
    }
    rafId = requestAnimationFrame(tickAnimation);
  }

  // ---------------------------------------------------------------- helpers
  async function api(path, body) {
    const opts = body === undefined
      ? {}
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
    const r = await fetch(path, opts);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `Request failed (${r.status})`);
    return data;
  }

  function setMsg(text) { els.msg.textContent = text || ""; }

  function setPill(state) {
    els.pill.className = "pill";
    if (state === "run") { els.pill.classList.add("run"); els.pill.textContent = "Running"; }
    else if (state === "done") { els.pill.classList.add("done"); els.pill.textContent = "Complete"; }
    else if (state === "warn") { els.pill.classList.add("warn"); els.pill.textContent = "Stuck"; }
    else { els.pill.textContent = "Paused"; }
  }

  // ---------------------------------------------------------------- drawing
  function cellCenter(r, c) {
    return [c * CELL() + CELL() / 2, r * CELLY() + CELLY() / 2];
  }

  function roundRectPath(x, y, w, h, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + w, y, x + w, y + h, radius);
    ctx.arcTo(x + w, y + h, x, y + h, radius);
    ctx.arcTo(x, y + h, x, y, radius);
    ctx.arcTo(x, y, x + w, y, radius);
    ctx.closePath();
  }

  function roundRect(x, y, w, h, radius) {
    roundRectPath(x, y, w, h, radius);
    ctx.fill();
  }

  function draw() {
    const W = canvas.width, H = canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#0D141B";
    ctx.fillRect(0, 0, W, H);
    if (!layout) return;

    drawFloor();
    drawShelves();
    drawGridLines();

    if (!latest) return;

    drawTrail();
    drawCharger(layout.charger);
    drawDispatch(layout.dispatch);
    drawPlan();
    drawSensorCone();
    (latest.pending || []).forEach(drawPackage);
    (latest.obstacles || []).forEach(([r, c]) => drawCart(...cellCenter(r, c)));
    drawRobot();
    drawParticles();
  }

  // -- floor: soft checkerboard tiling + ambient light pooling near dispatch
  function drawFloor() {
    const w = CELL(), h = CELLY();
    for (let r = 0; r < layout.rows; r++) {
      for (let c = 0; c < layout.cols; c++) {
        if (layout.grid[r][c] === "#") continue;
        ctx.fillStyle = (r + c) % 2 === 0 ? "#0F171F" : "#0C1319";
        ctx.fillRect(c * w, r * h, w, h);
      }
    }
    if (layout.dispatch) {
      const [dx, dy] = cellCenter(...layout.dispatch);
      const g = ctx.createRadialGradient(dx, dy, 10, dx, dy, Math.max(canvas.width, canvas.height) * 0.6);
      g.addColorStop(0, "rgba(255,120,90,0.10)");
      g.addColorStop(1, "rgba(255,120,90,0)");
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  }

  function drawGridLines() {
    ctx.strokeStyle = "rgba(148,178,199,0.05)";
    ctx.lineWidth = 1;
    for (let c = 0; c <= layout.cols; c++) {
      ctx.beginPath(); ctx.moveTo(c * CELL(), 0); ctx.lineTo(c * CELL(), canvas.height); ctx.stroke();
    }
    for (let r = 0; r <= layout.rows; r++) {
      ctx.beginPath(); ctx.moveTo(0, r * CELLY()); ctx.lineTo(canvas.width, r * CELLY()); ctx.stroke();
    }
  }

  // -- shelving racks in place of flat "#" blocks
  function drawShelves() {
    const w = CELL(), h = CELLY();
    layout.grid.forEach((row, r) => {
      for (let c = 0; c < row.length; c++) {
        if (row[c] !== "#") continue;
        const x = c * w, y = r * h;
        ctx.fillStyle = "rgba(0,0,0,0.35)";
        ctx.fillRect(x + 2, y + h - 4, w - 4, 4);
        ctx.fillStyle = "#182530";
        ctx.fillRect(x + 2, y + 2, w - 4, h - 6);
        ctx.strokeStyle = "#2A3A48";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 2, y + 2, w - 4, h - 6);
        ctx.fillStyle = "#0E161D";
        const beamY = [y + h * 0.32, y + h * 0.62];
        beamY.forEach((by) => { ctx.fillRect(x + 3, by, w - 6, Math.max(2, h * 0.045)); });
        const boxColors = ["#D9A05B", "#B98747", "#8FA6B8"];
        let bi = 0;
        [y + 4, y + h * 0.34, y + h * 0.64].forEach((by, row2) => {
          const bh = h * 0.24;
          for (let bx = x + 5; bx < x + w - 6; bx += w * 0.32) {
            ctx.fillStyle = boxColors[bi % boxColors.length];
            const bw = w * 0.24;
            ctx.fillRect(bx, by + bh * 0.15, Math.min(bw, x + w - 6 - bx), bh * 0.7);
            bi++;
          }
        });
      }
    });
  }

  function fadedTrailColor(alpha) { return `rgba(70,199,255,${alpha.toFixed(3)})`; }

  function drawTrail() {
    const trail = latest.trail;
    if (!trail || trail.length < 2) return;
    const n = trail.length;
    for (let i = 1; i < n; i++) {
      const [x0, y0] = cellCenter(...trail[i - 1]);
      const [x1, y1] = cellCenter(...trail[i]);
      const alpha = 0.05 + 0.35 * (i / n);
      ctx.strokeStyle = fadedTrailColor(alpha);
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
      ctx.stroke();
    }
  }

  function drawPlan() {
    if (!latest.plan || !latest.plan.length) return;
    const pts = [ (animRobotPx || cellCenter(...latest.robot)), ...latest.plan.map(([r, c]) => cellCenter(r, c)) ];
    ctx.strokeStyle = "#46C7FF";
    ctx.lineWidth = 3;
    ctx.setLineDash([7, 6]);
    ctx.beginPath();
    pts.forEach(([x, y], i) => { if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.stroke();
    ctx.setLineDash([]);
    // small directional arrowheads every few segments
    for (let i = 2; i < pts.length; i += 3) {
      const [x0, y0] = pts[i - 1], [x1, y1] = pts[i];
      const ang = Math.atan2(y1 - y0, x1 - x0);
      drawArrowhead((x0 + x1) / 2, (y0 + y1) / 2, ang, "#46C7FF");
    }
  }

  function drawArrowhead(x, y, ang, color) {
    const s = CELL() * 0.16;
    ctx.save();
    ctx.translate(x, y); ctx.rotate(ang);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(s, 0); ctx.lineTo(-s * 0.6, s * 0.6); ctx.lineTo(-s * 0.6, -s * 0.6);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function drawSensorCone() {
    if (!latest.robot) return;
    const [x, y] = animRobotPx || cellCenter(...latest.robot);
    const rad = 2.5 * CELL();
    const spread = Math.PI * 0.62;
    ctx.fillStyle = "rgba(245,179,1,0.09)";
    ctx.strokeStyle = "rgba(245,179,1,0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, rad, robotHeading - spread / 2, robotHeading + spread / 2);
    ctx.closePath();
    ctx.fill(); ctx.stroke();
  }

  // -- packages as small crates with a lit top face and shaded side face
  function drawPackage(k) {
    const [x, y] = cellCenter(...k.pos);
    const colors = { 1: "#D9A05B", 2: "#F5B301", 3: "#FF5B60" };
    const base = colors[k.priority] || "#D9A05B";
    const s = CELL() * 0.52;
    const pulse = k.priority === 3 ? 1 + 0.06 * Math.sin(Date.now() / 180) : 1;
    const hs = (s * pulse) / 2;
    ctx.save();
    ctx.translate(x, y);
    // drop shadow
    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.beginPath(); ctx.ellipse(0, hs * 0.85, hs * 0.9, hs * 0.32, 0, 0, Math.PI * 2); ctx.fill();
    // side face (darker)
    ctx.fillStyle = shade(base, -0.35);
    ctx.fillRect(-hs, -hs * 0.2, hs * 2, hs * 1.1);
    // top face (lighter parallelogram)
    ctx.fillStyle = shade(base, 0.18);
    ctx.beginPath();
    ctx.moveTo(-hs, -hs * 0.2);
    ctx.lineTo(0, -hs * 0.85);
    ctx.lineTo(hs, -hs * 0.2);
    ctx.lineTo(0, hs * 0.1);
    ctx.closePath(); ctx.fill();
    if (k.priority === 3) {
      ctx.strokeStyle = "rgba(255,91,96,0.85)";
      ctx.lineWidth = 2;
      ctx.strokeRect(-hs - 2, -hs * 0.2 - 2, hs * 2 + 4, hs * 1.1 + 4);
    }
    ctx.fillStyle = "#0A0E13";
    ctx.font = `700 ${Math.floor(CELL() * 0.32)}px 'Barlow Condensed', sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(k.priority, 0, hs * 0.35);
    ctx.restore();
  }

  function shade(hex, amt) {
    const c = hex.replace("#", "");
    const r = parseInt(c.substring(0, 2), 16), g = parseInt(c.substring(2, 4), 16), b = parseInt(c.substring(4, 6), 16);
    const f = (v) => Math.max(0, Math.min(255, Math.round(v + (amt > 0 ? (255 - v) * amt : v * amt))));
    return `rgb(${f(r)},${f(g)},${f(b)})`;
  }

  // -- moving obstacles as little carts, not plain dots
  function drawCart(x, y) {
    const s = CELL() * 0.42;
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.beginPath(); ctx.ellipse(0, s * 0.55, s * 0.75, s * 0.24, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#FF9142";
    roundRectPath(-s * 0.7, -s * 0.4, s * 1.4, s * 0.75, 4); ctx.fill();
    ctx.fillStyle = "#C9601C";
    ctx.fillRect(-s * 0.7, -s * 0.4, s * 1.4, s * 0.2);
    ctx.fillStyle = "#1B242B";
    [-s * 0.42, s * 0.42].forEach((wx) => { ctx.beginPath(); ctx.arc(wx, s * 0.42, s * 0.18, 0, Math.PI * 2); ctx.fill(); });
    ctx.restore();
  }

  function drawCharger([r, c]) {
    const [x, y] = cellCenter(r, c);
    const s = CELL() * 0.62;
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = "rgba(0,0,0,0.3)";
    ctx.beginPath(); ctx.ellipse(0, s * 0.42, s * 0.62, s * 0.2, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#163024";
    roundRectPath(-s / 2, -s / 2, s, s, 6); ctx.fill();
    ctx.strokeStyle = "#3EDC9E"; ctx.lineWidth = 2.5;
    roundRectPath(-s / 2, -s / 2, s, s, 6); ctx.stroke();
    ctx.fillStyle = "#3EDC9E";
    ctx.beginPath();
    ctx.moveTo(2, -s * 0.32); ctx.lineTo(-s * 0.14, s * 0.05); ctx.lineTo(0, s * 0.05);
    ctx.lineTo(-2, s * 0.32); ctx.lineTo(s * 0.14, -s * 0.05); ctx.lineTo(0, -s * 0.05);
    ctx.closePath(); ctx.fill();
    ctx.restore();
  }

  function drawDispatch([r, c]) {
    const [x, y] = cellCenter(r, c);
    const s = CELL() * 0.7;
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = "#301418";
    roundRectPath(-s / 2, -s / 2, s, s, 5); ctx.fill();
    ctx.strokeStyle = "#FF5B60"; ctx.lineWidth = 2.5;
    roundRectPath(-s / 2, -s / 2, s, s, 5); ctx.stroke();
    // loading-bay stripes along the top edge
    ctx.fillStyle = "rgba(255,91,96,0.55)";
    const stripeW = s * 0.14;
    for (let sx = -s / 2 + 3; sx < s / 2 - 3; sx += stripeW * 1.8) {
      ctx.fillRect(sx, -s / 2 + 3, stripeW, s * 0.16);
    }
    ctx.restore();
  }

  // -- robot: small AGV body, wheels, headlight, drop shadow, optional crate
  function drawRobot() {
    if (!latest.robot) return;
    if (!targetRobotPx) targetRobotPx = cellCenter(...latest.robot);
    const [x, y] = animRobotPx || targetRobotPx;
    const s = CELL() * 0.66;
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.beginPath(); ctx.ellipse(0, s * 0.5, s * 0.62, s * 0.2, 0, 0, Math.PI * 2); ctx.fill();
    ctx.rotate(robotHeading);
    // wheels
    ctx.fillStyle = "#1B242B";
    [-s * 0.34, s * 0.34].forEach((wy) => { ctx.beginPath(); ctx.arc(0, wy, s * 0.16, 0, Math.PI * 2); ctx.fill(); });
    // body
    ctx.fillStyle = "#F5B301";
    roundRectPath(-s * 0.5, -s * 0.42, s, s * 0.84, 6); ctx.fill();
    ctx.strokeStyle = "#B9840A"; ctx.lineWidth = 1.5;
    roundRectPath(-s * 0.5, -s * 0.42, s, s * 0.84, 6); ctx.stroke();
    // headlight / direction marker
    ctx.fillStyle = "#FFF3D0";
    ctx.beginPath();
    ctx.moveTo(s * 0.5, -s * 0.14); ctx.lineTo(s * 0.72, 0); ctx.lineTo(s * 0.5, s * 0.14);
    ctx.closePath(); ctx.fill();
    ctx.restore();
    // carried crate sits on top, upright regardless of heading
    if (latest.carrying) {
      ctx.save();
      ctx.translate(x, y - s * 0.05);
      const cs = s * 0.34;
      ctx.fillStyle = "#D9A05B";
      ctx.fillRect(-cs / 2, -cs / 2, cs, cs);
      ctx.strokeStyle = "#0A0E13"; ctx.lineWidth = 1;
      ctx.strokeRect(-cs / 2, -cs / 2, cs, cs);
      ctx.restore();
    }
  }

  function drawParticles() {
    particles.forEach((p) => {
      const t = p.age / p.life;
      const alpha = 1 - t;
      if (p.kind === "ring") {
        const r = CELL() * (0.3 + t * 1.1);
        ctx.strokeStyle = p.color.replace("ALPHA", alpha.toFixed(2));
        ctx.lineWidth = 3 * (1 - t) + 1;
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.stroke();
      } else {
        const r = CELL() * (0.18 + t * 0.5);
        ctx.fillStyle = p.color.replace("ALPHA", (alpha * 0.6).toFixed(2));
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fill();
      }
    });
  }

  // ---------------------------------------------------------------- UI sync
  function classifyLog(line) {
    if (line.includes("Picked up")) return "l-pick";
    if (line.includes("Delivered")) return "l-done";
    if (line.includes("replanning")) return "l-plan";
    if (line.includes("selected task")) return "l-sel";
    if (line.includes("Rush order")) return "l-rush";
    return "";
  }

  function appendLog(lines) {
    lines.forEach((line) => {
      const m = line.match(/^\[t=(\d+)\]\s*(.*)$/);
      const t = m ? m[1] : "";
      const msg = m ? m[2] : line;
      const li = document.createElement("li");
      li.className = classifyLog(line);
      li.innerHTML = `<span class="tk">t${t}</span><span>${msg}</span>`;
      els.log.insertBefore(li, els.log.firstChild);
    });
    while (els.log.children.length > 80) els.log.removeChild(els.log.lastChild);
  }

  function renderQueue(pending) {
    if (!pending || !pending.length) {
      els.queue.innerHTML = `<tr><td colspan="4">No pending orders</td></tr>`;
      return;
    }
    const sorted = [...pending].sort((a, b) => b.utility - a.utility);
    els.queue.innerHTML = sorted.map((k, i) => `
      <tr class="${i === 0 ? "next" : ""}">
        <td>T${k.id}</td>
        <td><span class="pr ${k.priority === 2 ? "p2" : k.priority === 3 ? "p3" : ""}">${k.priority}</span></td>
        <td>${k.utility.toFixed(2)}</td>
        <td>${k.ai_score == null ? "—" : k.ai_score.toFixed(2)}</td>
      </tr>`).join("");
  }

  function renderCycle(state) {
    els.cSense.textContent = `${state.sensed} obstacle${state.sensed === 1 ? "" : "s"} in range`;
    els.cModel.textContent = `battery ${state.battery}%, carrying ${state.carrying ? "T" + state.carrying : "nothing"}`;
    const gk = state.goal.kind;
    els.cDecide.textContent = gk === "charger" ? "Head to charger"
      : gk === "pickup" ? "Go pick up an order"
      : gk === "dispatch" ? "Deliver current order"
      : "Idle";
    els.cPlan.textContent = state.plan_len ? `${state.plan_len} cell route` : (state.robot ? "At goal / none" : "-");
    const last = state.log.length ? state.log[state.log.length - 1] : null;
    els.cAct.textContent = last ? last.replace(/^\[t=\d+\]\s*/, "") : "Move / wait";
  }

  function render(state) {
    latest = state;
    updateScenarioStatus(state.config);

    if (state.robot) {
      if (prevRobotCell && (prevRobotCell[0] !== state.robot[0] || prevRobotCell[1] !== state.robot[1])) {
        const dr = state.robot[0] - prevRobotCell[0], dc = state.robot[1] - prevRobotCell[1];
        robotHeading = Math.atan2(dr, dc);
      }
      prevRobotCell = state.robot;
      targetRobotPx = cellCenter(...state.robot);
      if (!animRobotPx) animRobotPx = targetRobotPx.slice();
    }

    (state.log || []).forEach((line) => {
      if (line.includes("Picked up") && state.robot) {
        const [x, y] = cellCenter(...state.robot);
        spawnBurst(x, y, "rgba(245,179,1,ALPHA)", "ring");
      } else if (line.includes("Delivered") && layout) {
        const [x, y] = cellCenter(...layout.dispatch);
        spawnBurst(x, y, "rgba(62,220,158,ALPHA)", "ring");
      } else if (line.includes("replanning") && state.robot) {
        const [x, y] = cellCenter(...state.robot);
        spawnBurst(x, y, "rgba(70,199,255,ALPHA)", "dot");
      }
    });

    draw();

    const battery = state.battery;
    els.battery.textContent = battery;
    els.batfill.style.width = `${battery}%`;
    els.batfill.className = battery < 25 ? "low" : battery < 55 ? "mid" : "";
    els.carry.textContent = state.carrying ? `· carrying T${state.carrying}` : "";

    els.sT.textContent = state.t;
    els.sSteps.textContent = state.stats.steps;
    els.sReplans.textContent = state.stats.replans;
    els.sCharges.textContent = state.stats.charges;

    const pct = state.n_tasks ? Math.round((state.stats.completed / state.n_tasks) * 100) : 0;
    els.progfill.style.width = `${pct}%`;
    els.proglabel.textContent = `${state.stats.completed} of ${state.n_tasks} delivered`;

    if (state.ai) {
      const online = state.ai.status === "online";
      els.aiDot.className = `dot-live${online ? "" : " warn"}`;
      els.aiModel.textContent = `${state.ai.algorithm} · ${online ? "ONLINE" : "FALLBACK"}`;
      els.aiMeta.textContent = `${state.ai.features.length} learned features • hard battery + A* safety guards`;
      els.aiChoice.textContent = state.ai.choice === "charger"
        ? "CHARGER"
        : state.ai.choice ? `T${state.ai.choice}` : "—";
    }
    renderQueue(state.pending);
    renderCycle(state);
    if (state.log && state.log.length) appendLog(state.log);

    els.now.textContent = state.done
      ? "Run complete"
      : state.goal.kind
        ? `t=${state.t} — ${state.goal.kind === "pickup" ? "heading to pickup" : state.goal.kind === "dispatch" ? "delivering" : "heading to charge"}`
        : `t=${state.t} — deciding next move`;

    if (state.done) {
      stopPlaying();
      setPill("done");
      showOverlay(state);
    } else {
      setPill(timer ? "run" : "idle");
    }
  }

  function showOverlay(state) {
    els.resTitle.textContent = state.stats.completed >= state.n_tasks ? "All orders delivered" : "Run ended";
    els.resStats.innerHTML = `
      <div><dt>Delivered</dt><dd>${state.stats.completed}/${state.n_tasks}</dd></div>
      <div><dt>Ticks</dt><dd>${state.t}</dd></div>
      <div><dt>Distance</dt><dd>${state.stats.steps}</dd></div>
      <div><dt>Replans</dt><dd>${state.stats.replans}</dd></div>
      <div><dt>Charges</dt><dd>${state.stats.charges}</dd></div>
      <div><dt>Avg delay</dt><dd>${state.stats.avg_delay}</dd></div>`;
    els.overlay.hidden = false;
  }

  function hideOverlay() { els.overlay.hidden = true; }

  // ---------------------------------------------------------------- actions
  function readScenarioSettings() {
    const seed = Math.max(0, Math.min(9999, parseInt(els.seed.value, 10) || 1));
    const tasks = Math.max(1, Math.min(20, parseInt(els.tasks.value, 10) || 10));
    const obstacles = Math.max(0, Math.min(8, parseInt(els.obstacles.value, 10) || 0));
    els.seed.value = seed;
    els.tasks.value = tasks;
    els.obstacles.value = obstacles;
    return { seed, tasks, obstacles };
  }

  function updateScenarioStatus(config) {
    if (!els.scenarioStatus || !config) return;
    els.scenarioStatus.textContent = `Scenario: Seed ${config.seed} · ${config.tasks} orders · ${config.obstacles} obstacles`;
  }

  let settingsTimer = null;
  function scheduleScenarioReset() {
    clearTimeout(settingsTimer);
    setMsg("Applying scenario settings…");
    settingsTimer = setTimeout(() => doReset(), 300);
  }

  async function doReset() {
    const seq = ++resetSeq;
    stopPlaying();
    hideOverlay();
    setMsg("");
    els.log.innerHTML = "";
    animRobotPx = null; targetRobotPx = null; prevRobotCell = null; robotHeading = 0; particles = [];
    const settings = readScenarioSettings();
    const body = { sid, ...settings };
    try {
      const data = await api("/api/reset", body);
      if (seq !== resetSeq) return;
      sid = data.sid;
      layout = { grid: data.layout.grid, charger: data.layout.charger, dispatch: data.layout.dispatch,
                 rows: data.layout.grid.length, cols: data.layout.grid[0].length };
      render(data);
      setMsg("");
      setPill("idle");
    } catch (e) {
      if (seq === resetSeq) setMsg(e.message);
    }
  }

  async function doStep(n = 1) {
    if (!sid || busy) return;
    busy = true;
    try {
      const data = await api("/api/step", { sid, n });
      render(data);
    } catch (e) {
      setMsg(e.message);
      stopPlaying();
    } finally {
      busy = false;
    }
  }

  async function doFinish() {
    if (!sid) return;
    stopPlaying();
    setPill("run");
    setMsg("");
    while (latest && !latest.done) {
      // eslint-disable-next-line no-await-in-loop
      await doStep(25);
      if (!latest) break;
    }
  }

  function startPlaying() {
    if (timer || !sid) return;
    setPill("run");
    timer = setInterval(() => doStep(1), speedMs);
  }

  function stopPlaying() {
    if (timer) { clearInterval(timer); timer = null; }
    if (latest && !latest.done) setPill("idle");
  }

  function togglePlay() {
    if (timer) stopPlaying(); else startPlaying();
  }

  async function doRush() {
    if (!sid || busy) return;
    busy = true;
    try {
      const data = await api("/api/order", { sid });
      render(data);
    } catch (e) {
      setMsg(e.message);
    } finally {
      busy = false;
    }
  }

  async function onCanvasClick(evt) {
    if (!sid || !layout || busy) return;
    busy = true;
    const rect = canvas.getBoundingClientRect();
    const x = (evt.clientX - rect.left) * (canvas.width / rect.width);
    const y = (evt.clientY - rect.top) * (canvas.height / rect.height);
    const c = Math.floor(x / CELL());
    const r = Math.floor(y / CELLY());
    try {
      const data = await api("/api/obstacle", { sid, r, c });
      render(data);
    } catch (e) {
      setMsg(e.message);
    } finally {
      busy = false;
    }
  }

  async function loadGroqStatus() {
    try {
      const data = await api("/api/llm-status");
      const online = data.configured;
      els.groqDot.className = `dot-live${online ? "" : " warn"}`;
      els.groqStatus.textContent = online ? `Groq · ${data.model} · ONLINE` : "Groq · NOT CONFIGURED";
    } catch (e) {
      els.groqStatus.textContent = "Groq · unavailable";
    }
  }

  function addGroqMessage(role, text) {
    const div = document.createElement("div");
    div.className = `groq-msg ${role}`;
    div.textContent = text;
    els.groqChat.appendChild(div);
    els.groqChat.scrollTop = els.groqChat.scrollHeight;
  }

  async function askGroq() {
    const message = els.groqInput.value.trim();
    if (!message || !sid || busy) return;
    els.groqInput.value = "";
    addGroqMessage("user", message);
    els.groqSend.disabled = true;
    try {
      const data = await api("/api/groq", { sid, message });
      addGroqMessage("bot", data.reply);
    } catch (e) {
      addGroqMessage("bot", e.message);
    } finally {
      els.groqSend.disabled = false;
    }
  }

  // ---------------------------------------------------------------- wire up
  els.play.addEventListener("click", togglePlay);
  els.step.addEventListener("click", () => doStep(1));
  els.finish.addEventListener("click", doFinish);
  els.reset.addEventListener("click", doReset);
  els.again.addEventListener("click", doReset);
  els.rush.addEventListener("click", doRush);

  // Scenario controls are live: changing any value immediately rebuilds the
  // server-side world, so the map, orders and obstacles always match the UI.
  [els.seed, els.tasks, els.obstacles].forEach((input) => {
    input.addEventListener("input", scheduleScenarioReset);
    input.addEventListener("change", scheduleScenarioReset);
  });
  els.groqSend.addEventListener("click", askGroq);
  els.groqInput.addEventListener("keydown", (e) => { if (e.key === "Enter") askGroq(); });
  loadGroqStatus();
  canvas.addEventListener("click", onCanvasClick);

  document.querySelectorAll(".seg button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".seg button").forEach((b) => b.setAttribute("aria-pressed", "false"));
      btn.setAttribute("aria-pressed", "true");
      speedMs = parseInt(btn.dataset.speed, 10);
      if (timer) { stopPlaying(); startPlaying(); }
    });
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); togglePlay(); }
    else if (e.key === "s" || e.key === "S") doStep(1);
    else if (e.key === "r" || e.key === "R") doReset();
  });

  // initial run
  doReset();
  rafId = requestAnimationFrame(tickAnimation);
})();
