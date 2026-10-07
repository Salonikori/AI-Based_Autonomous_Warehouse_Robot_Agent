"""Groq LLM integration for the Warehouse Robot Agent.

Set GROQ_API_KEY in .env. The browser never receives the secret key; requests
are made server-side to Groq's OpenAI-compatible API.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


_PLACEHOLDER_KEYS = {"", "your_groq_api_key_here"}


class GroqAgent:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.enabled = self.api_key.lower() not in _PLACEHOLDER_KEYS

    def status(self):
        return {
            "provider": "Groq",
            "model": self.model,
            "status": "online" if self.enabled else "not_configured",
            "configured": self.enabled,
            "message": "API key is stored server-side." if self.enabled else "Set GROQ_API_KEY in .env",
        }

    def _system_prompt(self):
        return """You are the LLM reasoning and operations assistant for a warehouse robot simulation.
You are not the safety controller. A deterministic controller (battery constraints + A* path planning)
and a Random Forest policy control physical actions. Never claim to have moved the robot or changed
its state. Analyze the supplied live state, explain decisions clearly, identify risks, and suggest
safe next actions. If asked about the ML decision, use the ranked predictions and simulation data.
Keep responses concise and practical. Do not invent sensor values, orders, or actions not present in state."""

    def _context(self, state, agent, env):
        pending = []
        for task in env.pending:
            pending.append({
                "id": task.tid,
                "pickup": task.pickup,
                "drop": task.drop,
                "priority": task.priority,
                "created": task.created,
            })
        return {
            "tick": state["t"],
            "robot_position": agent.pos,
            "battery": agent.battery,
            "carrying": agent.carrying.tid if agent.carrying else None,
            "goal": {"position": agent.goal, "kind": agent.goal_kind},
            "pending_orders": pending,
            "dynamic_obstacles": env.dynamic,
            "blocked_cells": list(agent.blocked),
            "ml_decision": state["ai"],
            "stats": state["stats"],
            "recent_events": agent.log[-10:],
        }

    def chat(self, message, state, agent, env):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": json.dumps({
                    "question": message,
                    "live_warehouse_state": self._context(state, agent, env),
                })},
            ],
            "temperature": 0.2,
            "max_completion_tokens": 900,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Network error: {exc.reason}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("Groq returned no choices.")
        content = choices[0].get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return {"model": self.model, "reply": content.strip()}
