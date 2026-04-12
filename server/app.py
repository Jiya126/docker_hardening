"""
FastAPI server for the Docker Hardening environment.

Custom server (not using create_app) for full control over endpoints,
action normalization, and the 2-step analyze+patch flow.
"""

import json
import os
import sys
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

try:
    from .docker_hardening_environment import DockerHardeningEnvironment, TASKS
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from server.docker_hardening_environment import DockerHardeningEnvironment, TASKS

try:
    from ..models import DockerHardeningAction, DockerHardeningObservation
except ImportError:
    from models import DockerHardeningAction, DockerHardeningObservation


_envs: Dict[str, DockerHardeningEnvironment] = {}
_leaderboard: list = []


@asynccontextmanager
async def lifespan(application: FastAPI):
    for task_name in TASKS:
        env = DockerHardeningEnvironment()
        _envs[task_name] = env
    yield
    for env in _envs.values():
        if hasattr(env, "_build_mgr") and env._build_mgr:
            env._build_mgr.cleanup()


app = FastAPI(
    title="SCA-Gym — Docker Hardening Environment",
    version="0.2.0",
    lifespan=lifespan,
)


# ── Action normalization ─────────────────────────────────────────────────────

def _normalize_action(raw: Any) -> Dict:
    """
    Coerce any incoming action into a valid format.
    Handles missing fields, wrong types, freeform text, etc.
    """
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.startswith("{"):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return {"patched_dockerfile": raw}
        else:
            return {"patched_dockerfile": raw}

    if isinstance(raw, dict):
        if "patched_dockerfile" in raw:
            return {"patched_dockerfile": str(raw["patched_dockerfile"])}

        if "identified_issues" in raw or "identified_categories" in raw:
            return {"patched_dockerfile": json.dumps(raw)}

        if "action" in raw:
            inner = raw["action"]
            if isinstance(inner, str):
                return {"patched_dockerfile": inner}
            if isinstance(inner, dict):
                return _normalize_action(inner)

        if "dockerfile" in raw:
            return {"patched_dockerfile": str(raw["dockerfile"])}

        return {"patched_dockerfile": json.dumps(raw)}

    return {"patched_dockerfile": str(raw) if raw else ""}


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "SCA-Gym",
        "env": "docker_hardening",
        "version": "0.2.0",
        "status": "running",
        "tasks": list(TASKS.keys()),
        "endpoints": ["/health", "/spec", "/reset", "/step", "/state", "/leaderboard"],
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "loaded_tasks": list(_envs.keys()),
        "num_tasks": len(_envs),
    }


@app.get("/spec")
async def spec():
    return {
        "env_name": "docker_hardening",
        "version": "0.2.0",
        "observation_type": "text",
        "action_type": "json",
        "reward_range": [0.01, 0.99],
        "episode_steps": 2,
        "step_descriptions": {
            "1": "Analyze — identify security issues (JSON with identified_issues + identified_categories)",
            "2": "Patch — submit fixed Dockerfile (patched_dockerfile field)",
        },
        "action_schema": {
            "analyze": {
                "type": "object",
                "properties": {
                    "identified_issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Free-text descriptions of security issues found",
                    },
                    "identified_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "enum": [
                            "supply_chain", "best_practice", "network_security",
                            "secret_exposure", "vulnerability", "permissions",
                        ],
                        "description": "Categories of issues found",
                    },
                },
            },
            "patch": {
                "type": "object",
                "properties": {
                    "patched_dockerfile": {
                        "type": "string",
                        "description": "The full text of the patched Dockerfile",
                    },
                },
                "required": ["patched_dockerfile"],
            },
        },
        "randomization": "Each reset() generates a unique Dockerfile from randomized issue pool",
        "tasks": {
            name: {
                "difficulty": t["difficulty"],
                "max_steps": t["max_steps"],
            }
            for name, t in TASKS.items()
        },
    }


class ResetRequest(BaseModel):
    task_name: Optional[str] = None


@app.get("/reset")
async def reset_get(task_name: Optional[str] = None):
    return await _do_reset(task_name)


@app.post("/reset")
async def reset_post(req: Request):
    try:
        body = await req.json()
        task_name = body.get("task_name") or body.get("task")
    except Exception:
        task_name = None
    return await _do_reset(task_name)


async def _do_reset(task_name: Optional[str] = None):
    task_name = task_name or os.environ.get("SCA_GYM_TASK", "patch_easy")
    if task_name not in _envs:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown task '{task_name}'. Valid: {list(TASKS.keys())}"},
        )

    env = _envs[task_name]
    os.environ["SCA_GYM_TASK"] = task_name
    obs = env.reset()
    return obs.model_dump()


class StepRequest(BaseModel):
    action: Any = None
    patched_dockerfile: Optional[str] = None
    task_name: Optional[str] = None


@app.post("/step")
async def step(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

    task_name = (
        body.get("task_name")
        or body.get("task")
        or os.environ.get("SCA_GYM_TASK", "patch_easy")
    )
    if task_name not in _envs:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown task '{task_name}'"},
        )

    raw_action = body.get("action") or body
    normalized = _normalize_action(raw_action)

    try:
        action = DockerHardeningAction(**normalized)
    except Exception as e:
        action = DockerHardeningAction(patched_dockerfile=str(raw_action))

    env = _envs[task_name]
    try:
        obs = env.step(action)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Environment error: {str(e)[:300]}"},
        )

    result = obs.model_dump()

    if obs.done:
        _leaderboard.append({
            "task": task_name,
            "score": obs.score,
            "reward": obs.reward,
            "steps": obs.step_number,
            "timestamp": time.time(),
        })

    return result


@app.get("/state")
async def state(task_name: Optional[str] = None):
    task_name = task_name or os.environ.get("SCA_GYM_TASK", "patch_easy")
    if task_name not in _envs:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown task '{task_name}'"},
        )
    env = _envs[task_name]
    s = env.state
    return {
        **s.model_dump(),
        "episode_seed": env._episode_seed,
        "active_issues": sorted(env._active_issues),
        "analysis_completed": env._analysis_completed,
    }


@app.get("/leaderboard")
async def leaderboard():
    return sorted(_leaderboard, key=lambda x: x["score"], reverse=True)[:50]


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
