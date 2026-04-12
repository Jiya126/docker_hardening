<!-- ---
title: SCA-Gym Docker Hardening
emoji: 🐳
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
license: bsd-3-clause
short_description: Docker security hardening RL environment
tags:
  - openenv
  - reinforcement-learning
  - docker-security
  - vulnerability-remediation
--- -->

# 🐳 SCA-Gym — Docker Security Hardening Environment

An OpenEnv RL environment that evaluates AI agents on **fixing security vulnerabilities in Dockerfiles**.

The agent reads a broken Dockerfile and its scan report, then:
1. **Analyzes** the security issues — identifying what's wrong
2. **Patches** the Dockerfile — submitting a hardened version

> Every `reset()` generates a **unique** broken Dockerfile from a randomized pool of security issues, preventing memorization and forcing genuine reasoning.

---

## 🔄 Episode Flow

Each episode is **2 steps**, giving trajectory-level reward signal:

```
                    ┌──────────────────────────────────────┐
                    │         reset()                       │
                    │  → Broken Dockerfile + scan report    │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  Step 1: ANALYZE                      │
                    │  Agent identifies security issues     │
                    │  → Intermediate reward (detection     │
                    │    accuracy score)                     │
                    │  → Actionable feedback returned       │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  Step 2: PATCH                        │
                    │  Agent submits fixed Dockerfile       │
                    │  → Final reward (improvement score)   │
                    │  → Episode ends                       │
                    └──────────────────────────────────────┘
```

---

## 📋 Tasks

| Task | Difficulty | Issues Pool | Selected Per Episode | What It Tests |
|:-----|:----------:|:-----------:|:--------------------:|:--------------|
| `patch_easy` | 🟢 Easy | 9 | 4–6 | Pipe-to-shell, missing USER/HEALTHCHECK, no apt cleanup, exposed ports |
| `patch_medium` | 🟡 Medium | 14 | 7–10 | Secrets in ENV/ARG, ADD instead of COPY, pip vulns, database ports |
| `patch_hard` | 🔴 Hard | 22 | 10–14 | ADD from URL, Docker-in-Docker, credential URLs, base64 secrets, chmod 777 |

---

## 🎯 Action Space

### Step 1 — Analyze (identify issues)

```json
{
  "identified_issues": [
    "running as root — no USER instruction",
    "pipe-to-shell pattern (curl | bash)",
    "secrets exposed in ENV instructions"
  ],
  "identified_categories": ["best_practice", "supply_chain", "secret_exposure"]
}
```

**Valid categories:**
`supply_chain` · `best_practice` · `network_security` · `secret_exposure` · `vulnerability` · `permissions`

### Step 2 — Patch (fix the Dockerfile)

```json
{
  "patched_dockerfile": "FROM python:3.12-slim\nRUN apt-get update && apt-get upgrade -y..."
}
```

---

## 💰 Reward Function

**Reward range:** `[0.01, 0.99]`

**Analysis reward** — Weighted detection accuracy:
- 60% issue identification accuracy
- 40% category identification accuracy
- Negation guard rejects dismissive claims (e.g. "not an issue")

**Patch reward** — Weighted improvement across three dimensions:

| Grader | Vuln Reduction | Best Practices | Anti-Patterns | Penalties |
|:-------|:--------------:|:--------------:|:-------------:|:----------|
| **Easy** | 50% | 30% | 20% | — |
| **Medium** | 40% | 35% | 25% | −0.10 if secrets remain |
| **Hard** | 35% | 30% | 35% | −0.15 secrets, −0.08 per regression |

**Final episode score** = `0.3 × analysis + 0.7 × patch`

### Randomization

Each `reset()` randomly selects issues from the pool, composes a unique Dockerfile with randomized secret values, URLs, and ports, and samples a fresh CVE subset. No two episodes are the same.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|:------:|:---------|:------------|
| `GET` | `/health` | Returns `{"status": "healthy"}` |
| `GET` | `/spec` | Environment metadata + full action schemas |
| `POST` | `/reset` | Start new episode → returns Dockerfile + scan report |
| `POST` | `/step` | Analyze (step 1) or Patch (step 2) → returns observation |
| `GET` | `/state` | Current episode state, active issues, analysis status |
| `GET` | `/leaderboard` | Top scores across episodes |

---

## 🚀 Getting Started

### 1. Set environment variables

```bash
export HF_TOKEN="your-huggingface-token"
export API_BASE_URL="https://router.huggingface.co/v1"   # optional
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"            # optional
```

### 2. Run locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run with Docker

```bash
docker build -t sca-gym .
docker run -p 7860:7860 -e HF_TOKEN=$HF_TOKEN sca-gym
```

### 4. Run inference

```bash
python inference.py
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

### 6. Validate before submission

```bash
# Start server first, then in a second terminal:
python validate.py
```

---

## 📁 Project Structure

```
├── Dockerfile                               # Container for HF Spaces (port 7860)
├── inference.py                             # Baseline 2-step agent (analyze → patch)
├── validate.py                              # Pre-submission validation (6 checks)
├── openenv.yaml                             # OpenEnv spec with action schemas
├── models.py                                # Pydantic Action / Observation / State
├── requirements.txt
│
├── server/
│   ├── app.py                               # FastAPI server with action normalization
│   └── docker_hardening_environment.py      # Core env: randomization + 2-step episodes
│
├── tasks/
│   └── generators.py                        # Dynamic Dockerfile composition from issue pools
│
├── graders/
│   ├── easy_grader.py                       # 50/30/20 vuln/bp/ap scoring
│   ├── medium_grader.py                     # + secret leak penalty
│   ├── hard_grader.py                       # + regression penalty
│   └── analysis_grader.py                   # Detection accuracy scoring with negation guard
│
├── tools/
│   ├── scanner.py                           # Mock CVE scanner + best-practice checks
│   └── docker_manager.py                    # Mock Docker build manager
│
└── tests/                                   # 79 tests (all pass offline, no Docker needed)
    ├── test_environment.py
    └── test_graders.py
```

---

## 📤 Output Format

`inference.py` outputs to stdout in the required format:

```
[START] task=patch_easy env=docker_hardening model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=analyze(4 issues: running as root, pipe-to-shell, ...) reward=0.6500 done=false error=null
[STEP] step=2 action=patch(0 vulns remaining) reward=0.8200 done=true error=null
[END] success=true steps=2 score=0.7690 rewards=0.6500,0.8200
```
