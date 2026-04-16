#!/usr/bin/env python3
"""
Causal Depth Analyzer — Production API backend
FastAPI backend — serves the interactive depth analysis demo.

Endpoints:
  GET  /              → serves the demo HTML (requires auth)
  POST /analyze       → runs question against base + tuned, returns depth scores
  GET  /health        → checks model availability
  POST /auth/verify   → password gate, returns session token
  POST /admin/set-tunnel → update tunnel URL at runtime
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import time
import os
import uvicorn
from openai import OpenAI
from secrets import token_urlsafe
import requests as req_lib

app = FastAPI()

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BASE_MODEL_ID = "gpt-4o-mini"

# Tunnel URL — mutable at runtime via /admin/set-tunnel
_tunnel_url = os.environ.get("TUNED_MODEL_URL", "").rstrip("/")

# Auth tokens — in-memory store (resets on restart, fine for a demo)
_active_tokens: dict[str, float] = {}  # token -> expiry timestamp
DEMO_PASSWORD = os.environ.get("DEMO_PASSWORD", "rungs2026")
ADMIN_SECRET = os.environ.get("ADMIN_SECRET", "")

# Tunnel health cache
_tunnel_cache = {"ok": False, "checked_at": 0.0, "last_error": ""}

# Rate limiting — simple per-token counter
_rate_limits: dict[str, list[float]] = {}  # token -> list of timestamps
RATE_LIMIT_PER_HOUR = 20

# ── Auth helpers ──────────────────────────────────────────────────────────────

def verify_token(request: Request) -> str:
    token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not token or token not in _active_tokens:
        raise HTTPException(401, "unauthorized — please enter the demo password")
    if _active_tokens[token] < time.time():
        del _active_tokens[token]
        raise HTTPException(401, "session expired — please re-enter the demo password")
    return token

def check_rate_limit(token: str):
    now = time.time()
    if token not in _rate_limits:
        _rate_limits[token] = []
    # Prune old entries
    _rate_limits[token] = [t for t in _rate_limits[token] if now - t < 3600]
    if len(_rate_limits[token]) >= RATE_LIMIT_PER_HOUR:
        raise HTTPException(429, f"rate limit exceeded — max {RATE_LIMIT_PER_HOUR} analyses per hour")
    _rate_limits[token].append(now)

# ── Tunnel helpers ────────────────────────────────────────────────────────────

def get_tunnel_url() -> str:
    return _tunnel_url

def tunnel_is_up() -> bool:
    now = time.time()
    if now - _tunnel_cache["checked_at"] < 30:
        return _tunnel_cache["ok"]
    url = get_tunnel_url()
    if not url:
        _tunnel_cache.update(ok=False, checked_at=now, last_error="no tunnel configured")
        return False
    try:
        r = req_lib.get(f"{url}/health", timeout=5)
        ok = r.status_code == 200
        _tunnel_cache.update(ok=ok, checked_at=now, last_error="" if ok else f"status {r.status_code}")
    except Exception as e:
        _tunnel_cache.update(ok=False, checked_at=now, last_error=str(e)[:100])
    return _tunnel_cache["ok"]

# ── Tier detection ────────────────────────────────────────────────────────────

TIER_PATTERNS = {
    "T1": [
        r"tier\s*1", r"observat", r"correlat", r"pattern",
        r"conditional probability", r"we (see|observe|notice)",
        r"data show", r"statistic", r"in the (data|record|case)",
    ],
    "T2": [
        r"tier\s*2", r"mechanism", r"causal", r"what (would|will) happen",
        r"if (we|the|it) (change|set|force|alter|modif)",
        r"direct (cause|effect|impact)", r"would (change|differ|result)",
        r"because of", r"leads? to", r"drives?",
    ],
    "T3": [
        r"tier\s*3", r"project", r"anticipat", r"expect.*effect",
        r"before (the|it|this)", r"forecast", r"likely.*effect",
        r"probable.*outcome", r"predict.*impact", r"forward.{0,20}model",
    ],
    "T4": [
        r"tier\s*4", r"simulat", r"what if", r"had.*not", r"would have",
        r"if.*instead", r"alternate", r"in a world", r"scenario",
        r"most likely.*happened", r"never (taken|done|happened)",
    ],
}

def detect_tiers(text: str) -> dict:
    text_lower = text.lower()
    results = {}
    for tier, patterns in TIER_PATTERNS.items():
        matches = []
        for pat in patterns:
            found = re.findall(pat, text_lower)
            matches.extend(found)
        present = len(matches) >= 1
        substantive = False
        if present:
            lines = text.split('\n')
            for i, para in enumerate(lines):
                para_lower = para.lower()
                if any(re.search(pat, para_lower) for pat in patterns):
                    window = ' '.join(lines[i:i+3])
                    if len(window.split()) >= 15:
                        substantive = True
                        break
        results[tier] = {
            "present": present,
            "substantive": substantive,
            "hits": len(matches),
        }
    return results

def depth_score(tier_results: dict, required_tiers: list) -> dict:
    if not required_tiers:
        return {"depth_score": 100, "completed": 0, "required": 0}
    completed = sum(
        1 for t in required_tiers
        if tier_results.get(t, {}).get("substantive", False)
    )
    score = round((completed / len(required_tiers)) * 100)
    return {
        "depth_score": score,
        "completed": completed,
        "required": len(required_tiers),
        "by_tier": {
            t: tier_results.get(t, {}).get("substantive", False)
            for t in required_tiers
        }
    }

# ── System prompt ─────────────────────────────────────────────────────────────

BASE_SYSTEM = """\
You are a causal analyst using the Rungs framework. For every question you MUST \
structure your answer across all four tiers — do not skip any tier:

TIER 1 — Observation: What the data, records, or facts directly show. \
Correlations, patterns, documented outcomes.

TIER 2 — Mechanism: The causal pathway. How and why did this happen. \
What forces, incentives, or processes drove the outcome.

TIER 3 — Projection: Given the mechanism, what are the expected forward effects. \
Probable outcomes, downstream consequences.

TIER 4 — Simulation: Counterfactual analysis. What would have happened under \
different conditions. Alternate scenarios and their likely outcomes.

Label each tier clearly. Be specific and analytical, not generic.\
"""

# ── Model calls ───────────────────────────────────────────────────────────────

def call_base(question: str, use_prompt: bool = True) -> tuple[str, float]:
    t0 = time.time()
    try:
        messages = []
        if use_prompt:
            messages.append({"role": "system", "content": BASE_SYSTEM})
        messages.append({"role": "user", "content": question})
        resp = client.chat.completions.create(
            model=BASE_MODEL_ID,
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip(), time.time() - t0
    except Exception as e:
        return f"[Error: {e}]", 0.0

def call_tuned(question: str) -> tuple[str, float]:
    t0 = time.time()
    url = get_tunnel_url()
    if url:
        # Quick probe first
        if not tunnel_is_up():
            return "[RungsX model offline — Colab tunnel not active. Start the Colab notebook to enable.]", 0.0
        try:
            resp = req_lib.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "rungsx-tuned",
                    "messages": [{"role": "user", "content": question}],
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
                timeout=90,
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
            return text, time.time() - t0
        except Exception as e:
            # Invalidate cache on failure
            _tunnel_cache.update(ok=False, checked_at=time.time(), last_error=str(e)[:100])
            return f"[RungsX model error: {str(e)[:80]}]", 0.0
    else:
        return "[RungsX model not configured — no tunnel URL set]", 0.0

# ── API routes ────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    password: str

class AnalyzeRequest(BaseModel):
    question: str
    required_tiers: list = ["T1", "T2", "T4"]
    use_prompt: bool = True

class TunnelRequest(BaseModel):
    url: str

@app.post("/auth/verify")
async def auth_verify(req: AuthRequest):
    if req.password != DEMO_PASSWORD:
        return JSONResponse({"error": "wrong password"}, status_code=401)
    token = token_urlsafe(32)
    _active_tokens[token] = time.time() + 86400  # 24 hours
    return {"token": token}

@app.post("/admin/set-tunnel")
async def set_tunnel(request: Request, req: TunnelRequest):
    global _tunnel_url
    secret = request.headers.get("X-Admin-Secret", "")
    if not ADMIN_SECRET or secret != ADMIN_SECRET:
        raise HTTPException(403, "forbidden")
    _tunnel_url = req.url.rstrip("/")
    # Invalidate cache so next health check probes the new URL
    _tunnel_cache.update(ok=False, checked_at=0)
    up = tunnel_is_up()
    return {"tunnel_url": _tunnel_url, "reachable": up}

@app.post("/analyze")
async def analyze(request: Request, req: AnalyzeRequest):
    token = verify_token(request)
    check_rate_limit(token)

    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "empty question"}, status_code=400)

    base_text, base_time = call_base(q, use_prompt=req.use_prompt)
    tuned_text, tuned_time = call_tuned(q)

    base_tiers = detect_tiers(base_text)
    tuned_tiers = detect_tiers(tuned_text)

    base_score = depth_score(base_tiers, req.required_tiers)
    tuned_score = depth_score(tuned_tiers, req.required_tiers)

    return {
        "question": q,
        "base": {
            "text": base_text,
            "time": round(base_time, 1),
            "tiers": base_tiers,
            "score": base_score,
        },
        "tuned": {
            "text": tuned_text,
            "time": round(tuned_time, 1),
            "tiers": tuned_tiers,
            "score": tuned_score,
        },
        "depth_delta": tuned_score["depth_score"] - base_score["depth_score"],
        "use_prompt": req.use_prompt,
    }

@app.get("/health")
async def health():
    # Check OpenAI
    try:
        client.models.list()
        base_ok = True
    except:
        base_ok = False

    tuned_ok = tunnel_is_up()

    return {
        "base_model": base_ok,
        "tuned_model": tuned_ok,
        "tunnel_url": bool(get_tunnel_url()),
        "tunnel_error": _tunnel_cache.get("last_error", "") if not tuned_ok else "",
    }

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("causal_depth_demo.html") as f:
        return f.read()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 9000))
    print("Causal Depth Analyzer")
    print(f"Open: http://localhost:{port}")
    print()
    uvicorn.run(app, host="0.0.0.0", port=port)
