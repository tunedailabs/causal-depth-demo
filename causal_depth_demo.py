#!/usr/bin/env python3
"""
Causal Depth Analyzer — API backend
FastAPI backend — serves the interactive depth analysis demo.

  python3 laziness_api.py

Endpoints:
  GET  /              → serves the demo HTML
  POST /analyze       → runs question against base + tuned, returns depth scores
  GET  /health        → checks model availability
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import time
import os
import uvicorn
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
from openai import OpenAI

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

BASE_MODEL_ID  = "gpt-4o-mini"
TUNED_MODEL_ID = "gpt-4o-mini"

# If a Colab/hosted tuned model is running, set TUNED_MODEL_URL to its public URL
# e.g. https://xxxx.ngrok-free.app  — overrides OpenAI for the tuned side
# Default: local MLX server on port 9001
TUNED_MODEL_URL = os.environ.get("TUNED_MODEL_URL", "").rstrip("/")

# ── Tier detection ─────────────────────────────────────────────────────────────

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

TIER_LABELS = {
    "T1": "Tier 1 — Observation",
    "T2": "Tier 2 — Mechanism",
    "T3": "Tier 3 — Projection",
    "T4": "Tier 4 — Simulation",
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
                    # Check the matching line plus the next 2 lines as a window.
                    # This handles "### TIER 1 (Observation)" header-on-own-line
                    # format where content follows on the next line.
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

# ── System prompts ────────────────────────────────────────────────────────────

# Base model gets the full Rungs framework injected as a system prompt.
# This is the "prompt engineering" approach — you have to tell it every time.
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

# Tuned model needs no system prompt — TIER 1-4 output is in the weights (v4)

# ── Case context ─────────────────────────────────────────────────────────────

CASE_BRIEF = """\
Case: State v. Howell — Yamhill County, Oregon — CR100629
Defendant: Jennifer Clare Howell
Charges:
(1) Mfg/Deliver Controlled Substance within 1,000ft of school — Felony A (ORS 475.904)
(2) Possession of Meth — Felony C (dismissed)
(3) Endanger Welfare of Minor — Misdemeanor A
Disposition: Guilty plea 12/27/2010. Counts 1 & 3 convicted, Count 2 dismissed.
Sentence: 34 months Oregon DOC, 3 years post-prison supervision, license revoked 6 months.
Timeline: Arraigned 10/28/2010, Bail $75,000, Release denied 11/16/2010, \
60-day rule waived 11/30/2010, Plea entered ~60 days after arraignment.\
"""

def build_question_with_context(question: str) -> str:
    return f"Case context:\n{CASE_BRIEF}\n\nQuestion: {question}"

# ── Model calls ───────────────────────────────────────────────────────────────

def call_base(question: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=BASE_MODEL_ID,
            messages=[
                {"role": "system", "content": BASE_SYSTEM},
                {"role": "user", "content": build_question_with_context(question)},
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip(), time.time() - t0
    except Exception as e:
        return f"[Error: {e}]", 0.0

def call_tuned(question: str) -> tuple[str, float]:
    t0 = time.time()
    try:
        if TUNED_MODEL_URL:
            # Use real tuned Qwen model hosted on Colab/GPU
            import requests as req_lib
            resp = req_lib.post(
                f"{TUNED_MODEL_URL}/v1/chat/completions",
                json={
                    "model": "/tmp/tunedai-mlx-q4",
                    "messages": [{"role": "user", "content": build_question_with_context(question)}],
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
                timeout=120,
            )
            text = resp.json()["choices"][0]["message"]["content"].strip()
        else:
            # Fallback: OpenAI, no system prompt (mirrors tuned model behavior)
            resp = client.chat.completions.create(
                model=TUNED_MODEL_ID,
                messages=[
                    {"role": "user", "content": build_question_with_context(question)},
                ],
                temperature=0.3,
                max_tokens=800,
            )
            text = resp.choices[0].message.content.strip()
        return text, time.time() - t0
    except Exception as e:
        return f"[Error: {e}]", 0.0

# ── API routes ────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    question: str
    required_tiers: list = ["T1", "T2", "T4"]

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    q = req.question.strip()
    if not q:
        return JSONResponse({"error": "empty question"}, status_code=400)

    base_text, base_time = call_base(q)
    tuned_text, tuned_time = call_tuned(q)

    base_tiers  = detect_tiers(base_text)
    tuned_tiers = detect_tiers(tuned_text)

    base_score  = depth_score(base_tiers,  req.required_tiers)
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
    }

@app.get("/health")
async def health():
    try:
        client.models.list()
        api_ok = True
    except:
        api_ok = False
    return {"base_model": api_ok, "tuned_model": api_ok, "backend": "openai"}

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("causal_depth_demo.html") as f:
        return f.read()

if __name__ == "__main__":
    print("Causal Depth Analyzer")
    print("Open: http://localhost:9000")
    print()
    uvicorn.run(app, host="0.0.0.0", port=9000)
