"""
SHL Conversational Assessment Recommender
FastAPI — /health + /chat
LLM: OpenRouter (meta-llama/llama-3.3-70b-instruct:free) — completely free
"""

import json, os, re
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Catalog ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_URLS = {item["url"] for item in CATALOG}
CATALOG_BY_URL = {item["url"]: item for item in CATALOG}
CATALOG_BY_NAME = {item["name"].lower(): item for item in CATALOG}
with open(os.path.join(BASE_DIR, "catalog.json")) as f:
    CATALOG = json.load(f)

def build_compact_catalog(catalog):
    lines = []
    for item in catalog:
        types = ",".join(item.get("test_types", []))
        levels = "|".join(item.get("job_levels", [])[:3])
        desc = item.get("description", "")[:120]
        lines.append(
            f'[{item["name"]}] type={types} dur={item.get("duration","?")} '
            f'levels={levels} url={item["url"]} | {desc}'
        )
    return "\n".join(lines)

CATALOG_COMPACT = build_compact_catalog(CATALOG)

# ── OpenRouter client ─────────────────────────────────────────────────────────
OR_KEY = os.environ.get("OPENROUTER_API_KEY", "")
if not OR_KEY:
    raise ValueError("OPENROUTER_API_KEY not set. Add it to your .env file.")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OR_KEY,
)

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are an expert SHL assessment consultant helping HR professionals pick the right assessments.

SCOPE: Only recommend assessments from the SHL catalog below. Refuse off-topic requests politely.

## SHL CATALOG
{CATALOG_COMPACT}

## RULES
- CLARIFY: If query is vague, ask exactly ONE question. Do NOT recommend on the first turn for vague queries.
- RECOMMEND: Once you have role + seniority + purpose, return 1-10 assessments from catalog only. Never invent URLs.
- REFINE: If user changes requirements, update the list in place.
- COMPARE: Answer comparison questions using catalog info only.
- CLOSE: When user confirms the list is final, set end_of_conversation=true.

## OUTPUT
Respond ONLY with valid JSON — no markdown, no explanation outside JSON:

{{"reply":"...", "recommendations":null, "end_of_conversation":false}}

When recommending:

{{"reply":"...", "recommendations":[{{"name":"...","url":"https://www.shl.com/...","test_type":"K","duration":"X min","remote_testing":true,"adaptive_irt":false,"description":"why it fits"}}], "end_of_conversation":false}}

test_type codes: A=Ability B=Biodata/SJT C=Competencies D=Development K=Knowledge P=Personality S=Simulations

- recommendations=null while clarifying
- Once recommendations given, always repeat full list in every reply
- end_of_conversation=true only when user explicitly confirms done
"""

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="SHL Assessment Recommender", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str
    duration: str
    remote_testing: bool
    adaptive_irt: bool
    description: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: Optional[list[Recommendation]]
    end_of_conversation: bool

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    if len(request.messages) > 8:
        request.messages = request.messages[-8:]

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in request.messages:
        api_messages.append({"role": m.role, "content": m.content})

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-3.3-70b-instruct:free",
            messages=api_messages,
            temperature=0.2,
            max_tokens=1024,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM API error: {str(e)}")

    # Strip markdown fences
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except:
                parsed = {"reply": raw_text, "recommendations": None, "end_of_conversation": False}
        else:
            parsed = {"reply": raw_text, "recommendations": None, "end_of_conversation": False}

    # Validate — only allow catalog URLs
    recs = parsed.get("recommendations")
    validated = None
    if recs is not None:
        validated = []
        for rec in recs[:10]:
            url = rec.get("url", "")
            if url not in CATALOG_URLS:
               hit = CATALOG_BY_NAME.get(rec.get("name", "").lower())
               if hit:
                    url = hit["url"]
               else:
                    rec_name_lower = rec.get("name", "").lower()
                    hit = next(
                        (c for c in CATALOG if rec_name_lower in c["name"].lower()
                         or c["name"].lower() in rec_name_lower),
                         None
                    )
                    if hit:
                        url = hit["url"]
                    else:
                         continue
        cat = CATALOG_BY_URL.get(url)
        if not cat:
           continue
        validated.append(Recommendation(
            name=cat["name"],  # canonical name from catalog
                test_type=rec.get("test_type", cat["test_types"][0] if cat["test_types"] else "K"),
                duration=rec.get("duration", cat["duration"]),
                remote_testing=cat["remote"],
                adaptive_irt=cat["adaptive"],
                description=rec.get("description", cat["description"][:150]),
            ))

    return ChatResponse(
        reply=parsed.get("reply", ""),
        recommendations=validated,
        end_of_conversation=bool(parsed.get("end_of_conversation", False)),
    )
