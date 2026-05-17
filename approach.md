# Approach Document — SHL Conversational Assessment Recommender

## Design Overview

### Problem Decomposition

The core challenge is bridging two gaps: (1) users who know what job they are hiring for but not what assessment vocabulary to use, and (2) a catalog of assessments that requires domain knowledge to navigate. I treated this as a **context-accumulation agent** problem: the agent must gather enough signal to commit to a grounded shortlist, and must not recommend anything outside the catalog.

### Architecture

**Single-service FastAPI app** with two endpoints:
- `GET /health` — readiness probe
- `POST /chat` — stateless; full conversation history sent on every call

**No database, no vector store.** The entire catalog (87 Individual Test Solutions) is loaded into a JSON file at startup and injected directly into the system prompt. This avoids retrieval latency and fits within Gemini's context window. The tradeoff is that scaling to the full 300+ catalog would require chunked retrieval, but for this scope direct injection gives perfect recall coverage.

### Retrieval Strategy

Full-catalog prompt injection rather than semantic search. Every call sees the complete catalog in the system prompt. This guarantees zero hallucination on catalog URLs, since a secondary URL validator runs on every response: it checks each recommended URL against the catalog and drops any invented entries.

### Agent Design

The agent follows four behavioral states defined via the system prompt:

1. **CLARIFY** — vague query triggers ONE clarifying question. Never recommends on turn 1 for vague intent. Key dimensions: role, seniority, purpose (selection vs development), language.
2. **RECOMMEND** — once context is sufficient, returns 1–10 assessments with catalog URLs, test type codes, and a brief fit rationale.
3. **REFINE** — on constraint changes mid-conversation, updates the shortlist in-place rather than restarting.
4. **COMPARE** — answers product comparison questions grounded in catalog descriptions; holds recommendations steady.

State transitions are implicit — the LLM infers state from conversation history, guided by explicit behavioral rules in the system prompt.

### Prompt Design

Three layers:
- **Scope guard** — explicitly lists what the agent refuses (legal advice, general hiring, non-SHL products, prompt injections)
- **Behavioral rules** — CLARIFY/RECOMMEND/REFINE/COMPARE patterns with examples
- **Schema contract** — the exact JSON schema the model must return, with field-level documentation

### Schema Compliance

The response schema is enforced at two levels:
1. The system prompt defines the schema and states deviating breaks the evaluator
2. The application layer validates the parsed JSON, fills defaults from catalog data, and drops any recommendations not in the catalog

`end_of_conversation` is set to `true` only when the user explicitly confirms the shortlist.

### What Did Not Work

- **Asking multiple clarifying questions at once** — fixed by the "ONE question per turn" rule in the prompt.
- **Letting the model write its own URLs** — early versions generated plausible-looking but wrong URLs. Fixed by the URL validator that re-derives URLs from the catalog.
- **recommendations: null vs []** — the evaluator treats these differently. The prompt explicitly distinguishes: `null` for clarification turns, array for committed shortlists.

### Evaluation Approach

The evaluator replays all 10 provided conversation traces and reports:
- **Schema compliance** — correct JSON keys on every turn, turn cap honored, catalog-only URLs
- **Recall@10** — fraction of expected assessments appearing in the final shortlist
- **EOC accuracy** — `end_of_conversation` correct on the final turn

### Tools Used

- **Gemini 2.0 Flash** via Google Generative AI Python SDK — reasoning, classification, generation
- **FastAPI + Uvicorn** — API layer
- **Pydantic v2** — request/response validation
- No vector store, no embeddings, no external search

### Trade-offs and Future Work

- Full 300+ catalog would require chunked retrieval (FAISS or pgvector) to avoid exceeding context limits
- Adding a structured slot-filler (role, level, purpose) would make state transitions more deterministic
- Caching frequent queries would reduce latency and API costs
