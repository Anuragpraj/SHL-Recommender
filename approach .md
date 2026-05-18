# Approach Document --- SHL Conversational Assessment Recommender

## Design Overview

### Problem Decomposition

The core challenge is bridging two gaps: (1) users who know what job
they are hiring for but not what assessment vocabulary to use, and (2) a
catalog of assessments that requires domain knowledge to navigate. I
treated this as a **context-accumulation agent** problem: the agent must
gather enough signal to commit to a grounded shortlist, and must not
recommend anything outside the catalog.

### Architecture

**Single-service FastAPI app** with two endpoints:
- `GET /health` --- readiness probe, returns `{"status": "ok"}`
- `POST /chat` --- stateless; full conversation history sent on every call

**No database, no vector store.** The entire catalog (377 Individual Test
Solutions) is loaded from a JSON file at startup and injected directly
into the system prompt. This avoids retrieval latency and fits within
the model's context window. The tradeoff is that scaling to the full
300+ catalog would require chunked retrieval, but for this scope direct
injection gives perfect recall coverage with zero retrieval errors.

### Retrieval Strategy

Full-catalog prompt injection rather than semantic search. Every call
sees the complete catalog in the system prompt. This guarantees zero
hallucination on catalog URLs, since a secondary URL validator runs on
every response: it checks each recommended URL and name against the
catalog and drops any invented entries. Name-based fuzzy fallback
(case-insensitive match) covers cases where the model gets the name
right but the URL slightly wrong.

### Agent Design

The agent follows four behavioral states defined via the system prompt:

1.  **CLARIFY** --- vague query triggers exactly ONE clarifying
    question. Never recommends on turn 1 for vague intent. Key
    dimensions: role, seniority, purpose (selection vs development),
    language.
2.  **RECOMMEND** --- once context is sufficient, returns 1--10
    assessments with catalog URLs, test type codes, and a brief fit
    rationale. Repeats the full list on every subsequent turn.
3.  **REFINE** --- on constraint changes mid-conversation (add/drop
    items), updates the shortlist in-place rather than restarting. Full
    updated list always repeated.
4.  **COMPARE** --- answers product comparison questions grounded in
    catalog descriptions; holds recommendations steady (does not drop to
    null if a shortlist is already committed).

State transitions are implicit --- the LLM infers state from
conversation history, guided by explicit behavioral rules in the system
prompt.

### Prompt Design

Three layers:
- **Scope guard** --- explicitly lists what the agent
  refuses (legal advice, general hiring, non-SHL products, prompt
  injections)
- **Behavioral rules** ---
  CLARIFY/RECOMMEND/REFINE/COMPARE/CLOSE patterns with explicit trigger
  words for each state
- **Schema contract** --- the exact JSON schema the
  model must return, with field-level rules (`null` vs array, never `[]`,
  max 10 items, repeat full list every turn)

### Schema Compliance

The response schema is enforced at two levels:
1. The system prompt defines the schema and states deviating breaks the evaluator
2. The application layer validates the parsed JSON, derives URLs and metadata
   from catalog data, and drops any recommendations not found in the catalog

`end_of_conversation` is set to `true` only when the user explicitly
confirms (words: "confirmed", "perfect", "done", "locking it in", etc.).

### Code-Level Safeguards

- **Turn cap**: Maximum 8 turns (4 user + 4 assistant) enforced in code
- **Empty array guard**: `[]` is automatically converted to `null`
- **Model fallback**: Multi-model fallback (deepseek, llama-4, qwen3, openrouter/free)
  ensures availability even if primary model is down
- **Graceful degradation**: If all LLM calls fail, returns a polite retry
  message instead of crashing

### What Did Not Work

-   **Asking multiple clarifying questions at once** --- fixed by the
    "exactly ONE question per turn" rule in the prompt.
-   **Letting the model write its own URLs** --- early versions
    generated plausible-looking but wrong URLs. Fixed by the URL
    validator that re-derives URLs from the catalog, with name-based
    fallback.
-   `recommendations: null` **vs** `[]` --- the evaluator treats these
    differently. The prompt explicitly distinguishes: `null` for
    clarification turns, populated array for committed shortlists, and
    `[]` is explicitly forbidden. Code-level guard converts `[]` to `null`.
-   **Model dropping recommendations on compare turns** --- fixed by an
    explicit rule: if a shortlist is already committed, keep the array
    during compare turns, do not revert to null.
-   `max_tokens` **too low** --- increased to 2048 to handle up to 10
    recommendations in a single response without truncation.
-   **Temperature too high** --- reduced to 0.1 for more consistent JSON
    output.
-   **Model availability issues** ---
    `mistralai/mistral-7b-instruct:free` became unavailable on
    OpenRouter. Switched to multi-model fallback with free-tier models:
    deepseek-chat-v3, llama-4-maverick, qwen3-235b, and openrouter/free.

### Evaluation Approach

`evaluate.py` replays all 10 provided conversation traces end-to-end
against the live `/chat` endpoint and reports:

-   **Schema compliance (hard eval)** --- correct JSON keys on every
    turn, turn cap honored, catalog-only URLs, no empty arrays
-   **Recall@10** --- fraction of expected assessments appearing in the
    final shortlist, averaged across traces
-   **EOC accuracy** --- `end_of_conversation` correct on the final turn
    of each trace
-   **Behavior probes** --- vague turn-1 queries do not receive
    recommendations; compare turns hold the existing shortlist

Run: `python evaluate.py --url https://your-deployed-url.onrender.com`

### Tools Used

-   **Multi-model fallback via OpenRouter** ---
    deepseek/deepseek-chat-v3-0324:free (primary),
    meta-llama/llama-4-maverick:free,
    qwen/qwen3-235b-a22b:free,
    openrouter/free (fallback)
    reasoning, classification, generation (65K+ context, free tier)
-   **FastAPI + Uvicorn** --- API layer
-   **Pydantic v2** --- request/response validation
-   No vector store, no embeddings, no external search

### Trade-offs and Future Work

-   Full 377-item catalog fits in context but uses significant tokens.
    Future: chunked retrieval (FAISS or pgvector) for larger catalogs.
-   Adding a structured slot-filler (role, level, purpose) would make
    state transitions more deterministic
-   Caching frequent queries would reduce latency and API costs
-   A streaming endpoint would improve perceived latency for longer
    responses
