"""
evaluate.py — SHL Assessment Recommender Evaluator

Replays all 10 conversation traces against your deployed /chat endpoint.
Reports schema compliance, Recall@10, EOC accuracy, and behavior probes.

Usage:
    python evaluate.py
    python evaluate.py --url https://shl-recommender-m5kx.onrender.com
"""

import argparse, json, time, sys
import requests

# ── Conversation Traces ───────────────────────────────────────────────────────
TRACES = [
    {
        "id": "C1",
        "description": "Senior leadership selection — OPQ + reports",
        "turns": [
            "We need a solution for senior leadership.",
            "The pool consists of CXOs, director-level positions; people with more than 15 years of experience.",
            "Selection — comparing candidates against a leadership benchmark.",
            "Perfect, that's what we need.",
        ],
        "expected_final": [
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ Universal Competency Report 2.0",
            "OPQ Leadership Report",
        ],
        "expected_eoc": True,
        "vague_turn_1": True,
    },
    {
        "id": "C2",
        "description": "Senior Rust engineer — technical + cognitive + personality",
        "turns": [
            "I'm hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?",
            "Yes, go ahead. Should I also add a cognitive test for this level?",
            "That works. Thanks.",
        ],
        "expected_final": [
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
        "expected_eoc": True,
        "vague_turn_1": False,
    },
    {
        "id": "C3",
        "description": "Contact centre agents — SVAR + simulation stack",
        "turns": [
            "We're screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?",
            "English.",
            "US.",
            "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?",
            "Perfect — new simulation for volume, old solution for finalists. Confirmed.",
        ],
        "expected_final": [
            "Contact Center Call Simulation (New)",
            "Customer Service Phone Simulation",
        ],
        "expected_eoc": True,
        "vague_turn_1": True,
    },
    {
        "id": "C4",
        "description": "Graduate financial analysts — numerical + SJT + domain tests",
        "turns": [
            "Hiring graduate financial analysts — final-year students, no work experience. We need numerical reasoning and a finance knowledge test.",
            "Good. Can you also add a situational judgement element — work-context decision making for graduates?",
            "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates.",
        ],
        "expected_final": [
            "SHL Verify Interactive – Numerical Reasoning",
            "Graduate Scenarios",
            "Financial Accounting (New)",
        ],
        "expected_eoc": True,
        "vague_turn_1": False,
    },
    {
        "id": "C5",
        "description": "Sales org re-skilling audit — GSA + OPQ + Sales reports",
        "turns": [
            "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?",
            "What's the difference between OPQ and OPQ MQ Sales Report?",
            "Clear. We'll use OPQ for everyone and add MQ only where we want motivators in the Sales Report; keeping the five solutions as our audit stack.",
        ],
        "expected_final": [
            "Global Skills Assessment",
            "Occupational Personality Questionnaire OPQ32r",
            "OPQ MQ Sales Report",
        ],
        "expected_eoc": True,
        "vague_turn_1": False,
    },
    {
        "id": "C6",
        "description": "Chemical plant operators — safety-critical personality focus",
        "turns": [
            "We're hiring plant operators for a chemical facility. Safety is absolute top priority — reliability, procedure compliance, never cutting corners. What do you recommend?",
            "What's the difference between the DSI and the Safety & Dependability 8.0?",
            "We're industrial. The 8.0 bundle is the right fit. Confirmed.",
        ],
        "expected_final": [
            "Dependability and Safety Instrument (DSI)",
            "Workplace Health and Safety (New)",
        ],
        "expected_eoc": True,
        "vague_turn_1": False,
    },
    {
        "id": "C7",
        "description": "Bilingual healthcare admin — hybrid English/Spanish battery",
        "turns": [
            "We're hiring bilingual healthcare admin staff in South Texas — they handle patient records and need to be assessed in Spanish. HIPAA compliance is critical. What assessments work?",
            "They're functionally bilingual — English fluent for written work. Go with the hybrid.",
            "Are we legally required under HIPAA to test all staff who touch patient records? And does this SHL test satisfy that requirement?",
            "Understood. Keep the shortlist as-is.",
        ],
        "expected_final": [
            "HIPAA (Security)",
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r",
        ],
        "expected_eoc": True,
        "vague_turn_1": True,
    },
    {
        "id": "C8",
        "description": "Admin assistants — Excel and Word with simulation upgrade",
        "turns": [
            "I need to quickly screen admin assistants for Excel and Word daily.",
            "In that case, I am OK with adding a simulation - we want to capture the capabilities.",
            "That's good.",
        ],
        "expected_final": [
            "MS Excel (New)",
            "MS Word (New)",
        ],
        "expected_eoc": True,
        "vague_turn_1": False,
    },
    {
        "id": "C9",
        "description": "Senior full-stack engineer — 7 turns with add/drop refinements",
        "turns": [
            'Here\'s the JD for an engineer we need to fill. Can you recommend an assessment battery?\n\n"Senior Full-Stack Engineer — 5+ years across Core Java, Spring, REST API design, Angular, SQL/relational databases, AWS deployment, and Docker. Will own end-to-end microservice delivery, contribute to architectural decisions, and mentor mid-level engineers."',
            "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional — they'd review frontend PRs but not own features.",
            "Senior IC. They lead design on their own services but don't manage other engineers directly.",
            "Add AWS and Docker. Drop REST — the API design signal will already come through in Spring and the live interview.",
            "On Java — they'd be working on existing services, not greenfield. Is the Advanced level the right pick?",
            "Do we really need Verify G+ on top of all the technical tests? Feels redundant.",
            "Keep Verify G+. Locking it in.",
        ],
        "expected_final": [
            "Core Java (Advanced Level) (New)",
            "Spring (New)",
            "SQL (New)",
            "Amazon Web Services (AWS) Development (New)",
            "Docker (New)",
            "SHL Verify Interactive G+",
            "Occupational Personality Questionnaire OPQ32r",
        ],
        "expected_eoc": True,
        "vague_turn_1": False,
    },
    {
        "id": "C10",
        "description": "Graduate management trainees — OPQ dropped on user request",
        "turns": [
            "We run a graduate management trainee scheme. We need a full battery — cognitive, personality, and situational judgement. All recent graduates.",
            "But can you remove the OPQ32r and replace it with something shorter? Candidates complain it takes too long.",
            "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios.",
        ],
        "expected_final": [
            "SHL Verify Interactive G+",
            "Graduate Scenarios",
        ],
        "expected_eoc": True,
        "vague_turn_1": False,
    },
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def recall_at_k(recommended: list, expected: list, k: int = 10) -> float:
    """Fraction of expected assessments found in top-k recommendations."""
    if not expected:
        return 1.0
    top_k = recommended[:k]
    hits = sum(
        1 for exp in expected
        if any(exp.lower() in rec.lower() or rec.lower() in exp.lower() for rec in top_k)
    )
    return hits / len(expected)


def check_schema(resp: dict) -> tuple[bool, str]:
    """Validate response has all required fields with correct types."""
    for field in ("reply", "recommendations", "end_of_conversation"):
        if field not in resp:
            return False, f"Missing field: '{field}'"

    if not isinstance(resp["reply"], str):
        return False, "'reply' must be a string"

    if not isinstance(resp["end_of_conversation"], bool):
        return False, "'end_of_conversation' must be a boolean"

    recs = resp["recommendations"]
    if recs is not None:
        if not isinstance(recs, list):
            return False, f"'recommendations' must be null or array, got {type(recs).__name__}"
        if len(recs) == 0:
            return False, "'recommendations' is [] — use null for clarifying turns"
        if len(recs) > 10:
            return False, f"'recommendations' has {len(recs)} items — max is 10"
        required = {"name", "url", "test_type", "duration", "remote_testing", "adaptive_irt", "description"}
        for i, r in enumerate(recs):
            missing = required - set(r.keys())
            if missing:
                return False, f"Recommendation[{i}] missing fields: {missing}"
            # Validate test_type is single letter code
            test_type = r.get("test_type", "")
            if not isinstance(test_type, str) or len(test_type) != 1 or test_type not in "ABCDKPS":
                return False, f"Recommendation[{i}] invalid test_type: '{test_type}' — must be one of A,B,C,D,K,P,S"

    return True, "OK"


def post_chat(base_url: str, messages: list, timeout: int = 30) -> dict | None:
    try:
        r = requests.post(
            f"{base_url}/chat",
            json={"messages": messages},
            timeout=timeout,
        )
        if r.status_code != 200:
            print(f"      HTTP {r.status_code}: {r.text[:300]}")
            return None
        return r.json()
    except requests.exceptions.Timeout:
        print(f"      Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"      Request error: {e}")
        return None

# ── Main evaluator ────────────────────────────────────────────────────────────

def run_evaluation(base_url: str):
    base_url = base_url.rstrip("/")

    print(f"\n{'='*62}")
    print(f"  SHL Assessment Recommender — Evaluator")
    print(f"  Endpoint : {base_url}")
    print(f"{'='*62}\n")

    # Health check first
    try:
        h = requests.get(f"{base_url}/health", timeout=15)
        data = h.json()
        assert data.get("status") == "ok", f"Unexpected response: {data}"
        print(f"  /health check ... OK\n")
    except Exception as e:
        print(f"  /health check FAILED: {e}")
        print("  Make sure the service is running and try again.")
        sys.exit(1)

    all_results = []

    for trace in TRACES:
        print(f"{'─'*62}")
        print(f"  {trace['id']}: {trace['description']}")
        print(f"{'─'*62}")

        history        = []
        schema_ok_all  = True
        final_recs     = None
        final_eoc      = False
        probe_vague_ok = True
        probe_commit_ok = True
        has_committed  = False

        for i, user_msg in enumerate(trace["turns"]):
            turn_num = i + 1
            short_msg = user_msg[:70] + "..." if len(user_msg) > 70 else user_msg
            print(f"  Turn {turn_num}: "{short_msg}"")

            history.append({"role": "user", "content": user_msg})

            resp = post_chat(base_url, history, timeout=30)
            if resp is None:
                schema_ok_all = False
                print(f"      [NO RESPONSE — stopping trace]\n")
                break

            # Add assistant reply to history for next turn
            history.append({"role": "assistant", "content": resp.get("reply", "")})

            # Schema check
            ok, msg = check_schema(resp)
            schema_marker = "PASS" if ok else f"FAIL ({msg})"
            if not ok:
                schema_ok_all = False

            recs = resp.get("recommendations")
            n_recs = len(recs) if recs else 0

            # Track commitment state
            if recs is not None and n_recs > 0:
                has_committed = True

            # Turn-1 vague probe
            if turn_num == 1 and trace["vague_turn_1"]:
                if recs is not None:
                    probe_vague_ok = False
                    print(f"      Schema:{schema_marker} | PROBE FAIL: recommended on vague turn 1!")
                else:
                    print(f"      Schema:{schema_marker} | Probe:no-rec-on-vague-T1=PASS | recs=null | eoc={resp.get('end_of_conversation')}")
            else:
                # Check for regression: going back to null after commitment
                if has_committed and recs is None and turn_num > 1:
                    probe_commit_ok = False
                    print(f"      Schema:{schema_marker} | PROBE FAIL: dropped recommendations after commitment!")
                else:
                    print(f"      Schema:{schema_marker} | recs={'null' if recs is None else n_recs} | eoc={resp.get('end_of_conversation')}")

            # Track final state
            if recs:
                final_recs = [r["name"] for r in recs]
            final_eoc = resp.get("end_of_conversation", False)

            time.sleep(0.4)  # avoid rate limiting

        # Per-trace scoring
        recall    = recall_at_k(final_recs or [], trace["expected_final"])
        eoc_match = final_eoc == trace["expected_eoc"]

        missing = [
            e for e in trace["expected_final"]
            if not any(
                e.lower() in r.lower() or r.lower() in e.lower()
                for r in (final_recs or [])
            )
        ]

        print(f"\n  Results:")
        print(f"    Schema all turns : {'PASS' if schema_ok_all else 'FAIL'}")
        print(f"    Recall@10        : {recall:.2f}  ({len(trace['expected_final']) - len(missing)}/{len(trace['expected_final'])} expected found)")
        print(f"    EOC correct      : {'PASS' if eoc_match else 'FAIL'}  (expected={trace['expected_eoc']}, got={final_eoc})")
        if missing:
            print(f"    Missing          : {missing}")
        print()

        all_results.append({
            "id"          : trace["id"],
            "schema_pass" : schema_ok_all,
            "recall"      : recall,
            "eoc_pass"    : eoc_match,
            "probe_pass"  : probe_vague_ok and probe_commit_ok,
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    n             = len(all_results)
    schema_passes = sum(1 for r in all_results if r["schema_pass"])
    mean_recall   = sum(r["recall"] for r in all_results) / n
    eoc_passes    = sum(1 for r in all_results if r["eoc_pass"])
    probe_passes  = sum(1 for r in all_results if r["probe_pass"])

    print(f"\n{'='*62}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*62}")
    print(f"  Hard evals (schema)  : {schema_passes}/{n} traces passed")
    print(f"  Mean Recall@10       : {mean_recall:.3f}")
    print(f"  EOC accuracy         : {eoc_passes}/{n} traces")
    print(f"  Behavior probes      : {probe_passes}/{n} traces")
    print()
    print(f"  Per-trace:")
    for r in all_results:
        status = "OK" if (r["schema_pass"] and r["recall"] >= 0.5 and r["eoc_pass"]) else "!!"
        print(
            f"    [{status}] {r['id']}: "
            f"schema={'PASS' if r['schema_pass'] else 'FAIL'}  "
            f"recall={r['recall']:.2f}  "
            f"eoc={'PASS' if r['eoc_pass'] else 'FAIL'}  "
            f"probe={'PASS' if r['probe_pass'] else 'FAIL'}"
        )

    overall = (
        schema_passes == n and
        mean_recall   >= 0.6 and
        eoc_passes    >= n - 1
    )
    print(f"\n  Overall: {'PASS' if overall else 'NEEDS IMPROVEMENT'}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHL Recommender Evaluator")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of your deployed FastAPI service (default: http://localhost:8000)",
    )
    args = parser.parse_args()
    run_evaluation(args.url)
