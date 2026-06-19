import json, time, os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

DRIVE_OBS = "/content/drive/MyDrive/adaptiq/observability.jsonl"
LOCAL_OBS = "data/observability.jsonl"

def get_obs_path() -> str:
    if os.path.exists("/content/drive/MyDrive"):
        return DRIVE_OBS
    return LOCAL_OBS

class AdaptIQTracker:
    """
    Observability tracker for AdaptIQ sessions

    Tracks per question:
    ├── latency (time to answer + time to explain)
    ├── IRT metrics (θ before/after, delta, difficulty)
    ├── quality (correct/wrong, explanation triggered)
    └── cost (LLM tokens used)

    Tracks per session:
    ├── accuracy per category
    ├── θ progression
    ├── avg latency
    └── total cost estimate
    """

    GROQ_COST_PER_1M = 0.05   # $0.05 per 1M tokens (Llama 3.1 8B)

    def __init__(self, student_id: str):
        self.student_id  = student_id
        self.session_id  = datetime.now(timezone.utc).strftime(
            "%Y%m%d_%H%M%S"
        )
        self.records     = []
        self.current     = {}
        self.session_start = time.perf_counter()

    def start_question(self, question, theta_before: float):
        """Call when question is shown to student"""
        self.current = {
            "student_id":    self.student_id,
            "session_id":    self.session_id,
            "question_id":   question.id,
            "category":      question.category,
            "difficulty":    question.difficulty,
            "theta_before":  round(theta_before, 4),
            "q_start_time":  time.perf_counter(),
            "timestamp":     datetime.now(timezone.utc).isoformat()
        }

    def record_answer(self, is_correct: bool,
                      theta_after: float,
                      answer_given: str):
        """Call immediately after student answers"""
        elapsed = time.perf_counter() - self.current["q_start_time"]

        self.current.update({
            "is_correct":      is_correct,
            "answer_given":    answer_given,
            "theta_after":     round(theta_after, 4),
            "theta_delta":     round(theta_after - self.current["theta_before"], 4),
            "answer_latency_ms": round(elapsed * 1000, 2),
            "explanation_triggered": not is_correct
        })

    def record_explanation(self, chunks_used: int,
                           tokens_used: int = 0):
        """Call after explanation is generated"""
        self.current.update({
            "chunks_retrieved": chunks_used,
            "tokens_used":      tokens_used,
            "cost_usd":         round(
                tokens_used / 1_000_000 * self.GROQ_COST_PER_1M, 6
            )
        })

    def finish_question(self):
        """Call after plan_agent runs — saves record"""
        total_elapsed = (
            time.perf_counter() - self.current["q_start_time"]
        ) * 1000
        self.current["total_latency_ms"] = round(total_elapsed, 2)

        self.records.append(self.current.copy())

        # Append to JSONL log
        obs_path = get_obs_path()
        Path(os.path.dirname(obs_path)).mkdir(
            parents=True, exist_ok=True
        )
        with open(obs_path, "a") as f:
            f.write(json.dumps(self.current) + "\n")

        return self.current.copy()

    def session_summary(self) -> dict:
        """
        Print p50/p95 latency, accuracy, cost summary
        Same pattern as SEC RAG observability
        """
        if not self.records:
            print("No records yet")
            return {}

        latencies  = [r["total_latency_ms"] for r in self.records]
        correct    = [r for r in self.records if r["is_correct"]]
        wrong      = [r for r in self.records if not r["is_correct"]]
        total_cost = sum(r.get("cost_usd", 0) for r in self.records)
        total_tokens = sum(r.get("tokens_used", 0) for r in self.records)

        # Per category accuracy
        cat_stats = {}
        for r in self.records:
            cat = r["category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"correct": 0, "total": 0}
            cat_stats[cat]["total"]   += 1
            cat_stats[cat]["correct"] += 1 if r["is_correct"] else 0

        session_elapsed = time.perf_counter() - self.session_start

        print(f"\n{'='*55}")
        print(f"  ADAPTIQ OBSERVABILITY — Session {self.session_id}")
        print(f"{'='*55}")

        print(f"\n📊 SESSION OVERVIEW")
        print(f"  Student:          {self.student_id}")
        print(f"  Questions:        {len(self.records)}")
        print(f"  Correct:          {len(correct)}/{len(self.records)} "
              f"({len(correct)/len(self.records)*100:.0f}%)")
        print(f"  Session duration: {session_elapsed:.0f}s")

        print(f"\n⏱️  LATENCY")
        print(f"  p50: {np.percentile(latencies, 50):.0f}ms")
        print(f"  p95: {np.percentile(latencies, 95):.0f}ms")
        print(f"  max: {max(latencies):.0f}ms")

        print(f"\n💰 COST")
        print(f"  Total tokens:  {total_tokens:,}")
        print(f"  Total cost:    ${total_cost:.6f}")
        print(f"  Avg/question:  ${total_cost/len(self.records):.6f}")

        print(f"\n✅ ACCURACY BY CATEGORY")
        for cat, stats in sorted(cat_stats.items()):
            acc = stats["correct"] / stats["total"] * 100
            bar = "█" * stats["correct"] + "░" * (stats["total"] - stats["correct"])
            print(f"  {cat:<25} {bar} {acc:.0f}%")

        print(f"\n📈 θ PROGRESSION")
        for r in self.records:
            direction = "↑" if r["theta_delta"] > 0 else "↓"
            print(f"  [{r['category'][:12]:<12}] "
                  f"{r['theta_before']:>6.3f} → {r['theta_after']:>6.3f} "
                  f"{direction} ({r['theta_delta']:+.3f}) "
                  f"{'✅' if r['is_correct'] else '❌'}")

        print(f"{'='*55}\n")

        return {
            "total_questions": len(self.records),
            "accuracy":        len(correct)/len(self.records),
            "p50_latency_ms":  np.percentile(latencies, 50),
            "p95_latency_ms":  np.percentile(latencies, 95),
            "total_cost_usd":  total_cost,
            "category_stats":  cat_stats
        }