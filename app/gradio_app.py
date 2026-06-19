import gradio as gr
import random, time
import numpy as np

# Imports from all modules:
from irt.scorer import IRTScorer, StudentState
from irt.selector import QuestionSelector
from questions.bank import load_questions, get_categories
from agents.state import AdaptIQState
from agents.quiz_agent import quiz_agent
from agents.eval_agent import eval_agent
from agents.explain_agent import explain_agent
from agents.plan_agent import plan_agent
from memory.session_store import load_session, save_session
from observability.tracker import AdaptIQTracker
from retrieval.explainer import ExplanationEngine
from retrieval.ingester import (
    qdrant, embedder, COLLECTION_NAME,
    run_ingestion, ensure_collection,
    chunks_saved, load_chunks_from_drive
)


from retrieval.ingester import qdrant, embedder, run_ingestion
from retrieval.explainer import ExplanationEngine
from irt.scorer import IRTScorer
from irt.selector import QuestionSelector
from questions.bank import load_questions

# Initialize once at startup
engine        = ExplanationEngine()
scorer        = IRTScorer()
selector      = QuestionSelector(scorer)
all_questions = load_questions("path-to-questions/questions/questions.json")


class GradioSession:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state         = None
        self.tracker       = None
        self.waiting       = False
        self.student_id    = "pavan"
        self.max_questions = 9

session = GradioSession()


# ══════════════════════════════════════════════════════
# OBSERVABILITY HTML BUILDERS
# ══════════════════════════════════════════════════════

def build_obs_html(tracker) -> str:
    """
    Builds live observability panel shown in right column

    Shows: latency p50/p95, accuracy, cost, θ progression
    Updates after every question answered
    """
    if not tracker or not tracker.records:
        return """
<div style='background:#1e1e2e;padding:16px;border-radius:12px;
            border:1px solid #313244;color:#6c7086;text-align:center'>
  <div style='font-size:24px;margin-bottom:8px'>📊</div>
  <div style='font-size:13px'>Answer questions to see<br>live performance metrics</div>
</div>"""

    records  = tracker.records
    correct  = sum(1 for r in records if r["is_correct"])
    total    = len(records)
    accuracy = correct / total * 100 if total > 0 else 0

    # Latency stats
    latencies = [r.get("total_latency_ms", 0) for r in records]
    p50 = np.percentile(latencies, 50) if latencies else 0
    p95 = np.percentile(latencies, 95) if latencies else 0

    # Cost
    total_tokens = sum(r.get("tokens_used", 0) for r in records)
    total_cost   = total_tokens / 1_000_000 * 0.05

    # Category accuracy bars
    cat_stats = {}
    for r in records:
        cat = r["category"]
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0, "theta": 0}
        cat_stats[cat]["total"]   += 1
        cat_stats[cat]["correct"] += 1 if r["is_correct"] else 0
        cat_stats[cat]["theta"]    = r["theta_after"]

    cat_rows = ""
    for cat, stats in sorted(cat_stats.items(),
                             key=lambda x: x[1]["theta"]):
        acc   = stats["correct"] / stats["total"] * 100
        theta = stats["theta"]
        color = ("#f38ba8" if theta < 0
                 else "#fab387" if theta < 0.5
                 else "#a6e3a1")
        bar_w = int(acc)
        cat_rows += f"""
<div style='margin-bottom:6px'>
  <div style='display:flex;justify-content:space-between;
              font-size:11px;margin-bottom:2px'>
    <span style='color:#cdd6f4'>{cat[:18]}</span>
    <span style='color:{color}'>
      {acc:.0f}% · θ={theta:.2f}
    </span>
  </div>
  <div style='background:#313244;height:4px;border-radius:2px'>
    <div style='background:{color};height:100%;
                width:{bar_w}%;border-radius:2px'></div>
  </div>
</div>"""

    # θ progression rows
    theta_rows = ""
    for r in records[-6:]:   # last 6 questions
        delta = r.get("theta_delta", 0)
        arrow = "↑" if delta > 0 else "↓"
        color = "#a6e3a1" if delta > 0 else "#f38ba8"
        icon  = "✅" if r["is_correct"] else "❌"
        theta_rows += f"""
<tr>
  <td style='padding:3px 6px;color:#cdd6f4;font-size:11px'>
    {icon}</td>
  <td style='padding:3px 6px;color:#89b4fa;font-size:11px'>
    {r['category'][:12]}</td>
  <td style='padding:3px 6px;color:#cdd6f4;font-size:11px'>
    {r['theta_before']:.3f}→{r['theta_after']:.3f}</td>
  <td style='padding:3px 6px;font-size:11px;color:{color}'>
    {arrow}{abs(delta):.3f}</td>
</tr>"""

    # Latency color
    p50_color = ("#a6e3a1" if p50 < 2000
                 else "#fab387" if p50 < 5000
                 else "#f38ba8")

    return f"""
<div style='background:#1e1e2e;border-radius:12px;
            border:1px solid #313244;overflow:hidden'>

  <!-- Header -->
  <div style='background:#181825;padding:10px 14px;
              border-bottom:1px solid #313244'>
    <span style='color:#cba6f7;font-weight:700;font-size:13px'>
      📊 Live Observability
    </span>
    <span style='color:#6c7086;font-size:11px;float:right'>
      {total} questions answered
    </span>
  </div>

  <!-- Top metrics row -->
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
              gap:1px;background:#313244'>
    <div style='background:#1e1e2e;padding:10px;text-align:center'>
      <div style='font-size:18px;font-weight:700;
                  color:{"#a6e3a1" if accuracy >= 70 else "#fab387" if accuracy >= 50 else "#f38ba8"}'>
        {accuracy:.0f}%</div>
      <div style='font-size:10px;color:#6c7086'>Accuracy</div>
    </div>
    <div style='background:#1e1e2e;padding:10px;text-align:center'>
      <div style='font-size:18px;font-weight:700;color:{p50_color}'>
        {p50:.0f}ms</div>
      <div style='font-size:10px;color:#6c7086'>p50 Latency</div>
    </div>
    <div style='background:#1e1e2e;padding:10px;text-align:center'>
      <div style='font-size:18px;font-weight:700;color:#89b4fa'>
        ${total_cost:.5f}</div>
      <div style='font-size:10px;color:#6c7086'>Cost</div>
    </div>
  </div>

  <!-- Latency detail -->
  <div style='padding:10px 14px;border-bottom:1px solid #313244'>
    <div style='color:#cba6f7;font-size:11px;
                font-weight:600;margin-bottom:6px'>
      ⏱️ Latency
    </div>
    <div style='display:flex;gap:16px'>
      <div style='font-size:11px;color:#cdd6f4'>
        p50: <span style='color:{p50_color}'>{p50:.0f}ms</span></div>
      <div style='font-size:11px;color:#cdd6f4'>
        p95: <span style='color:#fab387'>{p95:.0f}ms</span></div>
      <div style='font-size:11px;color:#cdd6f4'>
        Tokens: <span style='color:#89b4fa'>{total_tokens:,}</span></div>
    </div>
  </div>

  <!-- Accuracy by category -->
  <div style='padding:10px 14px;border-bottom:1px solid #313244'>
    <div style='color:#cba6f7;font-size:11px;
                font-weight:600;margin-bottom:8px'>
      ✅ Accuracy by Category
    </div>
    {cat_rows}
  </div>

  <!-- θ Progression -->
  <div style='padding:10px 14px'>
    <div style='color:#cba6f7;font-size:11px;
                font-weight:600;margin-bottom:6px'>
      📈 θ Progression (last {min(6, total)} questions)
    </div>
    <table style='width:100%;border-collapse:collapse'>
      <thead>
        <tr style='color:#6c7086;font-size:10px'>
          <th style='text-align:left;padding:2px 6px'></th>
          <th style='text-align:left;padding:2px 6px'>Category</th>
          <th style='text-align:left;padding:2px 6px'>θ change</th>
          <th style='text-align:left;padding:2px 6px'>Δ</th>
        </tr>
      </thead>
      <tbody>{theta_rows}</tbody>
    </table>
  </div>

</div>"""


def build_question_html(q, state) -> str:
    asked    = state["questions_asked"] + 1
    max_q    = state["max_questions"]
    progress = int((asked / max_q) * 100)
    is_dyn   = q.id.startswith("dyn_")
    tag      = "⚡ AI-Generated" if is_dyn else "📚 Seed Bank"
    tag_col  = "#cba6f7" if is_dyn else "#89b4fa"

    options_html = "".join(
        f"<div style='padding:6px 0;font-size:14px;"
        f"color:#cdd6f4'>{opt}</div>"
        for opt in q.options
    )

    return f"""
<div style='background:#1e1e2e;padding:20px;border-radius:12px;
            border:1px solid #313244;color:#cdd6f4'>
  <div style='display:flex;justify-content:space-between;
              align-items:center;margin-bottom:12px'>
    <span style='font-size:12px;color:#6c7086'>
      Question {asked} of {max_q}
    </span>
    <span style='font-size:11px;color:{tag_col};
                 background:rgba(99,102,241,.1);
                 padding:2px 8px;border-radius:9999px'>
      {tag}
    </span>
    <span style='font-size:12px;color:#6c7086'>
      [{q.category}] · b={q.difficulty:.1f}
    </span>
  </div>
  <div style='background:#313244;height:6px;
              border-radius:3px;margin-bottom:16px'>
    <div style='background:#cba6f7;height:100%;
                width:{progress}%;border-radius:3px'></div>
  </div>
  <p style='font-size:16px;font-weight:600;
            margin-bottom:16px;line-height:1.5;
            color:#cdd6f4'>{q.text}</p>
  <div style='border-top:1px solid #313244;padding-top:12px'>
    {options_html}
  </div>
</div>"""


def build_plan_html(state) -> str:
    plan = state.get("study_plan", [])
    if not plan:
        return ("<p style='color:#6c7086;font-size:12px'>"
                "Study plan updates as you answer</p>")

    rows = ""
    for item in plan[:9]:
        theta = item["theta"]
        bar_w = max(3, min(100,
                           int((theta + 1.5) / 3.0 * 100)))
        color = ("#f38ba8" if theta < 0
                 else "#fab387" if theta < 0.5
                 else "#a6e3a1")
        rows += f"""
<div style='margin-bottom:7px'>
  <div style='display:flex;justify-content:space-between;
              font-size:11px;margin-bottom:2px'>
    <span style='color:#cdd6f4'>
      {item['emoji']} {item['category']}</span>
    <span style='color:{color}'>θ={theta:.2f}</span>
  </div>
  <div style='background:#313244;height:4px;border-radius:2px'>
    <div style='background:{color};height:100%;
                width:{bar_w}%;border-radius:2px'></div>
  </div>
</div>"""

    return f"""
<div style='background:#1e1e2e;padding:14px;border-radius:12px;
            border:1px solid #313244'>
  <div style='color:#cba6f7;font-weight:700;
              font-size:12px;margin-bottom:10px'>
    📚 Knowledge Map
  </div>
  {rows}
</div>"""


def build_history_html(tracker) -> str:
    if not tracker or not tracker.records:
        return ("<p style='color:#6c7086;font-size:12px'>"
                "Answer history appears here</p>")

    correct = sum(1 for r in tracker.records if r["is_correct"])
    total   = len(tracker.records)
    acc     = correct / total * 100 if total > 0 else 0

    rows = ""
    for r in reversed(tracker.records[-8:]):
        icon  = "✅" if r["is_correct"] else "❌"
        delta = r.get("theta_delta", 0)
        color = "#a6e3a1" if delta > 0 else "#f38ba8"
        rows += f"""
<tr style='border-bottom:1px solid #1e1e2e'>
  <td style='padding:4px 6px;font-size:12px'>{icon}</td>
  <td style='padding:4px 6px;color:#89b4fa;font-size:11px'>
    {r['category'][:14]}</td>
  <td style='padding:4px 6px;color:#cdd6f4;font-size:11px'>
    {r['theta_before']:.3f}→{r['theta_after']:.3f}</td>
  <td style='padding:4px 6px;font-size:11px;color:{color}'>
    {delta:+.3f}</td>
</tr>"""

    return f"""
<div style='background:#1e1e2e;padding:14px;border-radius:12px;
            border:1px solid #313244'>
  <div style='color:#cba6f7;font-weight:700;font-size:12px;
              margin-bottom:8px'>
    🕘 History — {correct}/{total} ({acc:.0f}%)
  </div>
  <table style='width:100%;border-collapse:collapse;
                background:#181825;border-radius:8px'>
    <thead>
      <tr style='color:#6c7086;font-size:10px'>
        <th style='padding:3px 6px;text-align:left'></th>
        <th style='padding:3px 6px;text-align:left'>Category</th>
        <th style='padding:3px 6px;text-align:left'>θ</th>
        <th style='padding:3px 6px;text-align:left'>Δ</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""


def build_summary_html(state, tracker) -> str:
    plan    = state.get("study_plan", [])
    records = tracker.records if tracker else []
    correct = sum(1 for r in records if r["is_correct"])
    total   = len(records)
    acc     = correct / total * 100 if total > 0 else 0

    # Latency + cost stats
    latencies    = [r.get("total_latency_ms", 0) for r in records]
    p50          = np.percentile(latencies, 50) if latencies else 0
    p95          = np.percentile(latencies, 95) if latencies else 0
    total_tokens = sum(r.get("tokens_used", 0) for r in records)
    total_cost   = total_tokens / 1_000_000 * 0.05

    plan_rows = ""
    for p in (plan or []):
        theta    = p["theta"]
        answered = p.get("answered", 0)
        missed   = p.get("missed_count", 0)
        acc_p    = p.get("accuracy")
        source   = p.get("recommended_source", "")

        # Accuracy badge
        if acc_p is None:
            acc_badge = (
                "<span style='color:#6c7086;font-size:10px'>"
                "not attempted</span>"
            )
        elif acc_p == 100:
            acc_badge = (
                f"<span style='color:#a6e3a1;font-size:10px'>"
                f"✓ {acc_p:.0f}%</span>"
            )
        else:
            acc_badge = (
                f"<span style='color:#f38ba8;font-size:10px'>"
                f"✗ {acc_p:.0f}% ({missed}/{answered} missed)</span>"
            )

        # Source recommendation (only for weak categories)
        source_html = ""
        if theta < 0 and source:
            source_html = (
                f"<div style='font-size:10px;color:#89b4fa;"
                f"margin-left:20px;margin-top:2px'>"
                f"📖 {source}</div>"
            )

        plan_rows += f"""
<div style='margin-bottom:8px;padding:6px 8px;
            background:#1e1e2e;border-radius:6px'>
  <div style='display:flex;justify-content:space-between;
              align-items:center'>
    <span style='color:#cdd6f4;font-size:12px'>
      {p['emoji']} {p['category']}</span>
    <div style='text-align:right'>
      <span style='color:{"#f38ba8" if theta < 0 else "#fab387" if theta < 0.5 else "#a6e3a1"};
                   font-size:11px'>θ={theta:.2f}</span>
      <span style='margin-left:8px'>{acc_badge}</span>
    </div>
  </div>
  {source_html}
</div>"""

    return f"""
<div style='background:#1e1e2e;padding:24px;border-radius:12px;
            border:1px solid #cba6f7;color:#cdd6f4'>
  <h2 style='color:#cba6f7;margin-bottom:16px;font-size:20px'>
    🎓 Session Complete!
  </h2>

  <!-- Score cards -->
  <div style='display:grid;grid-template-columns:1fr 1fr 1fr;
              gap:10px;margin-bottom:20px'>
    <div style='background:#313244;padding:12px;
                border-radius:8px;text-align:center'>
      <div style='font-size:22px;font-weight:700;
                  color:#a6e3a1'>{correct}/{total}</div>
      <div style='font-size:11px;color:#6c7086'>Correct</div>
    </div>
    <div style='background:#313244;padding:12px;
                border-radius:8px;text-align:center'>
      <div style='font-size:22px;font-weight:700;
                  color:#89b4fa'>{acc:.0f}%</div>
      <div style='font-size:11px;color:#6c7086'>Accuracy</div>
    </div>
    <div style='background:#313244;padding:12px;
                border-radius:8px;text-align:center'>
      <div style='font-size:22px;font-weight:700;
                  color:#cba6f7'>{total}</div>
      <div style='font-size:11px;color:#6c7086'>Questions</div>
    </div>
  </div>

  <!-- Observability -->
  <div style='background:#181825;padding:12px;border-radius:8px;
              margin-bottom:16px'>
    <div style='color:#cba6f7;font-weight:600;font-size:12px;
                margin-bottom:8px'>📊 Session Observability</div>
    <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px'>
      <div style='font-size:11px;color:#cdd6f4'>
        p50: <span style='color:#a6e3a1'>{p50:.0f}ms</span></div>
      <div style='font-size:11px;color:#cdd6f4'>
        p95: <span style='color:#fab387'>{p95:.0f}ms</span></div>
      <div style='font-size:11px;color:#cdd6f4'>
        Tokens: <span style='color:#89b4fa'>{total_tokens:,}</span></div>
      <div style='font-size:11px;color:#cdd6f4'>
        Cost: <span style='color:#89b4fa'>${total_cost:.5f}</span></div>
    </div>
  </div>

  <!-- Actionable study plan -->
  <div style='border-top:1px solid #313244;padding-top:14px'>
    <div style='color:#cba6f7;font-weight:600;font-size:13px;
                margin-bottom:10px'>
      📚 Actionable Study Plan
    </div>
    <div style='font-size:10px;color:#6c7086;margin-bottom:8px'>
      Sorted by θ (weakest first) · 📖 = recommended resource
    </div>
    {plan_rows}
  </div>

  <div style='margin-top:14px;padding:8px;background:#313244;
              border-radius:6px;font-size:11px;color:#6c7086'>
    ✅ Progress saved · JSONL log updated · Drive backup complete
  </div>
</div>"""


# ══════════════════════════════════════════════════════
# SESSION FUNCTIONS
# ══════════════════════════════════════════════════════

def start_session(student_name, num_questions, load_prev):
    session.student_id    = student_name or "student"
    session.max_questions = int(num_questions)

    restored = (load_session(session.student_id)
                if load_prev
                else StudentState(student_id=session.student_id))

    session.tracker = AdaptIQTracker(session.student_id)

    session.state = {
        "student_id":       session.student_id,
        "student_state":    restored,
        "current_question": None,
        "student_answer":   None,
        "is_correct":       None,
        "score_result":     None,
        "explanation":      None,
        "sources_cited":    None,
        "questions_asked":  0,
        "max_questions":    session.max_questions,
        "session_complete": False,
        "study_plan":       None,
        "target_category":  None,
        "message":          None,
        "next_action":      "quiz"
    }

    session.state   = quiz_agent(session.state)
    q               = session.state["current_question"]
    session.waiting = True

    theta_before = session.state["student_state"].get_theta(
        q.category
    )
    session.tracker.start_question(q, theta_before)
    session.tracker.current["q_start_time"] = time.perf_counter()

    return (
        build_question_html(q, session.state),  # question
        "",                                      # feedback
        "",                                      # explanation
        build_history_html(session.tracker),     # history
        build_plan_html(session.state),          # plan
        build_obs_html(session.tracker),         # observability
        gr.update(visible=True),                 # answer buttons
        gr.update(visible=False),                # start panel
    )


def submit_answer(answer: str):
    if not session.waiting or not session.state:
        return ("⚠️ Start a session first.",
                "", "", "", "",
                build_obs_html(None),
                gr.update(visible=True),
                gr.update(visible=False))

    session.state["student_answer"] = answer
    session.waiting = False

    # EvalAgent
    session.state = eval_agent(session.state)
    q             = session.state["current_question"]
    result        = session.state["score_result"]
    correct       = session.state["is_correct"]

    # Track answer
    theta_after = session.state["student_state"].get_theta(
        q.category
    )
    session.tracker.record_answer(
        is_correct   = correct,
        theta_after  = theta_after,
        answer_given = answer
    )

    # Feedback message
    if correct:
        feedback = (
            f"✅ **Correct!**\n\n"
            f"θ {q.category}: "
            f"{result['old_theta']:.3f} → {result['new_theta']:.3f} "
            f"({result['delta']:+.3f})\n"
            f"Level: {result['level']}"
        )
        explanation_text = ""
    else:
        feedback = (
            f"❌ **Incorrect.** Correct: **{q.correct}**\n\n"
            f"θ {q.category}: "
            f"{result['old_theta']:.3f} → {result['new_theta']:.3f} "
            f"({result['delta']:+.3f})\n"
            f"Level: {result['level']}"
        )
        session.state = explain_agent(session.state)
        explanation_text = session.state.get("explanation", "")

        session.tracker.record_explanation(
            chunks_used = 5,
            tokens_used = 280
        )

    # PlanAgent
    session.state = plan_agent(session.state)
    session.tracker.finish_question()

    # Session complete?
    if session.state.get("session_complete"):
        save_session(session.state["student_state"])
        return (
            build_summary_html(session.state, session.tracker),
            feedback,
            explanation_text,
            build_history_html(session.tracker),
            build_plan_html(session.state),
            build_obs_html(session.tracker),    # final obs panel
            gr.update(visible=False),
            gr.update(visible=True)
        )

    # Next question
    session.state   = quiz_agent(session.state)
    q_next          = session.state["current_question"]
    session.waiting = True

    theta_before = session.state["student_state"].get_theta(
        q_next.category
    )
    session.tracker.start_question(q_next, theta_before)
    session.tracker.current["q_start_time"] = time.perf_counter()

    return (
        build_question_html(q_next, session.state),
        feedback,
        explanation_text,
        build_history_html(session.tracker),
        build_plan_html(session.state),
        build_obs_html(session.tracker),        # live obs update
        gr.update(visible=True),
        gr.update(visible=False)
    )


# ══════════════════════════════════════════════════════
# GRADIO UI LAYOUT
# ══════════════════════════════════════════════════════
with gr.Blocks(
    title="AdaptIQ — AI Interview Coach",
    theme=gr.themes.Base(),
    css="""
    body,.gradio-container{background:#181825!important}
    .gr-button{font-family:Inter,sans-serif}
    """
) as demo:

    gr.Markdown("""
# 🧠 AdaptIQ — Adaptive AI Interview Coach
**IRT adaptive testing + RAG explanations + Live observability**
_Difficulty adapts to your level. Questions generated by AI after calibration._
    """)

    with gr.Row():

        # ── LEFT: Question + Controls (scale=2) ───────────
        with gr.Column(scale=2):

            # Start panel
            with gr.Group(visible=True) as start_panel:
                gr.Markdown("### 👤 Start Session")
                student_name = gr.Textbox(
                    label="Your name",
                    value="pavan",
                    placeholder="Enter your name"
                )
                num_q = gr.Slider(
                    5, 20, value=9, step=1,
                    label="Number of questions"
                )
                load_prev = gr.Checkbox(
                    label="Resume from previous session",
                    value=False
                )
                start_btn = gr.Button(
                    "🚀 Start Session",
                    variant="primary", size="lg"
                )

            # Question display
            question_display = gr.HTML(
                "<div style='background:#1e1e2e;padding:40px;"
                "border-radius:12px;border:1px solid #313244;"
                "text-align:center;color:#6c7086'>"
                "🧠 Click Start to begin your adaptive session"
                "</div>"
            )

            # Answer buttons
            with gr.Group(visible=False) as answer_panel:
                gr.Markdown("### Your Answer:")
                with gr.Row():
                    btn_a = gr.Button("A", variant="secondary",
                                      size="lg")
                    btn_b = gr.Button("B", variant="secondary",
                                      size="lg")
                    btn_c = gr.Button("C", variant="secondary",
                                      size="lg")
                    btn_d = gr.Button("D", variant="secondary",
                                      size="lg")

            # Feedback
            feedback_box = gr.Markdown("")

            # Explanation
            with gr.Accordion("📖 Explanation", open=True):
                explanation_box = gr.Markdown("")

        # ── RIGHT: Analytics (scale=1) ────────────────────
        with gr.Column(scale=1):

            # Live observability panel (NEW)
            gr.Markdown("### 📊 Live Observability")
            obs_display = gr.HTML(
                "<div style='background:#1e1e2e;padding:20px;"
                "border-radius:12px;border:1px solid #313244;"
                "text-align:center;color:#6c7086;font-size:12px'>"
                "Metrics appear here as you answer</div>"
            )

            gr.Markdown("### 📚 Knowledge Map")
            plan_display = gr.HTML(
                "<p style='color:#6c7086;font-size:12px'>"
                "Start a session to see your knowledge map</p>"
            )

            gr.Markdown("### 🕘 Answer History")
            history_display = gr.HTML(
                "<p style='color:#6c7086;font-size:12px'>"
                "Your answers will appear here</p>"
            )

    # ── Wire outputs ──────────────────────────────────────
    outputs = [
        question_display,   # 0
        feedback_box,       # 1
        explanation_box,    # 2
        history_display,    # 3
        plan_display,       # 4
        obs_display,        # 5  ← NEW observability
        answer_panel,       # 6
        start_panel,        # 7
    ]

    start_btn.click(
        fn=start_session,
        inputs=[student_name, num_q, load_prev],
        outputs=outputs
    )
    btn_a.click(fn=lambda: submit_answer("A"), outputs=outputs)
    btn_b.click(fn=lambda: submit_answer("B"), outputs=outputs)
    btn_c.click(fn=lambda: submit_answer("C"), outputs=outputs)
    btn_d.click(fn=lambda: submit_answer("D"), outputs=outputs)


demo.launch(share=True)