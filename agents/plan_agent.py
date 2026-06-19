import random
from questions.bank import load_questions, get_categories
from agents.state import AdaptIQState
from irt.scorer import IRTScorer
from irt.selector import QuestionSelector

scorer = IRTScorer()
selector = QuestionSelector(scorer)

all_questions = load_questions("questions/questions.json")

def plan_agent(state: AdaptIQState) -> AdaptIQState:
    """
    PlanAgent — updates study plan and decides next category

    What: Analyzes θ scores → picks weakest topic for next Q
    Why:  Directs student to where they need most work
    How:
      1. Generate updated study plan from current θ scores
      2. Find weakest category with enough data
      3. Check if session should end
      4. Routes → QuizAgent with target_category set

    Routing:
      session_complete → end
      else             → quiz_agent
    """

    student    = state["student_state"]
    categories = get_categories(all_questions)

    # ── Build detailed per-category analysis ──────────
    def build_detailed_plan(student, categories):
        plan = []
        for cat in categories:
            theta    = student.get_theta(cat)
            total    = student.total_count.get(cat, 0)
            correct  = student.correct_count.get(cat, 0)
            accuracy = (correct / total * 100) if total > 0 else None

            # Find specific questions missed
            missed_topics = []
            for record in student.session_log:
                if (record.get("category") == cat
                        and not record.get("is_correct")):
                    # Find the question text from session log
                    qid = record.get("question_id", "")
                    # Get topic from question id prefix
                    missed_topics.append(qid)

            # Determine priority + emoji
            if theta >= 1.0:
                priority = "Low — Review only"
                emoji    = "✅"
            elif theta >= 0.5:
                priority = "Medium — Needs practice"
                emoji    = "📈"
            elif theta >= 0.0:
                priority = "High — Focus here"
                emoji    = "🔴"
            else:
                priority = "Critical — Start here"
                emoji    = "🚨"

            # Recommended source per category
            source_map = {
                "Python":           "Python docs + Real Python",
                "Classical ML":     "HuggingFace NLP Course",
                "Deep Learning":    "HuggingFace — BERT docs",
                "NLP & Transformers": "HuggingFace NLP Course",
                "LLMs":             "HuggingFace — LoRA + Fine-tuning",
                "RAG Systems":      "LangChain Docs — RAG",
                "Agents":           "LangChain Docs — Agents",
                "MLOps":            "HuggingFace — Model Cards",
                "System Design":    "LangChain Docs — Vector Stores",
            }

            plan.append({
                "category":      cat,
                "theta":         round(theta, 2),
                "level":         scorer.level_label(theta),
                "priority":      priority,
                "emoji":         emoji,
                "answered":      total,
                "accuracy":      round(accuracy, 1) if accuracy is not None else None,
                "correct":       correct,
                "missed_count":  total - correct if total > 0 else 0,
                "missed_topics": missed_topics[:3],  # top 3
                "recommended_source": source_map.get(cat, ""),
            })

        return sorted(plan, key=lambda x: x["theta"])

    # ── Check session end ─────────────────────────────
    if state["questions_asked"] >= state["max_questions"]:
        plan = build_detailed_plan(student, categories)

        # Build actionable summary text
        plan_lines = []
        for p in plan:
            acc_str = (f"{p['accuracy']:.0f}%"
                       if p['accuracy'] is not None
                       else "not yet tested")

            if p["answered"] == 0:
                detail = "not yet attempted"
            elif p["missed_count"] == 0:
                detail = f"✓ {p['correct']}/{p['answered']} correct"
            else:
                detail = (f"✗ missed {p['missed_count']}"
                          f"/{p['answered']} questions")

            plan_lines.append(
                f"{p['emoji']} {p['category']:<25} "
                f"θ={p['theta']:>5.2f} | {detail}"
                + (f"\n   📖 Study: {p['recommended_source']}"
                   if p["theta"] < 0 else "")
            )

        return {
            **state,
            "session_complete": True,
            "study_plan":       plan,
            "message": (
                f"\n{'='*55}\n"
                f"🎓 SESSION COMPLETE!\n"
                f"{'='*55}\n"
                f"Questions answered: {state['questions_asked']}\n\n"
                f"ACTIONABLE STUDY PLAN:\n"
                + "\n".join(plan_lines)
            ),
            "next_action": "end"
        }

    # ── Mid-session: pick weakest unattempted category ─
    unattempted = [
        cat for cat in categories
        if student.total_count.get(cat, 0) == 0
    ]

    if unattempted:
        weak_cat = random.choice(unattempted)
    else:
        weak_cat = selector.select_weakest_category(
            student, min_questions=1
        )

    plan  = build_detailed_plan(student, categories)
    asked = state["questions_asked"]

    msg = (
        f"\n📊 Progress: {asked}/{state['max_questions']} questions\n"
        f"Next focus: {weak_cat or 'Balanced'}\n"
        f"Remaining: {state['max_questions'] - asked} questions"
    )

    return {
        **state,
        "study_plan":      plan,
        "target_category": weak_cat,
        "message":         msg,
        "next_action":     "quiz"
    }