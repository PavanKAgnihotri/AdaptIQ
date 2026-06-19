from agents.state import AdaptIQState
from retrieval.explainer import ExplanationEngine

engine  = ExplanationEngine()

def explain_agent(state: AdaptIQState) -> AdaptIQState:
    """
    ExplainAgent — generates cited explanation for wrong answers

    What: RAG retrieval + LLM generation for wrong answers
    Why:  Student needs to understand WHY they were wrong
    How:
      1. Calls ExplanationEngine.generate_explanation()
      2. Attaches explanation + sources to state
      3. Routes → PlanAgent

    Routing:
      Always → plan_agent
    """
    
    q      = state["current_question"]
    result = state["score_result"]

    if not q or not result:
        return {**state, "next_action": "plan"}

    # Generate RAG explanation
    explanation_result = engine.generate_explanation(
        question_text      = q.text,
        correct_answer     = f"{q.correct} — "
                             f"{q.options[ord(q.correct)-65]}",
        student_answer     = state["student_answer"],
        category           = q.category,
        static_explanation = q.explanation
    )

    explanation_text = explanation_result["explanation"]
    sources          = explanation_result["sources_cited"] \
                       if "sources_cited" in explanation_result \
                       else explanation_result.get("sources", [])

    msg = (
        f"\n📖 EXPLANATION:\n"
        f"{'─'*50}\n"
        f"{explanation_text}\n"
        f"{'─'*50}"
    )

    return {
        **state,
        "explanation":   explanation_text,
        "sources_cited": sources,
        "message":       msg,
        "next_action":   "plan"
    }