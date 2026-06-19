from agents.state import AdaptIQState
from irt.scorer import IRTScorer

scorer = IRTScorer()

def eval_agent(state: AdaptIQState) -> AdaptIQState:
    """
    EvalAgent — scores student answer with IRT

    What: Evaluates correctness and updates θ
    Why:  Core of adaptive system — θ drives all decisions
    How:
      1. score_answer() → updates θ in StudentState
      2. Records result in state
      3. Routes: correct → PlanAgent
                 wrong   → ExplainAgent

    Routing:
      correct → plan_agent
      wrong   → explain_agent
    """

    q      = state["current_question"]
    answer = state["student_answer"]

    if not q or not answer:
        return {**state, "next_action": "quiz"}

    # Score with IRT
    result     = scorer.score_answer(
        state    = state["student_state"],
        question = q,
        student_answer = answer
    )

    # Build feedback message
    if result["is_correct"]:
        msg = (
            f"\n✅ Correct!\n"
            f"θ {q.category}: "
            f"{result['old_theta']:.2f} → {result['new_theta']:.2f} "
            f"({result['delta']:+.3f})\n"
            f"Level: {result['level']}"
        )
        next_action = "plan"
    else:
        msg = (
            f"\n❌ Incorrect. Correct answer: {q.correct}\n"
            f"θ {q.category}: "
            f"{result['old_theta']:.2f} → {result['new_theta']:.2f} "
            f"({result['delta']:+.3f})\n"
            f"Level: {result['level']}\n"
            f"📖 Fetching explanation..."
        )
        next_action = "explain"

    return {
        **state,
        "is_correct":      result["is_correct"],
        "score_result":    result,
        "questions_asked": state["questions_asked"] + 1,
        "message":         msg,
        "next_action":     next_action
    }