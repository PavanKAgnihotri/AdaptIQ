from typing import TypedDict, Optional
from irt.scorer import Question, StudentState

class AdaptIQState(TypedDict):
    """
    Shared state passed between all agents in the graph

    What: Single source of truth for the entire session
    Why:  LangGraph passes state between nodes — every
          agent reads from and writes to this dict
    How:  TypedDict enforces schema — no missing fields
    """

    # ── Student info ──────────────────────────────────
    student_id:       str
    student_state:    StudentState      # θ per category

    # ── Current question ──────────────────────────────
    current_question: Optional[Question]
    student_answer:   Optional[str]

    # ── Evaluation result ─────────────────────────────
    is_correct:       Optional[bool]
    score_result:     Optional[dict]   # full IRT result

    # ── Explanation ───────────────────────────────────
    explanation:      Optional[str]
    sources_cited:    Optional[list]

    # ── Session control ───────────────────────────────
    questions_asked:  int
    max_questions:    int
    session_complete: bool

    # ── Study plan ────────────────────────────────────
    study_plan:       Optional[list]
    target_category:  Optional[str]    # forced category for next Q

    # ── Message to display to user ────────────────────
    message:          Optional[str]

    # ── Routing ───────────────────────────────────────
    next_action:      Optional[str]    # "quiz"|"explain"|"plan"|"end"