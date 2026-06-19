import numpy as np
from irt.scorer import IRTScorer, Question, StudentState

class QuestionSelector:
    """
    Selects the BEST next question for a student
    using Maximum Fisher Information criterion

    Core idea:
    Pick the question that gives us the MOST information
    about the student's current ability level

    This is the same algorithm used by GRE/GMAT computerized tests
    """

    def __init__(self, scorer: IRTScorer):
        self.scorer = scorer

    def select_next(self,
                    state: StudentState,
                    question_bank: list[Question],
                    target_category: str = None) -> Question:
        """
        Select the best next question

        Steps:
        1. Filter out already-answered questions
        2. Filter by category if specified
        3. Score each remaining question by Fisher Information
        4. Return the highest-information question

        Args:
          state:           current student state (knows θ per topic)
          question_bank:   all available questions
          target_category: force a specific topic (for study plan)

        Returns: best Question object
        """

        # Filter answered questions
        available = [
            q for q in question_bank
            if q.id not in state.answered
        ]

        # Filter by category if specified
        if target_category:
            category_qs = [
                q for q in available
                if q.category == target_category
            ]
            # Fall back to all if category exhausted
            if category_qs:
                available = category_qs

        if not available:
            return None

        # Score each question by information at student's θ
        best_question  = None
        best_info      = -1.0

        for question in available:
            theta    = state.get_theta(question.category)
            info_val = self.scorer.information(theta, question)

            if info_val > best_info:
                best_info     = info_val
                best_question = question

        return best_question

    def select_weakest_category(self,
                                state: StudentState,
                                min_questions: int = 3) -> str:
        """
        Find the topic the student needs most work on
        Used by PlanAgent to generate study roadmap

        Only considers topics with at least min_questions answered
        (avoids recommending topics with too little data)
        """
        scored = {}
        for category, theta in state.theta.items():
            if state.total_count.get(category, 0) >= min_questions:
                scored[category] = theta

        if not scored:
            return None

        # Return category with lowest θ
        return min(scored, key=scored.get)

    def generate_study_plan(self,
                            state: StudentState,
                            categories: list[str]) -> list[dict]:
        """
        Generate prioritized study plan based on θ scores

        Categories with θ < 0.5 → high priority
        Categories with θ 0.5-1.0 → medium priority
        Categories with θ > 1.0 → low priority (review only)
        """
        plan = []
        for category in categories:
            theta    = state.get_theta(category)
            answered = state.total_count.get(category, 0)

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

            plan.append({
                "category": category,
                "theta":    round(theta, 2),
                "level":    self.scorer.level_label(theta),
                "priority": priority,
                "emoji":    emoji,
                "answered": answered,
                "accuracy": round(state.accuracy(category)*100, 1)
            })

        return sorted(plan, key=lambda x: x["theta"])