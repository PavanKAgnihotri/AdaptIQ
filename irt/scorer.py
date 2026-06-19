import numpy as np
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Question:
    """Represents one interview question with IRT parameters"""
    id:             str
    text:           str
    options:        list[str]        # A, B, C, D
    correct:        str              # "A", "B", "C", or "D"
    category:       str              # "RAG", "Agents", "Python" etc
    difficulty:     float            # b: -3 to +3
    discrimination: float = 1.0     # a: how well it separates students
    guessing:       float = 0.25    # c: 4 options = 25% guess rate
    explanation:    str  = ""       # shown on wrong answer


@dataclass
class StudentState:
    """Tracks a student's knowledge state across all topics"""
    student_id:    str
    theta:         dict = field(default_factory=dict)   # topic → ability
    answered:      list = field(default_factory=list)   # question ids seen
    correct_count: dict = field(default_factory=dict)   # topic → correct
    total_count:   dict = field(default_factory=dict)   # topic → total
    session_log:   list = field(default_factory=list)   # full history

    def get_theta(self, category: str) -> float:
        """Get ability score for a topic, default 0.0"""
        return self.theta.get(category, 0.0)

    def accuracy(self, category: str) -> float:
        """Get accuracy percentage for a topic"""
        total = self.total_count.get(category, 0)
        if total == 0:
            return 0.0
        return self.correct_count.get(category, 0) / total


class IRTScorer:
    """
    IRT Model

    The same math used by GRE, GMAT, LSAT
    Adapted for AI Engineer interview questions
    """

    def probability_correct(self, theta: float,
                            question: Question) -> float:
        """
        P(correct | theta, question) — core IRT formula

        P = c + (1-c) × sigmoid(1.7 × a × (θ - b))

        Where:
          θ (theta) = student ability
          b = question difficulty
          a = discrimination
          c = guessing probability
          1.7 = scaling constant (standard in IRT)

        Example:
          theta=0.0, b=0.5, a=1.0, c=0.25
          → P = 0.25 + 0.75 × sigmoid(1.7×1.0×(0-0.5))
          → P = 0.25 + 0.75 × sigmoid(-0.85)
          → P = 0.25 + 0.75 × 0.299
          → P = 0.474 (47% chance of getting it right)
        """
        a = question.discrimination
        b = question.difficulty
        c = question.guessing

        exponent = 1.7 * a * (theta - b)
        sigmoid  = 1.0 / (1.0 + np.exp(-exponent))

        return c + (1.0 - c) * sigmoid

    def information(self, theta: float,
                    question: Question) -> float:
        """
        Fisher Information — how much does this question
        tell us about the student's ability level?

        High information → question is well-targeted for this θ
        Low information  → question too easy or too hard

        Used by selector.py to pick the BEST next question
        """
        p = self.probability_correct(theta, question)
        q = 1.0 - p
        a = question.discrimination
        c = question.guessing

        # IRT information formula
        numerator   = (a ** 2) * ((p - c) ** 2)
        denominator = ((1.0 - c) ** 2) * p * q

        if denominator < 1e-10:
            return 0.0

        return numerator / denominator

    def update_theta(self, theta: float,
                     question: Question,
                     is_correct: bool,
                     learning_rate: float = 0.3) -> float:
        """
        Update student ability after answering a question

        Uses simplified MLE update (Maximum Likelihood Estimation)
        Full MLE is expensive — this converges fast enough for
        real-time adaptive testing

        is_correct=True  → θ moves UP   (got harder question right)
        is_correct=False → θ moves DOWN (got easier question wrong)

        The AMOUNT of change depends on:
        - How surprising the answer was (P vs actual)
        - Question discrimination (high a = bigger update)

        Example:
          theta=0.0, question difficulty=0.5
          Expected P = 0.47 (47% chance correct)

          Got it RIGHT (surprising! expected <50%):
          → Big upward update → theta = 0.0 + 0.3×(1-0.47) = +0.16

          Got it WRONG (not surprising, hard question):
          → Small downward update → theta = 0.0 - 0.3×0.47 = -0.14
        """
        p = self.probability_correct(theta, question)

        if is_correct:
            # Correct: move theta up
            # Bigger jump if question was hard (low p)
            delta = learning_rate * (1.0 - p)
        else:
            # Wrong: move theta down
            # Bigger drop if question was easy (high p)
            delta = -learning_rate * p

        # Clamp to reasonable range
        new_theta = theta + delta
        return float(np.clip(new_theta, -3.0, 3.0))

    def level_label(self, theta: float) -> str:
        """Convert θ score to human-readable level"""
        if theta >= 1.5:   return "Expert 🏆"
        if theta >= 0.8:   return "Strong ✅"
        if theta >= 0.3:   return "Good 👍"      
        if theta >= 0.0:   return "Average 📈"   
        if theta >= -0.8:  return "Weak ⚠️"      
        return "Beginner 🔰"

    def score_answer(self,
                     state: StudentState,
                     question: Question,
                     student_answer: str) -> dict:
        """
        Main function — called after every answer

        Returns:
          is_correct:  bool
          old_theta:   float (before update)
          new_theta:   float (after update)
          delta:       float (change in ability)
          p_correct:   float (probability model expected)
          level:       str   (human label)
          needs_explanation: bool (True if wrong)
        """
        is_correct = student_answer.strip().upper() == \
                     question.correct.strip().upper()

        category  = question.category
        old_theta = state.get_theta(category)

        # Update theta for this category
        new_theta = self.update_theta(old_theta, question, is_correct)
        state.theta[category] = new_theta

        # Update counts
        state.answered.append(question.id)
        state.total_count[category]   = \
            state.total_count.get(category, 0) + 1
        state.correct_count[category] = \
            state.correct_count.get(category, 0) + (1 if is_correct else 0)

        result = {
            "is_correct":         is_correct,
            "old_theta":          round(old_theta, 3),
            "new_theta":          round(new_theta, 3),
            "delta":              round(new_theta - old_theta, 3),
            "p_correct":          round(
                self.probability_correct(old_theta, question), 3),
            "level":              self.level_label(new_theta),
            "category":           category,
            "needs_explanation":  not is_correct,
            "correct_answer":     question.correct,
        }

        # Log to session history
        state.session_log.append({
            "question_id": question.id,
            "category":    category,
            **result
        })

        return result