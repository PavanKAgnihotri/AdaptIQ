from irt.scorer import Question
from dataclasses import dataclass


@dataclass
class ValidationResult:
    is_valid:  bool
    errors:    list[str]
    warnings:  list[str]
    question_id: str


class QuestionValidator:
    """
    Validates questions before adding to bank or using in session
    
    What:  Checks every question meets quality standards
    Why:   Bad questions break IRT scoring and confuse students
    How:   Rule-based checks on each Question field
    
    Used for:
    - Validating seed bank on load
    - Validating dynamically generated questions (Step 4)
    """

    VALID_CATEGORIES = [
        "Python", "Classical ML", "Deep Learning",
        "NLP & Transformers", "LLMs", "RAG Systems",
        "Agents", "MLOps", "System Design"
    ]

    VALID_CORRECT = {"A", "B", "C", "D"}

    def validate(self, question: Question) -> ValidationResult:
        """
        Full validation of one question
        
        Returns ValidationResult with is_valid, errors, warnings
        """
        errors   = []
        warnings = []

        # ── Hard rules (errors) ───────────────────────────────
        # Must have exactly 4 options
        if len(question.options) != 4:
            errors.append(
                f"Must have exactly 4 options, got {len(question.options)}"
            )

        # Options must start with A. B. C. D.
        for i, opt in enumerate(question.options):
            expected_prefix = ["A.", "B.", "C.", "D."][i]
            if not opt.strip().startswith(expected_prefix):
                errors.append(
                    f"Option {i+1} must start with '{expected_prefix}', got: {opt[:20]}"
                )

        # Correct answer must be A/B/C/D
        if question.correct.upper() not in self.VALID_CORRECT:
            errors.append(
                f"Correct answer must be A/B/C/D, got: '{question.correct}'"
            )

        # Category must be valid
        if question.category not in self.VALID_CATEGORIES:
            errors.append(
                f"Invalid category: '{question.category}'. "
                f"Valid: {self.VALID_CATEGORIES}"
            )

        # Difficulty must be in range
        if not -2.0 <= question.difficulty <= 2.0:
            errors.append(
                f"Difficulty must be -2.0 to 2.0, got: {question.difficulty}"
            )

        # Question text must not be empty
        if not question.text or len(question.text.strip()) < 10:
            errors.append("Question text too short or empty")

        # ID must be present
        if not question.id:
            errors.append("Question ID is required")

        # ── Soft rules (warnings) ─────────────────────────────
        # Explanation should be present
        if not question.explanation or len(question.explanation.strip()) < 20:
            warnings.append(
                "Explanation is missing or too short — "
                "RAG explanation engine will be triggered"
            )

        # Discrimination should be reasonable
        if not 0.5 <= question.discrimination <= 2.0:
            warnings.append(
                f"Discrimination {question.discrimination} is outside "
                f"typical range (0.5-2.0)"
            )

        # Options should be meaningfully different length
        lengths = [len(opt) for opt in question.options]
        if max(lengths) - min(lengths) < 5:
            warnings.append(
                "All options are very similar length — "
                "may be too easy to guess by elimination"
            )

        return ValidationResult(
            is_valid     = len(errors) == 0,
            errors       = errors,
            warnings     = warnings,
            question_id  = question.id
        )

    def validate_bank(self,
                      questions: list[Question],
                      strict: bool = False) -> list[Question]:
        """
        Validate entire question bank
        
        Args:
          questions: list of Question objects
          strict:    if True, remove questions with warnings too
        
        Returns: list of valid questions only
        
        What:  Runs validate() on every question
        Why:   Catch bad questions before they enter IRT session
        How:   Filter out invalid, print summary
        """
        valid     = []
        invalid   = []
        all_warns = []

        for q in questions:
            result = self.validate(q)

            if result.is_valid:
                if strict and result.warnings:
                    invalid.append((q, result))
                else:
                    valid.append(q)
                    if result.warnings:
                        all_warns.append((q.id, result.warnings))
            else:
                invalid.append((q, result))

        # Print report
        print(f"\n{'='*50}")
        print(f"VALIDATION REPORT")
        print(f"{'='*50}")
        print(f"Total:   {len(questions)}")
        print(f"Valid:   {len(valid)}")
        print(f"Invalid: {len(invalid)}")

        if invalid:
            print(f"\n❌ Invalid questions:")
            for q, result in invalid:
                print(f"\n  ID: {q.id}")
                for err in result.errors:
                    print(f"    ERROR: {err}")

        if all_warns:
            print(f"\n⚠️  Warnings:")
            for qid, warns in all_warns:
                for w in warns:
                    print(f"  {qid}: {w}")

        print(f"{'='*50}\n")
        return valid