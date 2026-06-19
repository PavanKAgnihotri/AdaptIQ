import random, json, time, os
from agents.state import AdaptIQState
from groq import Groq
from questions.bank import get_topics_for_category, get_categories
from irt.scorer import Question
from irt.scorer import IRTScorer
from irt.selector import QuestionSelector
from questions.bank import load_questions

SEED_PHASE_COUNT = 9   # questions 1-9 from randomized seed bank
                        # questions 10+ generated dynamically by LLM

_scorer = IRTScorer()
selector = QuestionSelector(_scorer)

all_questions = load_questions("questions/questions.json")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

DYNAMIC_QUESTION_PROMPT = """Generate ONE multiple choice interview question for an AI Engineer interview.

Category: {category}
Target difficulty: {difficulty} (scale: -1.5=easy to 1.5=hard)
Student current ability (θ): {theta} — generate SLIGHTLY above this
Topic to focus on: {topic}
Do NOT repeat these topics: {avoid_topics}

CRITICAL: Return ONLY valid JSON, no markdown, no explanation outside JSON:
{{
  "text": "Your question here?",
  "options": [
    "A. First option",
    "B. Second option",
    "C. Third option",
    "D. Fourth option"
  ],
  "correct": "A",
  "explanation": "Why the correct answer is right, and why others are wrong.",
  "difficulty": 0.5,
  "topic": "specific topic this question covers"
}}"""

def get_answered_topics(student) -> list[str]:
    """Extract topics already covered from session log"""
    topics = []
    for record in student.session_log[-10:]:
        if "category" in record:
            topics.append(record.get("category", ""))
    return list(set(topics))


def quiz_agent(state: AdaptIQState) -> AdaptIQState:
    """
    QuizAgent — Two-phase question selection

    Phase 1 (Questions 1 to SEED_PHASE_COUNT):
      Randomized selection from seed bank
      Picks from difficulty band near current θ
      Prevents memorization — different Q each session
      IRT calibration: builds accurate θ estimate

    Phase 2 (Questions SEED_PHASE_COUNT+1 onwards):
      LLM generates fresh question targeted to exact θ
      Topic chosen from weakest area
      Never repeats — infinite variety
      This is the core unique feature of AdaptIQ
    """

    student  = state["student_state"]
    category = state.get("target_category")
    asked    = state["questions_asked"]
    theta    = student.get_theta(category) if category else 0.0

    # ════════════════════════════════════════════════════
    # PHASE 1: Randomized Seed Questions (IRT Calibration)
    # ════════════════════════════════════════════════════
    if asked < SEED_PHASE_COUNT:

        available = [
            q for q in all_questions
            if q.id not in student.answered
            and (q.category == category if category else True)
        ]

        if available:
            # Target difficulty slightly above current θ
            # Creates optimal challenge — not too easy, not too hard
            target_b = theta + 0.2

            # Band selection — prefer closest to target difficulty
            band1 = [q for q in available
                     if abs(q.difficulty - target_b) <= 0.5]
            band2 = [q for q in available
                     if abs(q.difficulty - target_b) <= 1.0]
            pool  = band1 if band1 else (band2 if band2 else available)

            # Random pick within band
            q = random.choice(pool)

            label = f"Question {asked+1} [{q.category}] (difficulty: {q.difficulty:.1f})"

            return {
                **state,
                "current_question": q,
                "student_answer":   None,
                "is_correct":       None,
                "score_result":     None,
                "explanation":      None,
                "sources_cited":    None,
                "message": (
                    f"\n{'='*55}\n"
                    f"{label}\n"
                    f"{'='*55}\n"
                    f"{q.text}\n\n"
                    + "\n".join(q.options)
                ),
                "next_action": "await_answer"
            }

    # ════════════════════════════════════════════════════
    # PHASE 2: Dynamic LLM Generation (Personalized)
    # θ is now calibrated — generate targeted questions
    # ════════════════════════════════════════════════════
    if category:
        theta         = student.get_theta(category)
        topic_list    = get_topics_for_category(category)
        avoid_topics  = get_answered_topics(student)

        # Filter topics already covered
        fresh_topics = [t for t in topic_list
                        if t not in avoid_topics]
        topic = random.choice(fresh_topics if fresh_topics
                              else topic_list)

        # Target slightly above θ — optimal challenge
        target_difficulty = round(min(theta + 0.3, 1.5), 1)
        import time as _time
        gen_start = _time.perf_counter()
        print(f"  ⚡ Generating dynamic question...")
        print(f"     Category: {category} | θ={theta:.2f} | "
              f"Target difficulty: {target_difficulty}")
        print(f"     Topic: {topic}")

        prompt = DYNAMIC_QUESTION_PROMPT.format(
            category     = category,
            difficulty   = target_difficulty,
            theta        = round(theta, 2),
            topic        = topic,
            avoid_topics = ", ".join(avoid_topics[:5]) or "none"
        )

        try:
            response = groq_client.chat.completions.create(
                model    = "llama-3.1-8b-instant",
                messages = [
                    {"role": "system",
                     "content": (
                         "You are an expert AI/ML interview question generator. "
                         "Generate challenging, accurate questions. "
                         "Return ONLY valid JSON — no markdown, no extra text."
                     )},
                    {"role": "user", "content": prompt}
                ],
                temperature = 0.7,
                max_tokens  = 500
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            data = json.loads(raw)

            # Validate required fields
            required = ["text", "options", "correct", "explanation"]
            for field in required:
                if field not in data:
                    raise ValueError(f"Missing field: {field}")
            if len(data["options"]) != 4:
                raise ValueError("Must have exactly 4 options")
            if data["correct"] not in ["A", "B", "C", "D"]:
                raise ValueError(f"Invalid correct: {data['correct']}")

            dynamic_q = Question(
                id             = f"dyn_{category[:3].lower()}_{asked}",
                text           = data["text"],
                options        = data["options"],
                correct        = data["correct"],
                category       = category,
                difficulty     = float(data.get("difficulty",
                                                 target_difficulty)),
                discrimination = 1.0,
                guessing       = 0.25,
                explanation    = data.get("explanation", "")
            )
            
            gen_end = _time.perf_counter()
            gen_ms = round((gen_end - gen_start) * 1000, 2)
            print(f"     ⏱️  Generation time: {gen_ms}ms")
            print(f"     ✅ Generated: '{dynamic_q.text[:60]}...'")

            return {
                **state,
                "current_question": dynamic_q,
                "generation_time_ms": round(                    # ← add this
                (gen_end - gen_start) * 1000, 2
                  ) if 'gen_start' in dir() else 0,
                "student_answer":   None,
                "is_correct":       None,
                "score_result":     None,
                "explanation":      None,
                "sources_cited":    None,
                "message": (
                    f"\n{'='*55}\n"
                    f"Question {asked+1} "
                    f"[{category}] ⚡ AI-Generated "
                    f"(difficulty: {dynamic_q.difficulty:.1f})\n"
                    f"{'='*55}\n"
                    f"{dynamic_q.text}\n\n"
                    + "\n".join(dynamic_q.options)
                ),
                "next_action": "await_answer"
            }

        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSON parse failed: {e}")
            print(f"  Raw response: {raw[:200]}")
        except ValueError as e:
            print(f"  ⚠️  Validation failed: {e}")
        except Exception as e:
            print(f"  ⚠️  Generation failed: {e}")

        # ── Fallback: remaining seed questions ────────────
        print(f"  ↩️  Falling back to seed bank...")
        fallback = [
            q for q in all_questions
            if q.id not in student.answered
            and q.category == category
        ]
        if not fallback:
            fallback = [
                q for q in all_questions
                if q.id not in student.answered
            ]

        if fallback:
            q = random.choice(fallback)
            return {
                **state,
                "current_question": q,
                "student_answer":   None,
                "is_correct":       None,
                "score_result":     None,
                "explanation":      None,
                "sources_cited":    None,
                "message": (
                    f"\n{'='*55}\n"
                    f"Question {asked+1} [{q.category}] "
                    f"(difficulty: {q.difficulty:.1f})\n"
                    f"{'='*55}\n"
                    f"{q.text}\n\n"
                    + "\n".join(q.options)
                ),
                "next_action": "await_answer"
            }

    # No questions available — end session
    plan = selector.generate_study_plan(
        student, get_categories(all_questions)
    )
    return {
        **state,
        "session_complete": True,
        "study_plan":       plan,
        "message":          "Session complete!",
        "next_action":      "end"
    }