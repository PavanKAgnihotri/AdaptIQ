import json
from pathlib import Path
from irt.scorer import Question


def load_questions(filepath: str = "questions/questions.json") -> list[Question]:
    """
    Load questions from JSON file and convert to Question dataclass objects
    
    What:  Reads JSON → returns list of Question objects IRT can use
    Why:   IRT scorer and selector work with Question dataclass not raw dicts
    How:   json.load → dict → Question(**dict) for each entry
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Question bank not found at {filepath}")

    with open(path, "r") as f:
        raw = json.load(f)

    questions = []
    for item in raw:
        try:
            q = Question(
                id             = item["id"],
                text           = item["text"],
                options        = item["options"],
                correct        = item["correct"],
                category       = item["category"],
                difficulty     = float(item["difficulty"]),
                discrimination = float(item.get("discrimination", 1.0)),
                guessing       = float(item.get("guessing", 0.25)),
                explanation    = item.get("explanation", "")
            )
            questions.append(q)
        except KeyError as e:
            print(f"Warning: skipping question {item.get('id','?')} — missing field {e}")

    print(f"Loaded {len(questions)} questions from {filepath}")
    return questions


def get_by_category(questions: list[Question],
                    category: str) -> list[Question]:
    """Filter questions by category"""
    return [q for q in questions if q.category == category]


def get_by_difficulty_range(questions: list[Question],
                             low: float,
                             high: float) -> list[Question]:
    """
    Filter questions within a difficulty range
    
    Used by QuizAgent (Step 4) to find questions near target difficulty
    """
    return [q for q in questions if low <= q.difficulty <= high]


def get_categories(questions: list[Question]) -> list[str]:
    """Return sorted list of unique categories"""
    return sorted(set(q.category for q in questions))


def summary(questions: list[Question]):
    """Print summary of question bank"""
    cats = {}
    for q in questions:
        cats[q.category] = cats.get(q.category, 0) + 1

    print(f"\n{'='*50}")
    print(f"QUESTION BANK SUMMARY")
    print(f"{'='*50}")
    print(f"Total questions: {len(questions)}")
    print(f"\nPer category:")
    for cat, count in sorted(cats.items()):
        bar = "█" * count
        print(f"  {cat:<25} {bar} {count}")

    diffs = [q.difficulty for q in questions]
    import numpy as np
    print(f"\nDifficulty distribution:")
    print(f"  Min:  {min(diffs):.2f}")
    print(f"  Max:  {max(diffs):.2f}")
    print(f"  Mean: {np.mean(diffs):.2f}")
    print(f"  Easy  (< -0.5): {sum(1 for d in diffs if d < -0.5)}")
    print(f"  Med   (-0.5 to 0.5): {sum(1 for d in diffs if -0.5 <= d <= 0.5)}")
    print(f"  Hard  (> 0.5): {sum(1 for d in diffs if d > 0.5)}")
    print(f"{'='*50}\n")

#questions/bank.py

CATEGORY_TOPICS = {
    "Python": [
        "Python generators decorators",
        "Python asyncio concurrency",
        "Python OOP metaclasses",
        "Python memory management GIL",
    ],
    "Classical ML": [
        "bias variance tradeoff regularization",
        "ensemble methods random forest XGBoost",
        "SVM kernel methods",
        "cross validation metrics imbalanced",
    ],
    "Deep Learning": [
        "backpropagation vanishing gradient",
        "batch normalization dropout regularization",
        "CNN RNN LSTM architecture",
        "residual networks skip connections",
    ],
    "NLP & Transformers": [
        "attention mechanism transformer architecture",
        "BERT GPT encoder decoder",
        "tokenization embeddings positional encoding",
        "multi-head attention self-attention",
    ],
    "LLMs": [
        "RLHF fine-tuning alignment",
        "LoRA QLoRA parameter efficient fine-tuning",
        "chain of thought prompting",
        "LLM hallucination temperature sampling",
    ],
    "RAG Systems": [
        "RAG retrieval augmented generation pipeline",
        "hybrid retrieval BM25 vector search RRF",
        "cross encoder reranking",
        "RAGAS evaluation faithfulness",
    ],
    "Agents": [
        "ReAct reasoning acting framework",
        "LangGraph stateful agents",
        "multi-agent coordination supervisor",
        "tool use function calling MCP",
    ],
    "MLOps": [
        "model drift monitoring production ML",
        "model deployment serving FastAPI",
        "quantization distillation compression",
        "feature store ML pipeline CI CD",
    ],
    "System Design": [
        "RAG system design architecture",
        "LLM application scaling",
        "semantic caching guardrails",
        "multi-agent production deployment",
    ],
}


def get_topics_for_category(category: str) -> list[str]:
    """
    Returns search topics for a category
    Used by RAG engine to fetch relevant explanations
    """
    return CATEGORY_TOPICS.get(category, [category])