# 🧠 AdaptIQ — Adaptive AI Interview Coach

AdaptIQ is an adaptive AI Engineer interview prep system that combines **Item Response Theory (IRT)** — the same adaptive-testing algorithm behind the GRE, GMAT, and LSAT — with **RAG-grounded explanations** and a **LangGraph multi-agent architecture**.

It doesn't just quiz you. It builds a live model of your knowledge, generates fresh questions targeted to your exact skill level, and explains every wrong answer with citations pulled from real documentation — not hallucinated sources.

---

## Why This Is Different

Most interview prep tools are static question banks. AdaptIQ is not.

| | Static quiz apps | AdaptIQ |
|---|---|---|
| Question order | Fixed or random | IRT-selected by information gain |
| Difficulty | Same for everyone | Adapts per category, per student |
| New questions | Pre-written only | LLM-generated, targeted to your θ |
| Wrong answers | Generic explanation | RAG-cited from real docs |
| Study plan | None or generic | Shows exact topics missed + sources to read |
| Performance tracking | None | Live observability — latency, cost, accuracy |

No existing open-source project combines IRT + RAG + multi-agent orchestration for interview prep.

---

## How It Works

```
Session starts
    ↓
Phase 1 — Seed Calibration (Questions 1–9)
    Randomized selection from a difficulty band near
    the student's current ability (θ). Prevents
    memorization across sessions while still calibrating
    an accurate IRT estimate.
    ↓
Phase 2 — Dynamic Generation (Question 10+)
    An LLM generates a fresh question once θ is calibrated,
    targeted to the student's exact weak category and
    difficulty level. Never repeats.
    ↓
Wrong answer?
    ↓
RAG Explanation Engine
    Retrieves relevant chunks from real documentation
    (LangChain, HuggingFace) — not just static text.
    Excludes hallucinated sources by validating every
    citation against the retrieved context.
    ↓
Study Plan Update
    Tracks exactly which topics were missed and
    recommends a specific source to review.
    ↓
Observability Layer
    Every question logs latency, token cost, and θ
    delta to a JSONL file — visible live in the UI.
```

---

## Architecture

```
adaptiq/
├── irt/                  IRT scoring engine (3PL model)
│   ├── scorer.py          probability_correct, update_theta, score_answer
│   └── selector.py        question selection, study plan generation
│
├── questions/            Seed question bank (44 questions, 9 categories)
│   ├── bank.py             load/filter/summarize questions
│   ├── validator.py        schema + quality validation
│   └── questions.json
│
├── retrieval/            RAG pipeline
│   ├── ingester.py         fetch + chunk + embed docs into Qdrant
│   └── explainer.py        retrieve + generate cited explanations
│
├── agents/               LangGraph multi-agent orchestration
│   ├── state.py            shared AdaptIQState schema
│   ├── quiz_agent.py       question selection / generation
│   ├── eval_agent.py       IRT scoring of answers
│   ├── explain_agent.py    RAG explanation on wrong answers
│   ├── plan_agent.py       study plan + category routing
│   └── coordinator.py      LangGraph state machine wiring
│
├── memory/               Session persistence across runs
│   └── session_store.py
│
├── observability/        Per-question metrics tracking
│   └── tracker.py          latency, cost, accuracy, θ progression
│
└── app/                  Gradio UI
    └── gradio_app.py
```

### The Multi-Agent Flow

```
        ┌─────────────┐
        │ QuizAgent   │  selects/generates next question
        └──────┬──────┘
               ▼
        ┌─────────────┐
        │ EvalAgent   │  scores answer via IRT
        └──────┬──────┘
          correct │ wrong
        ┌─────────┴─────────┐
        ▼                   ▼
  ┌──────────┐       ┌───────────────┐
  │PlanAgent │       │ExplainAgent   │  RAG-grounded explanation
  └────┬─────┘       └───────┬───────┘
       │                     │
       └─────────┬───────────┘
                  ▼
            ┌──────────┐
            │PlanAgent │  updates study plan, picks next category
            └────┬─────┘
                 ▼
          session complete? → END
          else → QuizAgent (loop)
```

---

## Core Algorithm — IRT (3PL Model)

```
P(correct | θ, question) = c + (1-c) · sigmoid(1.7 · a · (θ - b))

θ = student ability        (estimated, updates after every answer)
b = question difficulty    (fixed per question, -1.5 to +1.5)
a = discrimination         (how well the question separates skill levels)
c = guessing probability   (0.25 for 4-option multiple choice)
```

After each answer, θ updates via:
```
θ_new = θ + learning_rate × (1 - P)   if correct
θ_new = θ - learning_rate × P         if wrong
```

This means a hard question answered correctly moves θ up more than an easy one — and an easy question answered wrong drops θ more sharply than a hard one missed. The system always selects the next question using **Fisher Information** — the question that will tell it the most about the student's true ability at their current θ.

---

## RAG Explanation Engine

When a student answers incorrectly:

1. The question + correct answer is used to query a **Qdrant** vector store containing chunks ingested from LangChain and HuggingFace documentation
2. Web-doc chunks are prioritized over the static question-bank fallback
3. If fewer than 2 relevant web chunks are found, the search broadens beyond the category filter
4. The LLM (Groq `llama-3.1-8b-instant`) generates an explanation **constrained to cite only the exact source names present in the retrieved context** — a hallucination-detection step flags any invented source names
5. The explanation always includes a specific study tip

```
Example output:
"The ReAct framework interleaves reasoning and acting in a loop
[Source: LangChain Docs — Agents]. The agent alternates Thought →
Action → Observation, grounding decisions in real tool outputs..."
```

---

## Live Observability

Every question logs:
- **Latency** — p50/p95 response time, isolated from LLM generation time
- **Cost** — token usage × Groq pricing, shown per question and per session
- **Accuracy** — live bar chart per category
- **θ progression** — last 6 questions, with direction and magnitude of change

This is visible directly in the Gradio UI, not just buried in logs.

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/PavanKAgnihotri/AdaptIQ.git
cd AdaptIQ
pip install -r requirements.txt
```

### 2. Add your Groq API key

```bash
cp .env.example .env
# Edit .env and paste your key:
# GROQ_API_KEY=gsk_your_real_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 3. Run the app

```bash
python app/gradio_app.py
```

This launches a Gradio interface with a shareable public link.

---

## Tech Stack

```
Language:        Python
IRT Math:        NumPy
Multi-Agent:     LangGraph
LLM:             Groq (llama-3.1-8b-instant)
Embeddings:      BAAI/bge-small-en-v1.5 (Sentence Transformers)
Vector Store:    Qdrant (in-memory)
UI:              Gradio
Persistence:     JSON session state, JSONL observability logs
```

---

## What's Next

- Streamlit deployment for permanent public hosting
- Expand seed bank beyond 44 questions per category
- Multi-student leaderboard via shared storage
- Voice-based interview simulation mode

---

## Author

**Pavan Keshav Agnihotri**
MS Computer Science, University of Alabama at Birmingham
[GitHub](https://github.com/PavanKAgnihotri) · [LinkedIn](www.linkedin.com/in/pavankeshavagnihotri)
