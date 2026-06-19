from langgraph.graph import StateGraph, END
from agents.state import AdaptIQState
from agents.quiz_agent import quiz_agent
from agents.eval_agent import eval_agent
from agents.explain_agent import explain_agent
from agents.plan_agent import plan_agent

def route_after_eval(state: AdaptIQState) -> str:
    """Router: after EvalAgent decide next node"""
    return state.get("next_action", "plan")

def route_after_plan(state: AdaptIQState) -> str:
    """Router: after PlanAgent decide next node"""
    if state.get("session_complete"):
        return END
    return "quiz_agent"

def build_graph():
    """
    Build AdaptIQ LangGraph state machine

    What: Wires 4 agents into directed graph with routing
    Why:  LangGraph manages state flow + conditional edges
    How:
      Nodes: quiz → eval → (explain?) → plan → quiz...
      Edges: conditional based on state["next_action"]

    Graph structure:
      quiz_agent
          ↓
      eval_agent ──── correct ────→ plan_agent
          │                              │
          └──── wrong ──→ explain_agent ─┘
                                         │
                              session_complete?
                                    ↓ yes
                                   END
    """
    graph = StateGraph(AdaptIQState)

    # ── Add nodes ─────────────────────────────────────
    graph.add_node("quiz_agent",    quiz_agent)
    graph.add_node("eval_agent",    eval_agent)
    graph.add_node("explain_agent", explain_agent)
    graph.add_node("plan_agent",    plan_agent)

    # ── Entry point ───────────────────────────────────
    graph.set_entry_point("quiz_agent")

    # ── Edges ─────────────────────────────────────────
    # After quiz → always eval (waiting for answer)
    graph.add_edge("quiz_agent", "eval_agent")

    # After eval → conditional (correct=plan, wrong=explain)
    graph.add_conditional_edges(
        "eval_agent",
        route_after_eval,
        {
            "explain": "explain_agent",
            "plan":    "plan_agent"
        }
    )

    # After explain → always plan
    graph.add_edge("explain_agent", "plan_agent")

    # After plan → conditional (continue=quiz, done=END)
    graph.add_conditional_edges(
        "plan_agent",
        route_after_plan,
        {
            "quiz_agent": "quiz_agent",
            END:          END
        }
    )

    return graph.compile()

# Build the graph
adaptiq_graph = build_graph()
print("✅ AdaptIQ LangGraph compiled successfully!")

# Visualize structure
print("\nGraph nodes:", list(adaptiq_graph.nodes))