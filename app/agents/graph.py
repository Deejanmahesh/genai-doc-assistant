from langgraph.graph import StateGraph, END
from app.agents.state import GraphState
from app.agents.nodes import router_node, retriever_node, generator_node, validator_node

def build_graph():
    workflow = StateGraph(GraphState)

    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("validator", validator_node)

    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "validator")

    workflow.add_conditional_edges(
        "validator",
        lambda state: "end" if state["is_valid"] else "retry",
        {"end": END, "retry": "generator"}
    )

    return workflow.compile()