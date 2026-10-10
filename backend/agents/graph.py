"""graph.py: assembles the LangGraph state machine (compiled_graph).

Only the 3D-model leg of the pipeline is wired so far:

    staging_3d_node -> routing_node -> furniture_node   (model_state ready) -> END
                                     -> staging_3d_node  (retry)
                                     -> END               (error)

furniture_node reads furniture_state (wishes, planner, the user's own pieces) and writes
what it placed back into it: furniture_state.walkthrough_path is the furnished apartment.

input_node / matching_node fill selected_property and feed into this leg;
they are still being built on their own branches and are not wired in yet.
"""
from langgraph.graph import END, StateGraph

from backend.agents.furniture_node import furniture_node
from backend.agents.routing_node import routing_node
from backend.agents.staging_3d_node import staging_3d_node
from backend.schema.state import GraphState


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("3d_staging", staging_3d_node)
    graph.add_node("routing", routing_node)
    graph.add_node("furniture", furniture_node)

    graph.set_entry_point("3d_staging")
    graph.add_edge("3d_staging", "routing")

    graph.add_conditional_edges(
        "routing",
        lambda state: state.get("next_step"),
        {
            "furniture": "furniture",
            "retry": "3d_staging",
            "error": END,
        },
    )
    graph.add_edge("furniture", END)

    return graph


compiled_graph = build_graph().compile()
