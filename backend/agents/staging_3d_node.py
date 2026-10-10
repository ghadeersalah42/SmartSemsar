from backend.schema.state import AgentState, PropertyItem

def staging_3d_node(state: AgentState) -> dict:
    print("\n📱 [Node] Generating 3d...")
    return {"agent_response": "3d done"}