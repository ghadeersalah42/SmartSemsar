from backend.schema.state import AgentState, PropertyItem

def furniture_node(state: AgentState) -> dict:
    print("\n🛋️ [Node] Applying Furniture and Interior Design...")
    return {"furnished_result": {"status": "success", "furniture_layout": "Modern Style"}}