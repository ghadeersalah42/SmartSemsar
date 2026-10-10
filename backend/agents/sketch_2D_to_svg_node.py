from backend.schema.state import AgentState, PropertyItem


def sketch_to_svg_node(state: AgentState) -> dict:
    print("\n📐 [Node] Converting 2D Sketch to SVG Format...")
    sketch_path = state.get("uploaded_sketch_path")
    # توضع هنا دالة التحويل الفعلية مستقبلاً
    return {"svg_data": f"<svg>Data from {sketch_path}</svg>"}