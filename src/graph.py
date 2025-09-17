from langgraph.graph import StateGraph, END
from nodes import node_clasificar, node_ejecutar_python, node_validar_y_decidir, node_responder, route_after_validation
from state import AgentState

def create_graph():
    graph = StateGraph(AgentState)

    # Nodos
    graph.add_node("clasificar", node_clasificar)
    graph.add_node("ejecutar_python", node_ejecutar_python)
    graph.add_node("validar", node_validar_y_decidir)
    graph.add_node("responder", node_responder)

    # Punto de entrada
    graph.set_entry_point("clasificar")

    # Flujo principal
    graph.add_edge("clasificar", "ejecutar_python")
    graph.add_edge("ejecutar_python", "validar")
    
    # Enrutamiento condicional desde validación
    graph.add_conditional_edges("validar", route_after_validation)
    
    # Fin del grafo
    graph.add_edge("responder", END)

    return graph.compile()
