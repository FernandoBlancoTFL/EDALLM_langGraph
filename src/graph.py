from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import *
from checkpoints import postgres_saver, get_postgres_saver

def route_after_classification(state: AgentState):
    """Determina si ir a SQL_Executor, Python_Interpreter o responder directamente"""
    action = state.get("action", "Python_Interpreter")
    data_strategy = state.get("data_strategy", "dataframe")
    
    print(f"\n🔧 Routing después de clasificación:")
    print(f"   Action: {action}")
    print(f"   Strategy: {data_strategy}")
    
    # Manejar consultas de memoria
    if data_strategy == "memory":
        print("   → Routing to: responder (memoria)")
        return "responder"
    
    # Mapeo de acciones a nodos para consultas normales
    if action == "SQL_Executor" or (data_strategy == "sql" and action not in ["Python_Interpreter"]):
        print("   → Routing to: sql_executor")
        return "sql_executor"
    else:
        print("   → Routing to: ejecutar_python")
        return "ejecutar_python"

def route_after_validation_modificado(state: AgentState):
    """Routing que maneja fallbacks"""
    success = state.get("success", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    needs_fallback = state.get("needs_fallback", False)
    current_strategy = state.get("data_strategy", "dataframe")
    
    # print(f"\n🔧 DEBUG route_after_validation_modificado:")
    # print(f"   Success: {success}")
    # print(f"   Iteration: {iteration_count}")
    # print(f"   Strategy: {current_strategy}")
    # print(f"   Needs fallback: {needs_fallback}")
    
    if success:
        print("   → Routing to: responder (success)")
        return "responder"
    elif needs_fallback and current_strategy == "sql":
        print("   → Routing to: estrategia_datos (fallback)")
        return "estrategia_datos"  # Volver a evaluar estrategia
    elif iteration_count >= max_iterations:
        print("   → Routing to: responder (max iterations)")
        return "responder"
    else:
        print("   → Routing to: clasificar (continue)")
        return "clasificar"

def create_graph_with_sql():
    """Crea el grafo con los nuevos nodos SQL y PostgresSaver"""
    # global postgres_saver
    
    graph = StateGraph(AgentState)

    # Nodos existentes + nuevos
    graph.add_node("estrategia_datos", nodo_estrategia_datos)
    graph.add_node("clasificar", node_clasificar_modificado)
    graph.add_node("sql_executor", nodo_sql_executor)
    graph.add_node("ejecutar_python", node_ejecutar_python)
    graph.add_node("validar", node_validar_y_decidir_modificado)
    graph.add_node("responder", node_responder)

    # Punto de entrada
    graph.set_entry_point("estrategia_datos")

    # Flujo principal
    graph.add_edge("estrategia_datos", "clasificar")
    graph.add_conditional_edges("clasificar", route_after_classification)
    graph.add_edge("sql_executor", "validar")
    graph.add_edge("ejecutar_python", "validar")
    graph.add_conditional_edges("validar", route_after_validation_modificado)
    graph.add_edge("responder", END)

    saver = get_postgres_saver()
    if saver:
        print("🧠 Compilando grafo con memoria persistente")
        return graph.compile(checkpointer=saver)
    else:
        print("⚠️ Compilando grafo sin memoria persistente")
        return graph.compile()
