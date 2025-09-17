from typing import TypedDict, List, Any, Optional

class AgentState(TypedDict):
    query: str
    action: str
    result: Any
    thought: str
    history: List[str]
    execution_history: List[dict]  # Nuevo: historial detallado de ejecuciones
    iteration_count: int           # Nuevo: contador de iteraciones
    max_iterations: int           # Nuevo: límite de iteraciones
    df_info: dict                 # Nuevo: información cached del DataFrame
    success: bool                 # Nuevo: flag de éxito
    final_error: Optional[str]    # Nuevo: error final si no se pudo resolver
