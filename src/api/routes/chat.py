from fastapi import APIRouter, HTTPException
from datetime import datetime
from state import AgentState
from utils import extract_plot_filename_from_result, get_plot_metadata
import sys
import os

# Agregar src al path para poder importar los módulos existentes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from api.schemas.chat import ChatRequest, ChatResponse
from config import SINGLE_USER_THREAD_ID, SINGLE_USER_ID
from graph import create_graph_with_sql
from checkpoints import get_postgres_saver
from utils import clean_state_for_serialization

from src.services.chat_service import (
    get_conversation_history
)
from src.api.schemas.chat import (
    ChatHistoryResponse,
    ChatHistoryItem
)

router = APIRouter()

# Inicializar el grafo globalmente (se crea una sola vez)
app_graph = None

def get_graph():
    """Obtiene o inicializa el grafo de análisis"""
    global app_graph
    if app_graph is None:
        app_graph = create_graph_with_sql()
    return app_graph

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal de chat que procesa consultas del usuario.
    
    - Usa el thread único del sistema (single user)
    - Mantiene historial conversacional automáticamente
    - Retorna respuesta interpretativa con metadatos de ejecución
    """
    
    try:
        # Obtener el grafo compilado
        graph = get_graph()
        postgres_saver = get_postgres_saver()
        
        # Estado inicial
        initial_state = {
            "query": request.message,
            "action": "",
            "result": None,
            "thought": "",
            "history": [],
            "execution_history": [],
            "iteration_count": 0,
            "max_iterations": 3,
            "df_info": {},
            "success": False,
            "final_error": None,
            "next_node": "clasificar",
            "available_datasets": {},
            "selected_dataset": None,
            "active_dataframe": None,
            "dataset_context": {},
            "data_strategy": "dataframe",
            "sql_feasible": False,
            "table_metadata": {},
            "strategy_history": [],
            "sql_results": None,
            "strategy_switched": False,
            "needs_fallback": False,
            "strategy_reason": "",
            "sql_error": None,
            "session_metadata": {
                "thread_id": SINGLE_USER_THREAD_ID,
                "session_start": datetime.now().isoformat(),
                "user_id": SINGLE_USER_ID
            }
        }
        
        # Configurar thread único para memoria persistente
        config = {
            "configurable": {"thread_id": SINGLE_USER_THREAD_ID}
        } if postgres_saver else {}
        
        # Invocar el grafo (ejecuta todo el flujo de análisis)
        final_state = graph.invoke(initial_state, config=config)

        # Limpiar estado para serialización (asegurarse de NO eliminar response_metadata)
        final_state = clean_state_for_serialization(final_state)
        
        # Determinar tipo de respuesta
        response_type = determine_response_type(final_state)
        
        # Extraer datos adicionales según el tipo
        response_data = extract_response_data(final_state, response_type)
        
        # Construir respuesta
        return ChatResponse(
            response=final_state.get("llm_response", final_state.get("result", "No se pudo procesar la consulta")),
            type=response_type,
            data=response_data,
            sql_query=extract_sql_query(final_state),
            success=final_state.get("success", False),
            iterations=final_state.get("iteration_count", 0),
            strategy_used=final_state.get("data_strategy", "unknown")
        )
        
    except Exception as e:
        # Manejo de errores
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Error procesando la consulta"
            }
        )

def determine_response_type(state: dict) -> str:
    """
    Determina el tipo de respuesta basándose en el estado final.
    MEJORADO: Detecta gráficos por archivos realmente creados, no por código.
    """
    if not state.get("success", False):
        return "error"
    
    # NUEVO: Verificar si realmente se creó un archivo de gráfico
    plot_file = find_recently_created_plot()
    if plot_file:
        print(f"📊 Gráfico detectado: {plot_file}")
        return "plot"
    
    # Verificar si hay datos tabulares (SQL o DataFrame)
    if state.get("sql_results"):
        sql_results = state["sql_results"]
        if isinstance(sql_results, dict) and sql_results.get("data"):
            return "table"
    
    # Por defecto, respuesta de texto
    return "text"

def find_recently_created_plot(time_window_seconds: int = 10) -> str:
    """
    Busca archivos PNG creados recientemente en outputs/.
    
    Args:
        time_window_seconds: Ventana de tiempo en segundos para considerar "reciente"
    
    Returns:
        Nombre del archivo si se encontró, None en caso contrario
    """
    try:
        import time
        outputs_dir = "./src/outputs"
        
        if not os.path.exists(outputs_dir):
            return None
        
        current_time = time.time()
        png_files = [f for f in os.listdir(outputs_dir) if f.endswith('.png')]
        
        # Buscar archivos creados en los últimos X segundos
        for filename in png_files:
            filepath = os.path.join(outputs_dir, filename)
            file_mtime = os.path.getmtime(filepath)
            
            # Si fue modificado/creado recientemente
            if current_time - file_mtime <= time_window_seconds:
                return filename
        
        return None
        
    except Exception as e:
        print(f"⚠️ Error buscando archivos recientes: {e}")
        return None

def prepare_response_metadata(state: AgentState) -> dict:
    """
    Prepara metadata completa de la respuesta para guardar en checkpoint.
    Detecta tipo de respuesta y extrae datos relevantes.
    """
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "query": state.get("query"),
        "success": state.get("success", False),
        "strategy_used": state.get("data_strategy", "unknown"),
        "iterations": state.get("iteration_count", 0)
    }
    
    # Detectar tipo de respuesta
    if state.get("success", False):
        # Buscar si hay gráfico
        plot_file = find_recently_created_plot()
        
        if plot_file:
            plot_metadata = get_plot_metadata(plot_file)
            base_url = "http://localhost:8000"
            
            metadata["type"] = "plot"
            metadata["data"] = {
                "url": f"{base_url}/outputs/{plot_file}",
                "filename": plot_file,
                "created_at": plot_metadata.get("created_at"),
                "size_bytes": plot_metadata.get("size_bytes"),
                "exists": plot_metadata.get("exists")
            }
        
        # Verificar si hay resultados SQL
        elif state.get("sql_results"):
            sql_results = state["sql_results"]
            if isinstance(sql_results, dict) and sql_results.get("data"):
                metadata["type"] = "table"
                metadata["data"] = {
                    "rows": sql_results.get("data", [])[:50],
                    "columns": sql_results.get("columns", []),
                    "total_rows": len(sql_results.get("data", []))
                }
        
        # Respuesta de texto
        else:
            metadata["type"] = "text"
            metadata["data"] = None
    
    else:
        # Error
        metadata["type"] = "error"
        metadata["data"] = {
            "error_message": state.get("final_error", "Error desconocido"),
            "attempts": state.get("iteration_count", 0)
        }
    
    return metadata

def extract_response_data(state: dict, response_type: str):
    """
    Extrae datos adicionales según el tipo de respuesta.
    MEJORADO: Usa detección por archivo real en lugar de código.
    """
    if response_type == "plot":
        # Primero intentar encontrar el archivo más reciente
        plot_filename = find_recently_created_plot()
        
        # Si no se encontró por timestamp, buscar en los resultados
        if not plot_filename:
            # Método 1: Buscar en execution_history
            for record in state.get("execution_history", []):
                if record.get("success"):
                    result_text = record.get("result", "")
                    plot_filename = extract_plot_filename_from_result(result_text)
                    if plot_filename:
                        print(f"🔍 Archivo encontrado en execution_history: {plot_filename}")
                        break
            
            # Método 2: Buscar en result directo
            if not plot_filename and state.get("result"):
                plot_filename = extract_plot_filename_from_result(state["result"])
                if plot_filename:
                    print(f"🔍 Archivo encontrado en result: {plot_filename}")
            
            # Método 3: Buscar en llm_response
            if not plot_filename and state.get("llm_response"):
                plot_filename = extract_plot_filename_from_result(state["llm_response"])
                if plot_filename:
                    print(f"🔍 Archivo encontrado en llm_response: {plot_filename}")
        
        if plot_filename:
            # Obtener metadata del archivo
            metadata = get_plot_metadata(plot_filename)
            
            # Verificar que el archivo realmente existe
            if not metadata.get("exists"):
                print(f"❌ Archivo no existe: {plot_filename}")
                return {"error": "El archivo del gráfico no existe"}
            
            # Construir URL completa
            base_url = "http://localhost:8000"
            plot_url = f"{base_url}/outputs/{plot_filename}"
            
            return {
                "url": plot_url,
                "filename": plot_filename,
                "created_at": metadata.get("created_at"),
                "size_bytes": metadata.get("size_bytes"),
                "exists": metadata.get("exists")
            }
        
        print("❌ No se pudo encontrar ningún archivo de gráfico")
        return {"error": "No se pudo encontrar el archivo del gráfico"}
    
    elif response_type == "table":
        sql_results = state.get("sql_results")
        if isinstance(sql_results, dict):
            return {
                "rows": sql_results.get("data", [])[:50],
                "columns": sql_results.get("columns", []),
                "total_rows": len(sql_results.get("data", []))
            }
        return None
    
    elif response_type == "error":
        return {
            "error_message": state.get("final_error", "Error desconocido"),
            "attempts": state.get("iteration_count", 0)
        }
    
    return None

def extract_sql_query(state: dict) -> str:
    """
    Extrae la consulta SQL si se ejecutó una.
    """
    for record in state.get("execution_history", []):
        if record.get("code") and "SELECT" in record["code"].upper():
            return record["code"]
    return None

# ============================================================================
# Endpoints de historial de conversaciones
# ============================================================================

@router.get("/chat-history", response_model=ChatHistoryResponse)
async def get_chat_history(
    limit: int = 30,
    thread_id: str = SINGLE_USER_THREAD_ID,
    include_incomplete: bool = False
):
    """
    Obtiene el historial de conversaciones del usuario.
    
    - **limit**: Número máximo de conversaciones a retornar (default: 20)
    - **thread_id**: ID del thread (default: thread único del sistema)
    - **include_incomplete**: Incluir conversaciones incompletas (default: False)
    
    Retorna las conversaciones ordenadas de más reciente a más antigua,
    incluyendo solo aquellas que tienen respuesta completa del LLM.
    """
    try:
        conversations = get_conversation_history(
            thread_id=thread_id,
            limit=limit,
            include_incomplete=include_incomplete
        )
        
        # Convertir a objetos ChatHistoryItem
        conversation_items = [
            ChatHistoryItem(**conv) for conv in conversations
        ]
        
        return ChatHistoryResponse(
            thread_id=thread_id,
            total=len(conversation_items),
            conversations=conversation_items
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Error obteniendo historial de conversaciones"
            }
        )