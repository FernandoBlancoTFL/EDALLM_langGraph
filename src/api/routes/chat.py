from fastapi import APIRouter, HTTPException
from datetime import datetime
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

        # DEBUG: Imprimir información relevante sobre el archivo
        # print("\n🔍 DEBUG - Buscando archivo de gráfico:")
        # print(f"   Result: {final_state.get('result', 'N/A')[:200]}")
        # print(f"   LLM Response: {final_state.get('llm_response', 'N/A')[:200]}")
        # if final_state.get("execution_history"):
        #     for i, record in enumerate(final_state["execution_history"]):
        #         if record.get("success"):
        #             print(f"   Execution {i} result: {str(record.get('result', 'N/A'))[:200]}")
        # print()
        
        # Limpiar estado para serialización
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
    Tipos: "text", "plot", "table", "error"
    """
    if not state.get("success", False):
        return "error"
    
    # Verificar si se generó un gráfico
    if state.get("execution_history"):
        for record in state["execution_history"]:
            if record.get("success") and record.get("code"):
                code = record["code"].lower()
                if any(keyword in code for keyword in ["plt.", "plot", "hist", "scatter", "savefig"]):
                    return "plot"
    
    # Verificar si hay datos tabulares (SQL o DataFrame)
    if state.get("sql_results"):
        sql_results = state["sql_results"]
        if isinstance(sql_results, dict) and sql_results.get("data"):
            return "table"
    
    # Por defecto, respuesta de texto
    return "text"

def extract_response_data(state: dict, response_type: str):
    """
    Extrae datos adicionales según el tipo de respuesta.
    MODIFICADO: Mejor manejo de búsqueda de archivos de gráficos.
    """
    if response_type == "plot":
        plot_filename = None
        
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
        
        # Método 4: Buscar el archivo más reciente en outputs/
        if not plot_filename:
            print("⚠️ No se encontró nombre de archivo en resultados, buscando archivo más reciente...")
            try:
                outputs_dir = "./src/outputs"
                if os.path.exists(outputs_dir):
                    files = [f for f in os.listdir(outputs_dir) if f.endswith('.png')]
                    if files:
                        # Ordenar por fecha de modificación (más reciente primero)
                        files.sort(key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)), reverse=True)
                        plot_filename = files[0]
                        print(f"🔍 Usando archivo más reciente: {plot_filename}")
            except Exception as e:
                print(f"❌ Error buscando archivos: {e}")
        
        if plot_filename:
            # Obtener metadata del archivo
            metadata = get_plot_metadata(plot_filename)
            
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
        # ... (mantener el código existente para table)
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