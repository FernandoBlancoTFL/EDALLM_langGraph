from pydantic import BaseModel
from typing import Optional, Any, List

class ChatRequest(BaseModel):
    """Esquema para la petición del endpoint /chat"""
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": " ¿Cuántas filas tiene el dataset de cocodrilos?"
            }
        }

class ChatResponse(BaseModel):
    """Esquema para la respuesta del endpoint /chat"""
    response: str
    type: str  # "text", "plot", "table", "error"
    data: Optional[Any] = None
    sql_query: Optional[str] = None
    success: bool
    iterations: int
    strategy_used: str  # "sql", "dataframe"
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "He generado un histograma de la columna edad",
                "type": "plot",
                "data": {
                    "url": "http://localhost:8000/outputs/histogram_edad_20231008_143022.png",
                    "filename": "histogram_edad_20231008_143022.png",
                    "created_at": "2023-10-08T14:30:22",
                    "size_bytes": 125840,
                    "exists": True
                },
                "sql_query": None,
                "success": True,
                "iterations": 1,
                "strategy_used": "dataframe"
            }
        }

# ============================================================================
# Esquemas para historial de conversaciones
# ============================================================================

class ChatHistoryItem(BaseModel):
    """Representa una conversación individual en el historial"""
    checkpoint_id: str
    timestamp: str
    query: str
    llm_response: Optional[str] = None
    success: bool
    response_metadata: Optional[dict] = None

    class Config:
        json_schema_extra = {
            "example": {
                "checkpoint_id": "1f0aa458-40ec-6df3-8004-10848f5d6422",
                "timestamp": "2025-10-16T04:06:39.351300+00:00",
                "query": "genera un histograma de la columna edad",
                "llm_response": "Se generó un histograma que muestra...",
                "success": True,
                "response_metadata": {
                    "type": "plot",
                    "data": {
                        "url": "http://localhost:8000/outputs/histogram_edad.png",
                        "filename": "histogram_edad.png",
                        "exists": True
                    }
                }
            }
        }


class ChatHistoryResponse(BaseModel):
    """Respuesta del endpoint de historial"""
    thread_id: str
    total: int
    conversations: List[ChatHistoryItem]
    
    class Config:
        json_schema_extra = {
            "example": {
                "thread_id": "single_user_persistent_thread",
                "total": 5,
                "conversations": [
                    {
                        "checkpoint_id": "1f0aa458-40ec-6df3-8004-10848f5d6422",
                        "timestamp": "2025-10-16T04:06:39.351300+00:00",
                        "query": "genera un histograma",
                        "llm_response": "Se generó un histograma...",
                        "success": True,
                        "response_metadata": {"type": "plot"}
                    }
                ]
            }
        }