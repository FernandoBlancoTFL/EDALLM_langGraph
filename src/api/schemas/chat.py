from pydantic import BaseModel
from typing import Optional, Any

class ChatRequest(BaseModel):
    """Esquema para la petición del endpoint /chat"""
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Muéstrame un gráfico con las ventas por mes"
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