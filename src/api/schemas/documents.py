from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class DocumentUploadResponse(BaseModel):
    """Respuesta al subir un documento"""
    message: str
    file_id: str
    filename: str
    table_name: str
    rows_imported: int
    is_duplicate: bool = False
    duplicate_of: Optional[str] = None
    upload_date: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Archivo cargado correctamente",
                "file_id": "dfd83a2a",
                "filename": "ventas_2024.xlsx",
                "table_name": "ventas_2024_dfd83a2a",
                "rows_imported": 1500,
                "is_duplicate": False,
                "duplicate_of": None,
                "upload_date": None
            }
        }

class DocumentInfo(BaseModel):
    """Información de un documento almacenado"""
    file_id: str
    filename: str
    table_name: str
    row_count: int
    column_count: int
    created_at: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "file_id": "dfd83a2a",
                "filename": "ventas_2024.xlsx",
                "table_name": "ventas_2024",
                "row_count": 1500,
                "column_count": 12,
                "created_at": "2025-10-08T12:30:00"
            }
        }

class DocumentDeleteResponse(BaseModel):
    """Respuesta al eliminar un documento"""
    message: str
    file_id: str
    table_name: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Documento eliminado correctamente",
                "file_id": "dfd83a2a",
                "table_name": "ventas_2024"
            }
        }