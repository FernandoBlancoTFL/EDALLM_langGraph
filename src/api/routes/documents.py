from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from api.schemas.documents import (
    DocumentUploadResponse, 
    DocumentInfo, 
    DocumentDeleteResponse
)
from services.document_service import DocumentService

router = APIRouter()

# Inicializar servicio
document_service = DocumentService()

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    try:
        # Validar que se subió un archivo
        if not file.filename:
            raise HTTPException(status_code=400, detail="No se proporcionó ningún archivo")
        
        # Leer contenido del archivo
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="El archivo está vacío")
        
        print(f"📥 Archivo recibido: {file.filename} ({len(content)} bytes)")
        
        # Procesar y cargar documento
        result = await document_service.upload_document(content, file.filename)
        
        return DocumentUploadResponse(
            message="Archivo cargado correctamente",
            file_id=result["file_id"],
            filename=result["filename"],
            table_name=result["table_name"],
            rows_imported=result["rows_imported"],
            is_duplicate=False
        )
        
    except ValueError as e:
        # Esto capturará tanto errores de validación como duplicados
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"❌ Error al subir documento: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar el archivo: {str(e)}"
        )

@router.get("/", response_model=List[DocumentInfo])
async def list_documents():
    """
    Lista todos los documentos almacenados en la base de datos.
    
    - Retorna información de cada tabla/dataset
    - Incluye cantidad de filas y columnas
    - Fecha de creación
    """
    try:
        documents = document_service.list_documents()
        return documents
    except Exception as e:
        print(f"❌ Error al listar documentos: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al listar documentos: {str(e)}"
        )

@router.delete("/{file_id}", response_model=DocumentDeleteResponse)
async def delete_document(file_id: str):
    """
    Elimina un documento de la base de datos.
    
    - Busca la tabla por file_id
    - Elimina la tabla de PostgreSQL
    - Elimina el archivo físico (si existe)
    """
    try:
        result = document_service.delete_document(file_id)
        
        return DocumentDeleteResponse(
            message="Documento eliminado correctamente",
            file_id=result["file_id"],
            table_name=result["table_name"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"❌ Error al eliminar documento: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al eliminar documento: {str(e)}"
        )