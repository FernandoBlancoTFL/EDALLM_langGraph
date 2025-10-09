from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from api.routes import chat
from database import create_database_if_not_exists, test_target_database_connection, setup_data_connection
from dataset_manager import initialize_dataset_on_startup
from checkpoints import setup_postgres_saver

# Crear aplicación FastAPI
app = FastAPI(
    title="LLM Data Analysis API",
    description="API para análisis de datos con LLM y memoria conversacional",
    version="1.0.0"
)

app.mount("/outputs", StaticFiles(directory="src/outputs"), name="outputs")

# Configurar CORS para permitir requests desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

@app.on_event("startup")
async def startup_event():
    """
    Inicialización al arrancar el servidor.
    Configura BD, datasets y memoria.
    """
    print("🚀 Inicializando sistema de análisis de datos...")
    
    # Crear/verificar base de datos
    if not create_database_if_not_exists():
        print("❌ Error: No se pudo crear o acceder a la base de datos")
        return
    
    # Probar conexión
    if not test_target_database_connection():
        print("❌ Error: No se pudo conectar a la base de datos")
        return
    
    # Configurar conexión de datos
    setup_data_connection()
    
    # Inicializar datasets
    print("🔄 Inicializando sistema de dataset...")
    if not initialize_dataset_on_startup():
        print("⚠️ Advertencia: Error en inicialización del dataset")
    
    # Configurar PostgresSaver para memoria
    postgres_saver = setup_postgres_saver()
    
    print("✅ Sistema inicializado correctamente")
    print(f"🧠 Memoria conversacional: {'ACTIVADA' if postgres_saver else 'DESACTIVADA'}")
    print("📡 API lista para recibir requests en /api/chat")

@app.get("/")
async def root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {
        "message": "LLM Data Analysis API",
        "status": "online",
        "endpoints": {
            "chat": "/api/chat",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {"status": "healthy"}