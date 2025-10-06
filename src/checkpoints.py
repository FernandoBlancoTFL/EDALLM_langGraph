import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from datetime import datetime
from typing import List
from database import load_db_config
from config import SINGLE_USER_THREAD_ID

# Variable global
postgres_saver = None

def setup_postgres_saver():
    """
    Configura e inicializa PostgresSaver para memoria de conversaciones.
    CORREGIDO: Usa autocommit para evitar problemas con índices concurrentes
    """
    global _postgres_saver
    print("🧠 Configurando PostgresSaver para memoria conversacional...")
    
    try:
        db_config = load_db_config()
        
        # Crear connection string para PostgresSaver
        postgres_uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        # Crear con autocommit para evitar error de índices concurrentes
        conn = psycopg.connect(postgres_uri, autocommit=True)
        checkpointer = PostgresSaver(conn)
        
        # Configurar las tablas automáticamente
        try:
            checkpointer.setup()
            print("✅ PostgresSaver configurado exitosamente")
            print("📊 Tablas de memoria creadas: checkpoints, checkpoint_blobs, checkpoint_writes")
            _postgres_saver = checkpointer
            return checkpointer
        except Exception as setup_error:
            print(f"⚠️ Error en setup: {setup_error}")
            # Fallback a método alternativo
            return setup_postgres_saver_alternative()
        
    except Exception as e:
        print(f"❌ Error configurando PostgresSaver: {e}")
        # Intentar método alternativo
        return setup_postgres_saver_alternative()

def setup_postgres_saver_alternative():
    """
    Configuración alternativa de PostgresSaver usando conexión con autocommit.
    """
    print("🔄 Intentando configuración alternativa de PostgresSaver...")
    
    try:
        db_config = load_db_config()
        postgres_uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        # SOLUCIÓN: Crear conexión con autocommit=True para evitar problemas con CREATE INDEX CONCURRENTLY
        conn = psycopg.connect(postgres_uri, autocommit=True)
        
        # Crear PostgresSaver con la conexión configurada
        checkpointer = PostgresSaver(conn)
        
        # Intentar setup (ahora debería funcionar con autocommit)
        try:
            checkpointer.setup()
            print("✅ PostgresSaver configurado con conexión en modo autocommit")
            return checkpointer
        except Exception as setup_error:
            print(f"⚠️ Error en setup automático: {setup_error}")
            # Si falla, intentar crear tablas manualmente SIN índices concurrentes
            create_checkpoint_tables_manually_no_concurrent(conn)
            print("✅ PostgresSaver configurado con tablas manuales (sin índices concurrentes)")
            return checkpointer
            
    except Exception as e:
        print(f"❌ Error en configuración alternativa: {e}")
        print("⚠️ Continuando sin memoria persistente")
        return None

def get_automatic_thread_id():
    """
    Retorna el thread ID fijo para el usuario único.
    Elimina la necesidad de configuración manual.
    """
    print(f"🔑 Usando thread persistente automático: {SINGLE_USER_THREAD_ID}")
    return SINGLE_USER_THREAD_ID

def list_user_conversations(postgres_saver_instance, user_id: str = None):
    """
    Lista las conversaciones previas del usuario.
    """
    if not postgres_saver_instance:
        print("⚠️ PostgresSaver no disponible")
        return []
    
    try:
        # Obtener checkpoints del usuario
        if user_id:
            # Buscar threads que contengan el user_id
            thread_pattern = f"user_{user_id}_persistent"
        else:
            # Listar todos los threads de sesión recientes
            thread_pattern = "session_%"
        
        print(f"📋 Buscando conversaciones para patrón: {thread_pattern}")
        # Nota: La implementación específica depende de la API interna de PostgresSaver
        # Aquí se podría implementar una consulta directa a la tabla checkpoints
        
        return []  # Placeholder - requiere acceso directo a la tabla checkpoints
        
    except Exception as e:
        print(f"❌ Error listando conversaciones: {e}")
        return []

def load_conversation_history(thread_id: str):
    """
    Recupera el historial de conversaciones desde PostgresSaver para un thread específico.
    CORREGIDO: Usa correctamente la API de PostgresSaver
    """
    postgres_saver = get_postgres_saver()
    
    if not postgres_saver:
        print("⚠️ PostgresSaver no disponible para recuperar historial")
        return [], {}
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        
        # CORRECCIÓN: Usar get_tuple() en lugar de get()
        checkpoint_tuple = postgres_saver.get_tuple(config)
        
        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            checkpoint = checkpoint_tuple.checkpoint
            
            # CORRECCIÓN: Los valores del estado están en channel_values
            channel_values = checkpoint.get("channel_values") or checkpoint.get("channel_versions", {})
            
            if not channel_values:
                print("📭 No se encontraron valores en el checkpoint")
                print(f"🔍 Claves disponibles: {list(checkpoint.keys())}")
                return [], {}
            
            # Recuperar conversation_history y user_context
            conversation_history = channel_values.get("conversation_history", [])
            user_context = channel_values.get("user_context", {
                "preferred_analysis_type": None,
                "common_datasets": [],
                "visualization_preferences": [],
                "error_patterns": [],
                "last_interaction": None
            })
            
            print(f"📚 Historial recuperado: {len(conversation_history)} conversaciones")
            print(f"👤 Contexto: {len(user_context.get('common_datasets', []))} datasets")
            
            return conversation_history, user_context
        else:
            print("📭 No se encontró checkpoint previo para este thread")
            return [], {}
            
    except Exception as e:
        print(f"⚠️ Error recuperando historial: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

def debug_checkpoint_structure(thread_id: str):
    """
    Función de debugging para inspeccionar la estructura del checkpoint.
    """
    # global postgres_saver

    postgres_saver = get_postgres_saver()
    
    if not postgres_saver:
        return
    
    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint = postgres_saver.get(config)
        
        print("🔍 DEBUGGING CHECKPOINT STRUCTURE:")
        print(f"   Tipo de checkpoint: {type(checkpoint)}")
        
        if checkpoint:
            if isinstance(checkpoint, dict):
                print(f"   Claves en checkpoint: {list(checkpoint.keys())}")
                for key, value in checkpoint.items():
                    print(f"   {key}: {type(value)} - {str(value)[:100]}...")
            else:
                print(f"   Checkpoint no es dict: {checkpoint}")
        else:
            print("   Checkpoint es None")
            
    except Exception as e:
        print(f"❌ Error en debugging: {e}")

def diagnose_postgres_saver():
    """
    Función de diagnóstico mejorada para verificar el estado de PostgresSaver
    """
    print("🔍 Diagnosticando configuración de PostgresSaver...")
    
    try:
        # 1. Verificar instalación de módulos
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            print("✅ Módulo PostgresSaver importado correctamente")
        except ImportError as e:
            print(f"❌ Error importando PostgresSaver: {e}")
            return False
        
        # 2. Verificar conexión a BD
        db_config = load_db_config()
        postgres_uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        with psycopg.connect(postgres_uri) as conn:
            print("✅ Conexión PostgreSQL exitosa")
            
            # 3. Verificar tablas de checkpoint
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE '%checkpoint%'
                    ORDER BY table_name
                """)
                
                checkpoint_tables = [t[0] for t in cursor.fetchall()]
                expected_tables = ['checkpoints', 'checkpoint_blobs', 'checkpoint_writes', 'checkpoint_migrations']
                
                print(f"📋 Tablas encontradas: {checkpoint_tables}")
                missing_tables = [t for t in expected_tables if t not in checkpoint_tables]
                
                if missing_tables:
                    print(f"⚠️ Tablas faltantes: {missing_tables}")
                    return False
                else:
                    print("✅ Todas las tablas de checkpoint están presentes")
        
        # 4. Intentar crear PostgresSaver
        try:
            checkpointer = PostgresSaver.from_conn_string(postgres_uri)
            print("✅ PostgresSaver creado exitosamente")
            
            # 5. Verificar que tiene los métodos necesarios
            required_methods = ['get_next_version', 'setup', 'get', 'put']
            for method in required_methods:
                if hasattr(checkpointer, method):
                    print(f"✅ Método {method} disponible")
                else:
                    print(f"❌ Método {method} no encontrado")
                    return False
            
            return True
            
        except Exception as ps_error:
            print(f"❌ Error creando PostgresSaver: {ps_error}")
            return False
            
    except Exception as e:
        print(f"❌ Error en diagnóstico: {e}")
        return False

def create_checkpoint_tables_manually(conn):
    """
    Crea las tablas de checkpoint manualmente si el setup automático falla.
    """
    print("🛠️ Creando tablas de checkpoint manualmente...")
    
    try:
        with conn.cursor() as cursor:
            # Tabla principal de checkpoints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    checkpoint_blob BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (thread_id, checkpoint_id)
                )
            """)
            
            # Tabla para blobs grandes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    blob BYTEA,
                    PRIMARY KEY (thread_id, checkpoint_id, channel),
                    FOREIGN KEY (thread_id, checkpoint_id) REFERENCES checkpoints(thread_id, checkpoint_id)
                )
            """)
            
            # Tabla para escrituras
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    blob BYTEA,
                    PRIMARY KEY (thread_id, checkpoint_id, task_id, idx),
                    FOREIGN KEY (thread_id, checkpoint_id) REFERENCES checkpoints(thread_id, checkpoint_id)
                )
            """)
            
            # Tabla de migraciones (si no existe)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_migrations (
                    version INTEGER PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insertar versión de migración si no existe
            cursor.execute("""
                INSERT INTO checkpoint_migrations (version) 
                VALUES (1) 
                ON CONFLICT (version) DO NOTHING
            """)
            
            conn.commit()
            print("✅ Tablas de checkpoint creadas manualmente")
            
    except Exception as e:
        print(f"❌ Error creando tablas manualmente: {e}")
        conn.rollback()

def create_checkpoint_tables_manually_no_concurrent(conn):
    """
    Crea las tablas de checkpoint manualmente SIN índices concurrentes.
    Versión compatible con el error de transacción.
    """
    print("🛠️ Creando tablas de checkpoint sin índices concurrentes...")
    
    try:
        with conn.cursor() as cursor:
            # Tabla principal de checkpoints
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    parent_checkpoint_id TEXT,
                    type TEXT,
                    checkpoint JSONB NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
                )
            """)
            
            # Crear índice normal (no concurrente)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS checkpoints_thread_id_idx 
                ON checkpoints(thread_id, checkpoint_ns)
            """)
            
            # Tabla para blobs grandes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_blobs (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    channel TEXT NOT NULL,
                    version TEXT NOT NULL,
                    type TEXT NOT NULL,
                    blob BYTEA,
                    PRIMARY KEY (thread_id, checkpoint_ns, channel, version)
                )
            """)
            
            # Tabla para escrituras
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_writes (
                    thread_id TEXT NOT NULL,
                    checkpoint_ns TEXT NOT NULL DEFAULT '',
                    checkpoint_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    idx INTEGER NOT NULL,
                    channel TEXT NOT NULL,
                    type TEXT,
                    blob BYTEA NOT NULL,
                    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
                )
            """)
            
            # Crear índice normal para writes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS checkpoint_writes_thread_id_idx 
                ON checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id)
            """)
            
            # Tabla de migraciones
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_migrations (
                    v INTEGER PRIMARY KEY,
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insertar versión de migración
            cursor.execute("""
                INSERT INTO checkpoint_migrations (v) 
                VALUES (1) 
                ON CONFLICT (v) DO NOTHING
            """)
            
            print("✅ Tablas de checkpoint creadas sin índices concurrentes")
            
    except Exception as e:
        print(f"❌ Error creando tablas sin índices concurrentes: {e}")
        raise

def get_postgres_saver():
    """Retorna la instancia actual de PostgresSaver"""
    return _postgres_saver