import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools import PythonREPLTool  # intérprete de Python de Camel AI
from typing import TypedDict, List, Any, Optional, Optional
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any
import sys

import psycopg
from psycopg import sql
from langgraph.checkpoint.postgres import PostgresSaver
import uuid
import json

def clean_data_for_json(data):
    """Limpia datos para que sean serializables en JSON"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif pd.isna(data):
        return "NULL"
    elif hasattr(data, 'isoformat'):  # Timestamps
        return data.isoformat()
    else:
        return data

# Configuración del límite de conversaciones
MAX_CONVERSATIONS_PER_THREAD = 10  # Límite recomendado

def cleanup_old_conversations(thread_id: str, max_conversations: int = MAX_CONVERSATIONS_PER_THREAD):
    """Limpia conversaciones antiguas, manteniendo solo las últimas max_conversations"""
    global history_connection
    
    if history_connection is None:
        print("⚠️ No se puede limpiar: history_connection es None")
        return False
    
    try:
        with history_connection.cursor() as cursor:
            # Obtener conversaciones actuales
            cursor.execute(
                "SELECT conversation_data FROM conversation_memory WHERE thread_id = %s",
                (thread_id,)
            )
            result = cursor.fetchone()
            
            if not result:
                return False
            
            conversation_data = result[0]
            if not isinstance(conversation_data, dict):
                return False
            
            conversation_history = conversation_data.get("conversation_history", [])
            
            # Verificar si necesita limpieza
            if len(conversation_history) <= max_conversations:
                print(f"🔍 Historial tiene {len(conversation_history)} conversaciones, no necesita limpieza")
                return False
            
            # Conservar solo las últimas max_conversations
            cleaned_history = conversation_history[-max_conversations:]
            conversations_removed = len(conversation_history) - len(cleaned_history)
            
            # Actualizar datos
            conversation_data["conversation_history"] = cleaned_history
            conversation_data["total_queries"] = len(cleaned_history)
            
            # Guardar datos actualizados
            cursor.execute("""
                UPDATE conversation_memory 
                SET conversation_data = %s, 
                    total_queries = %s,
                    last_updated = CURRENT_TIMESTAMP
                WHERE thread_id = %s
            """, (
                json.dumps(conversation_data, default=str),
                len(cleaned_history),
                thread_id
            ))
            
            history_connection.commit()
            
            print(f"🧹 Limpieza completada: eliminadas {conversations_removed} conversaciones antiguas")
            print(f"   Conservadas: {len(cleaned_history)} conversaciones más recientes")
            
            return True
            
    except Exception as e:
        print(f"⚠️ Error en limpieza de conversaciones: {e}")
        return False

# Suprimir warnings de Google Cloud
import warnings
warnings.filterwarnings('ignore', message='.*ALTS.*')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# ======================
# Thread ID persistente para memoria a corto plazo
# ======================
PERSISTENT_THREAD_ID = "persistent_chat_session"

# Configuración del archivo Excel y tabla PostgreSQL
DATASET_CONFIG = {
    "excel_path": "./Data/ncr_ride_bookings.xlsx",
    "table_name": "dataset_rides",
    "table_schema": "public"
}

# ======================
# 0. Configuración y creación de BD PostgreSQL
# ======================

def load_db_config():
    """Carga la configuración de la base de datos desde variables de entorno"""
    load_dotenv()
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'database': os.getenv('POSTGRES_DB', 'langgraph_analysis')
    }
    
    # Verificar que las credenciales estén configuradas
    if not db_config['password']:
        print("❌ Error: POSTGRES_PASSWORD no está configurada en el archivo .env")
        sys.exit(1)
    
    return db_config

# Variable global para conexión de historial independiente
history_connection = None

def database_exists(cursor, db_name):
    """Verifica si una base de datos existe"""
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", 
        (db_name,)
    )
    return cursor.fetchone() is not None

def create_database_if_not_exists():
    """
    Crea la base de datos si no existe.
    Retorna True si la BD se creó o ya existía, False en caso de error.
    """
    db_config = load_db_config()
    target_db = db_config['database']
    
    print(f"🔍 Verificando existencia de base de datos: {target_db}")
    
    # Conectar a PostgreSQL usando la BD por defecto 'postgres'
    try:
        with psycopg.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            dbname='postgres',  # Conectar a la BD por defecto
            autocommit=True
        ) as conn:
            with conn.cursor() as cursor:
                # Verificar si la base de datos existe
                if database_exists(cursor, target_db):
                    print(f"✅ Base de datos '{target_db}' ya existe")
                    return True
                else:
                    print(f"📝 Base de datos '{target_db}' no existe. Creando...")
                    
                    # Crear la base de datos
                    cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(target_db)
                    ))
                    
                    print(f"✅ Base de datos '{target_db}' creada exitosamente")
                    return True
            
    except psycopg.Error as e:
        print(f"❌ Error al gestionar la base de datos PostgreSQL:")
        print(f"   Código de error: {e.pgcode}")
        print(f"   Mensaje: {e.pgerror}")
        
        # Errores comunes y sugerencias
        if "authentication failed" in str(e).lower():
            print("   💡 Sugerencia: Verifica las credenciales en el archivo .env")
        elif "connection refused" in str(e).lower():
            print("   💡 Sugerencia: Verifica que PostgreSQL esté ejecutándose")
        elif "permission denied" in str(e).lower():
            print("   💡 Sugerencia: El usuario necesita permisos CREATEDB")
        
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_target_database_connection():
    """Prueba la conexión a la base de datos objetivo"""
    db_config = load_db_config()
    
    try:
        with psycopg.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            dbname=db_config['database']
        ) as conn:
            pass  # Conexión exitosa
        print(f"✅ Conexión exitosa a la base de datos '{db_config['database']}'")
        return True
    except psycopg.Error as e:
        print(f"❌ Error al conectar a la base de datos objetivo: {e}")
        return False
    
def setup_history_connection():
    """
    Configura una conexión independiente solo para el sistema de historial.
    Esta función no depende de PostgresSaver.
    """
    global history_connection
    
    print("🔧 Configurando conexión independiente para historial...")
    
    try:
        db_config = load_db_config()
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        history_connection = psycopg.connect(connection_string)
        
        # Crear tabla inmediatamente
        with history_connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    thread_id VARCHAR(255) PRIMARY KEY,
                    conversation_data JSONB,
                    total_queries INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            history_connection.commit()
        
        print("✅ Conexión de historial configurada exitosamente")
        print("✅ Tabla conversation_memory creada/verificada")
        return True
        
    except Exception as e:
        print(f"❌ Error configurando conexión de historial: {e}")
        history_connection = None
        return False

# ======================
# Funciones de gestión de historial persistente
# ======================

def check_thread_exists():
    """Verifica si el thread tiene historial en la tabla personalizada"""
    global history_connection
    
    if history_connection is None:
        print("🔍 Debug - history_connection es None")
        return False
    
    try:
        with history_connection.cursor() as cursor:
            cursor.execute(
                "SELECT total_queries FROM conversation_memory WHERE thread_id = %s",
                (PERSISTENT_THREAD_ID,)
            )
            result = cursor.fetchone()
            
            if result:
                total_queries = result[0]
                print(f"🔍 Debug - Conversación encontrada con {total_queries} consultas")
                return True
            else:
                print(f"🔍 Debug - No hay historial para thread {PERSISTENT_THREAD_ID}")
                return False
                
    except Exception as e:
        print(f"⚠️ Error verificando historial: {e}")
        return False

def load_previous_context():
    """Carga el historial desde la tabla personalizada"""
    global history_connection
    
    if history_connection is None:
        print("🔍 Debug - history_connection es None en load_previous_context")
        return None
    
    try:
        with history_connection.cursor() as cursor:
            cursor.execute(
                "SELECT conversation_data FROM conversation_memory WHERE thread_id = %s",
                (PERSISTENT_THREAD_ID,)
            )
            result = cursor.fetchone()
            
            if result:
                conversation_data = result[0]  # JSONB se deserializa automáticamente
                print("🔍 Debug - Contexto cargado desde tabla personalizada")
                return conversation_data
            else:
                print("🔍 Debug - No hay contexto guardado")
                return None
                
    except Exception as e:
        print(f"⚠️ Error cargando contexto: {e}")
        return None

def get_conversation_summary():
    """Obtener resumen del historial con información del límite"""
    previous_context = load_previous_context()
    if not previous_context:
        return "No hay historial previo."
    
    if not isinstance(previous_context, dict):
        return "No hay historial previo válido."
    
    conversation_history = previous_context.get("conversation_history", [])
    if not conversation_history:
        return "No hay consultas previas en el historial."
    
    total_queries = len(conversation_history)
    last_query = conversation_history[-1].get("query", "N/A") if conversation_history else "N/A"
    
    # Información sobre el límite
    limit_info = f" (límite: {MAX_CONVERSATIONS_PER_THREAD})"
    
    return f"Historial: {total_queries}/{MAX_CONVERSATIONS_PER_THREAD} conversaciones{limit_info}. Última: '{last_query[:50]}...'"

def maintenance_cleanup_all_threads():
    """
    Función de mantenimiento para limpiar todas las conversaciones que excedan el límite.
    Útil para ejecutar periódicamente o cuando sea necesario.
    """
    if checkpoint_saver is None:
        print("⚠️ No se puede realizar mantenimiento: checkpoint_saver es None")
        return False
    
    try:
        with checkpoint_saver.conn.cursor() as cursor:
            # Verificar si la tabla existe
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'conversation_memory'
                )
            """)
            
            if not cursor.fetchone()[0]:
                print("ℹ️ No hay tabla de conversaciones para limpiar")
                return True
            
            # Obtener todos los threads
            cursor.execute("SELECT thread_id FROM conversation_memory")
            threads = cursor.fetchall()
            
            cleaned_threads = 0
            total_threads = len(threads)
            
            print(f"🧹 Iniciando mantenimiento de {total_threads} threads...")
            
            for (thread_id,) in threads:
                if cleanup_old_conversations(thread_id, MAX_CONVERSATIONS_PER_THREAD):
                    cleaned_threads += 1
            
            print(f"✅ Mantenimiento completado:")
            print(f"   Threads procesados: {total_threads}")
            print(f"   Threads limpiados: {cleaned_threads}")
            print(f"   Threads sin cambios: {total_threads - cleaned_threads}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error en mantenimiento general: {e}")
        return False

def save_conversation_state(state, config):
    """Guarda el historial de conversación usando la conexión independiente"""
    global history_connection
    
    if history_connection is None:
        print("⚠️ No se puede guardar: history_connection es None")
        return False
    
    try:
        conversation_history = state.get("conversation_history", [])
        if not conversation_history:
            print("⚠️ No hay historial de conversación para guardar")
            return False
        
        thread_id = config["configurable"]["thread_id"]
        
        # Aplicar límite de conversaciones antes de guardar
        if len(conversation_history) > MAX_CONVERSATIONS_PER_THREAD:
            conversation_history = conversation_history[-MAX_CONVERSATIONS_PER_THREAD:]
            conversations_removed = len(state.get("conversation_history", [])) - len(conversation_history)
            
            print(f"🧹 Aplicando límite: eliminadas {conversations_removed} conversaciones antiguas")
            print(f"   Conservando las últimas {len(conversation_history)} conversaciones")
            
            # Actualizar el estado con el historial limitado
            state["conversation_history"] = conversation_history
            state["total_queries"] = len(conversation_history)
        
        with history_connection.cursor() as cursor:
            # Preparar datos para guardar
            conversation_data = {
                "conversation_history": conversation_history,
                "session_start_time": state.get("session_start_time", pd.Timestamp.now().isoformat()),
                "total_queries": len(conversation_history),
                "df_info": clean_data_for_json(state.get("df_info", {})),
                "max_conversations_limit": MAX_CONVERSATIONS_PER_THREAD
            }
            
            # Insertar o actualizar
            cursor.execute("""
                INSERT INTO conversation_memory (thread_id, conversation_data, total_queries)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_id) 
                DO UPDATE SET 
                    conversation_data = EXCLUDED.conversation_data,
                    total_queries = EXCLUDED.total_queries,
                    last_updated = CURRENT_TIMESTAMP
            """, (
                thread_id,
                json.dumps(conversation_data, default=str),
                len(conversation_history)
            ))
            
            history_connection.commit()
            
        print(f"💾 Historial guardado - {len(conversation_history)} conversaciones activas")
        return True
        
    except Exception as e:
        print(f"⚠️ Error guardando en BD personalizada: {e}")
        return False
    
def verify_checkpoint_saved(config):
    """Verifica que el checkpoint se guardó correctamente"""
    if checkpoint_saver is None:
        return False
    
    try:
        checkpoints_list = list(checkpoint_saver.list(config))
        count = len(checkpoints_list)
        print(f"🔍 Verificación: {count} checkpoints en BD")
        
        if count > 0:
            latest = checkpoints_list[0]
            if hasattr(latest, 'checkpoint') and latest.checkpoint:
                channel_values = latest.checkpoint.get("channel_values", {})
                conversations = channel_values.get("conversation_history", [])
                print(f"🔍 Conversaciones guardadas: {len(conversations)}")
                return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error verificando checkpoint: {e}")
        return False
    
# ======================
# FUNCIONES PARA GESTIÓN DEL DATASET EN BD - Agregar después de las funciones de historial
# ======================

def get_postgres_data_types():
    """
    Mapeo de tipos de pandas a tipos PostgreSQL optimizados.
    """
    return {
        'int64': 'BIGINT',
        'int32': 'INTEGER',
        'int16': 'SMALLINT',
        'int8': 'SMALLINT',
        'float64': 'DOUBLE PRECISION',
        'float32': 'REAL',
        'object': 'TEXT',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'timedelta64[ns]': 'INTERVAL',
        'category': 'TEXT'
    }

def sanitize_column_name(column_name: str) -> str:
    """
    Limpia nombres de columnas para PostgreSQL.
    Convierte a minúsculas, reemplaza espacios y caracteres especiales.
    """
    import re
    # Convertir a minúsculas
    clean_name = column_name.lower()
    # Reemplazar espacios y caracteres especiales con guiones bajos
    clean_name = re.sub(r'[^a-z0-9_]', '_', clean_name)
    # Eliminar guiones bajos consecutivos
    clean_name = re.sub(r'_+', '_', clean_name)
    # Eliminar guiones bajos al inicio y final
    clean_name = clean_name.strip('_')
    # Asegurar que no empiece con número
    if clean_name and clean_name[0].isdigit():
        clean_name = f'col_{clean_name}'
    
    return clean_name or 'unnamed_column'

def check_dataset_table_exists(connection=None):
    """
    Verifica si la tabla del dataset existe en PostgreSQL.
    Acepta una conexión opcional o usa checkpoint_saver si está disponible.
    """
    conn = connection or (checkpoint_saver.conn if checkpoint_saver else None)
    
    if conn is None:
        print("⚠️ No se puede verificar tabla: no hay conexión disponible")
        return False
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_name = %s
                )
            """, (DATASET_CONFIG["table_schema"], DATASET_CONFIG["table_name"]))
            
            exists = cursor.fetchone()[0]
            print(f"🔍 Tabla '{DATASET_CONFIG['table_name']}' {'existe' if exists else 'no existe'} en BD")
            return exists
            
    except Exception as e:
        print(f"⚠️ Error verificando tabla del dataset: {e}")
        return False

def get_dataset_table_info(connection=None):
    """
    Obtiene información de la tabla del dataset.
    Acepta una conexión opcional o usa checkpoint_saver si está disponible.
    """
    conn = connection or (checkpoint_saver.conn if checkpoint_saver else None)
    
    if conn is None:
        return None
    
    try:
        with conn.cursor() as cursor:
            # Obtener información de columnas
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s
                ORDER BY ordinal_position
            """, (DATASET_CONFIG["table_schema"], DATASET_CONFIG["table_name"]))
            
            columns_info = cursor.fetchall()
            
            # Obtener conteo de filas
            cursor.execute(f"""
                SELECT COUNT(*) FROM {DATASET_CONFIG["table_schema"]}.{DATASET_CONFIG["table_name"]}
            """)
            
            row_count = cursor.fetchone()[0]
            
            # Formatear información
            columns = [col[0] for col in columns_info]
            dtypes = {col[0]: col[1] for col in columns_info}
            
            return {
                "columns": columns,
                "dtypes": dtypes,
                "row_count": row_count,
                "table_name": DATASET_CONFIG["table_name"]
            }
            
    except Exception as e:
        print(f"⚠️ Error obteniendo información de tabla: {e}")
        return None

def create_dataset_table_from_df(df: pd.DataFrame, connection=None):
    """
    Crea la tabla del dataset en PostgreSQL.
    Acepta una conexión opcional o usa checkpoint_saver si está disponible.
    """
    conn = connection or (checkpoint_saver.conn if checkpoint_saver else None)
    
    if conn is None:
        print("⚠️ No se puede crear tabla: no hay conexión disponible")
        return False, {}
    
    try:
        postgres_types = get_postgres_data_types()
        
        # Limpiar nombres de columnas y crear mapeo
        original_columns = list(df.columns)
        clean_columns = [sanitize_column_name(col) for col in original_columns]
        column_mapping = dict(zip(original_columns, clean_columns))
        
        print(f"📝 Creando tabla '{DATASET_CONFIG['table_name']}' con {len(df.columns)} columnas...")
        
        with conn.cursor() as cursor:
            # Construir DDL para crear tabla
            column_definitions = []
            
            for original_col, clean_col in column_mapping.items():
                # Obtener tipo de pandas
                pandas_type = str(df[original_col].dtype)
                
                # Mapear a tipo PostgreSQL
                postgres_type = postgres_types.get(pandas_type, 'TEXT')
                
                # Manejar casos especiales
                if pandas_type == 'object':
                    # Para object, verificar si es fecha o texto
                    try:
                        pd.to_datetime(df[original_col], errors='raise')
                        postgres_type = 'TIMESTAMP'
                    except:
                        # Estimar longitud máxima para TEXT
                        max_length = df[original_col].astype(str).str.len().max()
                        if max_length and max_length < 255:
                            postgres_type = f'VARCHAR({max_length + 50})'
                        else:
                            postgres_type = 'TEXT'
                
                column_definitions.append(f'"{clean_col}" {postgres_type}')
                print(f"   {original_col} -> {clean_col} ({pandas_type} -> {postgres_type})")
            
            # Crear tabla
            create_table_sql = f"""
                CREATE TABLE {DATASET_CONFIG["table_schema"]}.{DATASET_CONFIG["table_name"]} (
                    id SERIAL PRIMARY KEY,
                    {', '.join(column_definitions)},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            
            print(f"✅ Tabla '{DATASET_CONFIG['table_name']}' creada exitosamente")
            return True, column_mapping
            
    except Exception as e:
        print(f"❌ Error creando tabla del dataset: {e}")
        if conn:
            conn.rollback()
        return False, {}

def insert_dataframe_to_table(df: pd.DataFrame, column_mapping: dict, connection=None):
    """
    Inserta los datos del DataFrame en la tabla PostgreSQL.
    Acepta una conexión opcional o usa checkpoint_saver si está disponible.
    """
    conn = connection or (checkpoint_saver.conn if checkpoint_saver else None)
    
    if conn is None:
        print("⚠️ No se puede insertar datos: no hay conexión disponible")
        return False
    
    try:
        # Renombrar columnas según el mapeo
        df_clean = df.rename(columns=column_mapping)
        
        print(f"📥 Insertando {len(df_clean)} filas en la tabla...")
        
        # Preparar datos para inserción
        columns_list = list(column_mapping.values())
        placeholders = ', '.join(['%s'] * len(columns_list))
        columns_str = ', '.join([f'"{col}"' for col in columns_list])
        
        insert_sql = f"""
            INSERT INTO {DATASET_CONFIG["table_schema"]}.{DATASET_CONFIG["table_name"]} 
            ({columns_str}) VALUES ({placeholders})
        """
        
        # Convertir DataFrame a lista de tuplas
        data_rows = []
        for _, row in df_clean.iterrows():
            row_data = []
            for col in columns_list:
                value = row[col]
                if pd.isna(value):
                    row_data.append(None)
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    row_data.append(value.to_pydatetime() if hasattr(value, 'to_pydatetime') else str(value))
                else:
                    row_data.append(value)
            data_rows.append(tuple(row_data))
        
        # Inserción por lotes
        with conn.cursor() as cursor:
            cursor.executemany(insert_sql, data_rows)
            conn.commit()
            
            # Verificar inserción
            cursor.execute(f"SELECT COUNT(*) FROM {DATASET_CONFIG['table_schema']}.{DATASET_CONFIG['table_name']}")
            inserted_count = cursor.fetchone()[0] - 1  # Restar 1 por el ID serial
            
            print(f"✅ {inserted_count} filas insertadas correctamente")
            return True
            
    except Exception as e:
        print(f"❌ Error insertando datos: {e}")
        if conn:
            conn.rollback()
        return False

def load_excel_to_postgres(connection=None):
    """
    Función principal que carga el archivo Excel a PostgreSQL si no existe.
    Acepta una conexión opcional para resolver el problema de dependencias.
    """
    # Verificar si ya existe la tabla
    if check_dataset_table_exists(connection):
        print("✅ Dataset ya está cargado en PostgreSQL")
        table_info = get_dataset_table_info(connection)
        if table_info:
            return table_info, True  # True indica que se cargó desde BD
        else:
            print("⚠️ Error obteniendo info de tabla existente")
    
    # Si no existe, cargar Excel y crear tabla
    print(f"📂 Cargando archivo Excel: {DATASET_CONFIG['excel_path']}")
    
    try:
        # Verificar que el archivo existe
        if not os.path.exists(DATASET_CONFIG['excel_path']):
            print(f"❌ Archivo Excel no encontrado: {DATASET_CONFIG['excel_path']}")
            return None, False
        
        # Cargar DataFrame
        df = pd.read_excel(DATASET_CONFIG['excel_path'])
        print(f"📊 Excel cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Crear tabla en PostgreSQL
        success, column_mapping = create_dataset_table_from_df(df, connection)
        if not success:
            print("❌ No se pudo crear la tabla")
            return None, False
        
        # Insertar datos
        insert_success = insert_dataframe_to_table(df, column_mapping, connection)
        if not insert_success:
            print("❌ No se pudieron insertar los datos")
            return None, False
        
        # Obtener información final de la tabla
        table_info = get_dataset_table_info(connection)
        if table_info:
            print("✅ Dataset cargado exitosamente en PostgreSQL")
            return table_info, False  # False indica que se cargó desde Excel
        else:
            print("⚠️ Tabla creada pero error obteniendo información")
            return None, False
            
    except Exception as e:
        print(f"❌ Error procesando Excel: {e}")
        return None, False

# ======================
# 1. Configuración inicial
# ======================
load_dotenv()

# Crear BD antes de continuar
print("🚀 Inicializando sistema de análisis de datos...")
if not create_database_if_not_exists():
    print("❌ No se pudo crear o acceder a la base de datos. Terminando aplicación.")
    sys.exit(1)

# Probar conexión a la BD objetivo
if not test_target_database_connection():
    print("❌ No se pudo conectar a la base de datos objetivo. Terminando aplicación.")
    sys.exit(1)

# Configurar PostgresSaver PRIMERO
print("🔧 Configurando PostgresSaver...")
connection_string = f"postgresql://{load_db_config()['user']}:{load_db_config()['password']}@{load_db_config()['host']}:{load_db_config()['port']}/{load_db_config()['database']}"

try:
    conn = psycopg.connect(connection_string)
    checkpoint_saver = PostgresSaver(conn)
    checkpoint_saver.setup()
    print("✅ PostgresSaver configurado exitosamente")
except Exception as e:
    print(f"❌ Error configurando PostgresSaver: {e}")
    print("⚠️ Continuando sin checkpoints...")
    checkpoint_saver = None

# NUEVO: Configurar sistema de historial independiente
setup_history_connection()

# AHORA cargar dataset (ya tenemos checkpoint_saver configurado o None)
print("🔄 Inicializando gestión de dataset...")

# Crear conexión temporal si checkpoint_saver falló
dataset_connection = checkpoint_saver.conn if checkpoint_saver else None
if dataset_connection is None:
    try:
        dataset_connection = psycopg.connect(connection_string)
        print("🔗 Conexión temporal creada para dataset")
    except Exception as e:
        print(f"❌ Error creando conexión temporal: {e}")
        print("📄 Continuando solo con archivo Excel...")
        dataset_connection = None

# Cargar dataset
dataset_info, loaded_from_db = load_excel_to_postgres(dataset_connection)

if dataset_info is None:
    print("❌ No se pudo cargar el dataset. Terminando aplicación.")
    sys.exit(1)

# Crear DataFrame temporal
if loaded_from_db and dataset_connection:
    print("🔄 Creando DataFrame temporal desde PostgreSQL...")
    try:
        with dataset_connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {DATASET_CONFIG['table_schema']}.{DATASET_CONFIG['table_name']} LIMIT 1000")
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            print(f"✅ DataFrame temporal creado: {df.shape}")
    except Exception as e:
        print(f"❌ Error creando DataFrame temporal: {e}")
        print("📄 Fallback: cargando desde Excel...")
        df = pd.read_excel(DATASET_CONFIG['excel_path'])
else:
    df = pd.read_excel(DATASET_CONFIG['excel_path'])
    print("✅ Dataset cargado desde Excel")

print(f"📊 Dataset listo: {dataset_info['row_count']} filas, {len(dataset_info['columns'])} columnas")
print(f"🗄️ Tabla: {dataset_info['table_name']}")
print(f"📋 Columnas: {', '.join(dataset_info['columns'][:5])}{'...' if len(dataset_info['columns']) > 5 else ''}")

# Cerrar conexión temporal si se creó separada
if dataset_connection and dataset_connection != (checkpoint_saver.conn if checkpoint_saver else None):
    dataset_connection.close()
    print("🔗 Conexión temporal cerrada")

# Configurar API y directorios
api_key = os.getenv("GOOGLE_API_KEY")
os.makedirs("./outputs", exist_ok=True)

# Inicializar LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
# llm = ChatOllama(model="gemma3", temperature=0)

# ======================
# 2. Definir intérprete Python con acceso a df
# ======================
python_repl = PythonREPLTool()

def run_python_with_df(code: str, error_context: Optional[str] = None):
    """
    Ejecuta código Python con acceso al DataFrame `df` ya cargado.
    Incluye contexto de errores previos para mejor debugging.
    """
    local_vars = {"df": df, "pd": pd, "plt": plt, "sns": sns, "os": os}

    prohibited_patterns = [
        "pd.DataFrame",
        "df = ",
        "data = {",
        "= pd.DataFrame",
        "DataFrame(",
        "# Datos de ejemplo",
        "datos de ejemplo",
        "reemplaza con tu DataFrame"
    ]
    
    code_lower = code.lower()
    for pattern in prohibited_patterns:
        if pattern.lower() in code_lower:
            return {
                "success": False,
                "result": None,
                "error": f"Código bloqueado. Detectado intento de crear DataFrame: '{pattern}'. Usa SOLO el df existente.",
                "error_type": "prohibited_pattern"
            }

    try:
        import ast
        parsed = ast.parse(code)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            last_expr = parsed.body.pop()
            exec(compile(ast.Module(parsed.body, type_ignores=[]), filename="<ast>", mode="exec"), {}, local_vars)
            result = eval(compile(ast.Expression(last_expr.value), filename="<ast>", mode="eval"), {}, local_vars)
        else:
            exec(code, {}, local_vars)
            result = None

        return {
            "success": True,
            "result": result if result is not None else "✅ Código ejecutado con éxito.",
            "error": None,
            "error_type": None
        }
    except Exception as e:
        error_type = type(e).__name__
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "error_type": error_type
        }
    
def get_tools_summary(tools: List[Tool]) -> str:
    """Devuelve un resumen con nombre y descripción de cada tool."""
    return "\n".join([f"- {t.name}: {t.description}" for t in tools])


# ======================
# 3. Tools
# ======================
def get_dataframe(_):
    """
    Devuelve el DataFrame completo al LLM.
    Este tool permite que el agente acceda a 'df' directamente para cualquier análisis.
    """
    return df

def get_summary(_):
    """Devuelve un resumen general del dataset"""
    return str(df.describe(include="all"))

def get_columns(_):
    """Devuelve las columnas del dataset"""
    return str(df.columns.tolist())

def get_missing_values(_):
    """Devuelve la cantidad de valores nulos por columna"""
    return str(df.isnull().sum())

def get_dtypes_and_uniques(_):
    """Devuelve los tipos de datos de cada columna y la cantidad de valores únicos."""
    return str(pd.DataFrame({
        "dtype": df.dtypes,
        "unique_values": df.nunique()
}))

def get_categorical_distribution(column: str):
    """Devuelve la distribución de frecuencias de una columna categórica."""
    if column not in df.columns:
        return f"Columna {column} no encontrada."
    return str(df[column].value_counts(dropna=False).head(20))

def get_numeric_dispersion(_):
    """Devuelve rango, varianza y desviación estándar de variables numéricas."""
    numeric_cols = df.select_dtypes(include=["number"])
    return str(numeric_cols.agg(["min", "max", "var", "std"]))

def get_correlations(_):
    """Devuelve la matriz de correlaciones entre variables numéricas."""
    numeric_cols = df.select_dtypes(include=["number"])
    return str(numeric_cols.corr())

def detect_outliers(column: str):
    """Devuelve los valores atípicos (según IQR) de una columna numérica."""
    if column not in df.columns:
        return f"Columna {column} no encontrada."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"La columna {column} no es numérica."
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)][column]
    return str(outliers.head(50))  # solo mostramos algunos

def get_time_series_summary(_):
    """Devuelve la cantidad de viajes por fecha (si existe columna de fecha)."""
    if "Date" not in df.columns:
        return "No existe columna Date."
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return str(df.groupby(df["Date"].dt.date).size().head(30))

# Gráficas

def plot_histogram(column: str):
    """Genera un histograma de una columna numérica, lo guarda en carpeta y lo muestra en ventana."""
    if column not in df.columns:
        return f"Columna {column} no encontrada."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"La columna {column} no es numérica."
    
    plt.figure(figsize=(10,6))
    df[column].dropna().hist(bins=30, edgecolor="black", alpha=0.7)
    plt.title(f"Histograma de {column}", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.grid(axis="y", alpha=0.5)

    file_path = f"./outputs/histogram_{column}.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Histograma generado y guardado en {file_path}"

def plot_correlation_heatmap(_):
    """Genera un heatmap de correlaciones entre variables numéricas."""
    numeric_cols = df.select_dtypes(include=["number"])
    if numeric_cols.empty:
        return "No hay columnas numéricas para correlacionar."

    import seaborn as sns
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de calor de correlaciones", fontsize=16)

    file_path = "./outputs/correlation_heatmap.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Heatmap de correlaciones generado y guardado en {file_path}"

def plot_time_series(_):
    """Genera una serie temporal de la cantidad de viajes por día (si existe columna Date)."""
    if "Date" not in df.columns:
        return "No existe columna Date."
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    ts = df.groupby(df["Date"].dt.date).size()

    plt.figure(figsize=(14,6))
    ts.plot(kind="line", marker="o", alpha=0.7)
    plt.title("Cantidad de viajes por día", fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Cantidad de viajes", fontsize=12)
    plt.grid(True, alpha=0.5)

    file_path = "./outputs/time_series.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Serie temporal generada y guardada en {file_path}"

def plot_payment_method_distribution(_):
    """Genera un gráfico de barras de los métodos de pago ordenados por frecuencia."""
    if "Payment Method" not in df.columns:
        return "No existe columna Payment Method."
    
    counts = df["Payment Method"].value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(10,6))
    counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Métodos de Pago más frecuentes", fontsize=16)
    plt.xlabel("Método de Pago", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.5)

    file_path = "./outputs/payment_method_distribution.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Gráfico de métodos de pago generado y guardado en {file_path}"

def plot_booking_value_by_vehicle_type(_):
    """Genera un boxplot de Booking Value según Vehicle Type."""
    if "Booking Value" not in df.columns or "Vehicle Type" not in df.columns:
        return "No existen las columnas necesarias (Booking Value, Vehicle Type)."
    
    plt.figure(figsize=(12,6))
    import seaborn as sns
    sns.boxplot(data=df, x="Vehicle Type", y="Booking Value", palette="Set2")
    plt.title("Distribución de Booking Value por tipo de vehículo", fontsize=16)
    plt.xlabel("Tipo de Vehículo", fontsize=12)
    plt.ylabel("Booking Value", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.5)

    file_path = "./outputs/booking_value_by_vehicle_type.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Boxplot generado y guardado en {file_path}"

tools = [
    Tool(name="get_summary", func=get_summary, description="Muestra un resumen estadístico del dataset"),
    Tool(name="get_columns", func=get_columns, description="Muestra las columnas del dataset"),
    Tool(name="get_missing_values", func=get_missing_values, description="Muestra los valores nulos en el dataset"),
    Tool(name="get_dtypes_and_uniques", func=get_dtypes_and_uniques, description="Muestra tipos de datos y cantidad de valores únicos por columna"),
    Tool(name="get_categorical_distribution", func=get_categorical_distribution, description="Muestra distribución de valores en una columna categórica"),
    Tool(name="get_numeric_dispersion", func=get_numeric_dispersion, description="Muestra rango, varianza y desviación estándar de columnas numéricas"),
    Tool(name="get_correlations", func=get_correlations, description="Muestra correlaciones entre variables numéricas"),
    Tool(name="detect_outliers", func=detect_outliers, description="Detecta valores atípicos en una columna numérica"),
    Tool(name="get_time_series_summary", func=get_time_series_summary, description="Muestra cantidad de viajes por fecha"),
    Tool(name="plot_histogram", func=plot_histogram, description="Genera un histograma de una columna numérica"),
    Tool(name="plot_correlation_heatmap", func=plot_correlation_heatmap, description="Genera un heatmap de correlaciones entre variables numéricas"),
    Tool(name="plot_time_series", func=plot_time_series, description="Genera una serie temporal de cantidad de viajes por día"),
    Tool(name="plot_payment_method_distribution", func=plot_payment_method_distribution, description="Genera un gráfico de barras de métodos de pago ordenados por frecuencia"),
    Tool(name="plot_booking_value_by_vehicle_type", func=plot_booking_value_by_vehicle_type, description="Genera un boxplot de Booking Value por tipo de vehículo"),
    Tool(
        name="Python_Interpreter",
        func=run_python_with_df,
        description="Ejecuta código Python con acceso al DataFrame `df` cargado desde Excel. Usa este df para limpiar datos, convertir columnas y generar gráficos."
    ),
]

# ======================
# Prompt unificado para Python
# ======================
def build_code_prompt(query: str, execution_history: List[dict] = None, df_info: dict = None):
    """
    Genera un prompt contextual que incluye historial de errores y información del DataFrame.
    """
    
    # Información básica del DataFrame
    if df_info is None:
        # Limpiar sample para evitar valores NaN que no se pueden serializar
        sample_clean = df.head(2).fillna("NULL").to_dict()
        
        df_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "sample": sample_clean
        }
    
    base_prompt = f"""
Eres un experto en análisis de datos con Python y pandas.

INFORMACIÓN DEL DATAFRAME:
- Columnas disponibles: {df_info['columns']}
- Tipos de datos: {', '.join([f"{col}: {dtype}" for col, dtype in df_info['dtypes'].items()])}
- Dimensiones: {df_info['shape']}

REGLAS IMPORTANTES:
1. Usa EXCLUSIVAMENTE el DataFrame 'df' que ya está cargado
2. NO crees ni simules nuevos DataFrames ni datos de ejemplo
3. Para gráficos, guarda en './outputs/' y usa plt.show()
4. Si trabajas con columnas de tiempo, verifica su tipo primero

TAREA: {query}
"""

    # Agregar historial de errores si existe
    if execution_history:
        base_prompt += "\n\nHISTORIAL DE INTENTOS PREVIOS:\n"
        for i, attempt in enumerate(execution_history, 1):
            if not attempt['success']:
                base_prompt += f"""
Intento {i}:
Código: {attempt.get('code', 'N/A')}
Error: {attempt['error']} (Tipo: {attempt['error_type']})
"""
        
    base_prompt += "\n⚠️ IMPORTANTE: Analiza los errores anteriores y genera un código DIFERENTE que los evite. Solo genera un gráfico UNICAMENTE si el usuario lo pide.\n"
    base_prompt += "\nResponde SOLO con código Python ejecutable, sin explicaciones ni markdown:"
    
    return base_prompt

# Mapeo para invocarlos fácilmente
tool_dict = {t.name: t for t in tools}

# ======================
# 4. Definir estado del grafo
# ======================
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
    thread_id: str  # NUEVO: ID único para identificar la conversación
    # NUEVOS campos para memoria persistente
    conversation_history: List[dict]  # Historial completo de consultas y respuestas
    session_start_time: str          # Timestamp de inicio de sesión
    total_queries: int               # Contador total acumulativo

# ======================
# Función para gestionar historial conversacional
# ======================
def format_conversation_context(conversation_history: List[dict], max_entries: int = 5) -> str:
    """Formatea el historial de conversación para incluir en prompts"""
    if not conversation_history:
        return ""
    
    # Tomar las últimas max_entries consultas
    recent_history = conversation_history[-max_entries:]
    
    context_parts = []
    for i, entry in enumerate(recent_history, 1):
        query = entry.get("query", "N/A")
        response = entry.get("response", "N/A")
        success = entry.get("success", False)
        
        context_parts.append(f"""
Consulta {i}: {query}
Respuesta: {response[:200]}{"..." if len(str(response)) > 200 else ""}
Estado: {'Éxito' if success else 'Error'}
""")
    
    return "\n".join(context_parts)

# ======================
# 5. Nodos del grafo
# ======================
def node_clasificar(state: AgentState):
    """El LLM decide qué acción tomar con contexto mejorado Y memoria conversacional"""
    
    # Obtener información del DataFrame si no existe
    if not state.get("df_info"):
        # Limpiar sample para evitar valores NaN
        sample_clean = df.head(2).fillna("NULL").to_dict()
        
        state["df_info"] = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "sample": sample_clean
        }
    
    # Contexto de iteraciones previas
    iteration_context = ""
    if state["iteration_count"] > 0:
        iteration_context = f"\nEsta es la iteración #{state['iteration_count'] + 1}. Intentos previos han fallado."
        if state["execution_history"]:
            last_error = state["execution_history"][-1].get("error", "")
            iteration_context += f"\nÚltimo error: {last_error}"

    # NUEVO: Contexto conversacional
    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = f"""

HISTORIAL DE CONVERSACIÓN PREVIA:
{format_conversation_context(state["conversation_history"])}

IMPORTANTE: Tienes acceso al historial de consultas anteriores. Si el usuario hace referencia a preguntas previas ("la pregunta anterior", "lo que pregunté antes", etc.), usa este historial para responder.
"""

    tools_summary = get_tools_summary(tools)

    prompt = f"""
Eres un asistente de análisis de datos experto con memoria conversacional. Analiza esta consulta y decide la mejor acción.

CONSULTA ACTUAL: {state['query']}
DATAFRAME INFO: Columnas = {state['df_info']['columns']}, Shape = {state['df_info']['shape']}
{iteration_context}
{conversation_context}

HERRAMIENTAS DISPONIBLES:
{tools_summary}

DECISIÓN:
Analiza la consulta actual considerando el historial previo si es relevante. 
Selecciona la herramienta más adecuada. 
Si ninguna herramienta especializada es suficiente, usa Python_Interpreter.

Formato de salida:
Thought: <análisis detallado de la consulta y estrategia, considerando historial si aplica>
Action: <nombre exacto de la herramienta elegida>
"""

    response = llm.invoke(prompt).content.strip()

    # Extraer thought y action
    thought, action = "", "Python_Interpreter"
    for line in response.splitlines():
        if line.lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()

    state["thought"] = thought
    state["action"] = action
    state["history"].append(f"Iteración {state['iteration_count']} - Clasificar → {thought[:100]}...")

    print(f"\n🧠 Iteración {state['iteration_count']} - Thought: {thought}")
    print(f"➡️ Action: {action}")

    return state

def node_ejecutar_python(state: AgentState):
    """Ejecuta código Python con manejo robusto de errores y contexto"""
    
    print(f"⚙️ Ejecutando Python - Intento {state['iteration_count'] + 1}")
    
    # Generar código con contexto completo
    code_prompt = build_code_prompt(
        state["query"], 
        state["execution_history"], 
        state["df_info"]
    )
    
    # Generar código
    python_code = llm.invoke(code_prompt).content.strip()
    
    # Limpiar markdown
    if python_code.startswith("```"):
        python_code = python_code.strip("`")
        if python_code.lower().startswith("python"):
            python_code = python_code[len("python"):].strip()
        python_code = python_code.replace("```", "").strip()

    print(f"\n🔍 Código generado:\n{python_code}")
    
    # Ejecutar código
    execution_result = run_python_with_df(python_code)
    
    # Crear registro de ejecución
    execution_record = {
        "iteration": state["iteration_count"],
        "code": python_code,
        "success": execution_result["success"],
        "result": execution_result["result"],
        "error": execution_result["error"],
        "error_type": execution_result["error_type"]
    }
    
    # Actualizar historial
    state["execution_history"].append(execution_record)
    state["result"] = execution_result["result"]
    state["success"] = execution_result["success"]
    
    if execution_result["success"]:
        print(f"✅ Éxito: {execution_result['result']}")
    else:
        print(f"❌ Error: {execution_result['error']}")
        state["final_error"] = execution_result["error"]
    
    state["history"].append(f"Ejecutar Python → {'Éxito' if execution_result['success'] else 'Error: ' + str(execution_result['error'])}")
    
    return state

def node_validar_y_decidir(state: AgentState):
    """Valida el resultado y decide si continuar iterando"""
    
    state["iteration_count"] += 1
    success = state.get("success", False)
    max_iterations = state.get("max_iterations", 3)
    
    print(f"\n🔍 Validación - Iteración {state['iteration_count']}")
    print(f"   Éxito: {success}")
    print(f"   Iteraciones restantes: {max_iterations - state['iteration_count']}")
    
    # Decidir próxima acción
    if success:
        state["next_node"] = "responder"
        print("   ➡️ Decisión: Proceder a responder (éxito)")
    elif state["iteration_count"] >= max_iterations:
        state["next_node"] = "responder"
        print("   ➡️ Decisión: Proceder a responder (máximo de iteraciones alcanzado)")
    else:
        state["next_node"] = "clasificar"
        print("   ➡️ Decisión: Nueva iteración")
    
    state["history"].append(f"Validar → Iteración {state['iteration_count']}, Éxito: {success}, Próximo: {state['next_node']}")
    
    return state

def node_responder(state: AgentState):
    """Genera la respuesta final CON contexto conversacional y actualiza el historial"""
    
    success = state.get("success", False)
    
    # Contexto conversacional para el prompt
    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = f"""

HISTORIAL DE CONVERSACIÓN:
{format_conversation_context(state["conversation_history"])}
"""

    if success:
        prompt = f"""
Pregunta del usuario: {state['query']}
Resultado obtenido: {state['result']}
Número de iteraciones necesarias: {state['iteration_count']}
{conversation_context}

Genera una respuesta clara y amigable en español explicando qué se logró.
Si la consulta hace referencia a conversaciones anteriores, asegúrate de conectar tu respuesta con ese contexto.
"""
    else:
        errors_summary = []
        for record in state["execution_history"]:
            if not record["success"]:
                errors_summary.append(f"- {record['error_type']}: {record['error']}")
        
        prompt = f"""
Pregunta del usuario: {state['query']}
Después de {state['iteration_count']} iteraciones, no se pudo completar la tarea.
{conversation_context}

Errores encontrados:
{chr(10).join(errors_summary)}

Genera una respuesta empática en español explicando:
1. Que se intentó resolver la consulta múltiples veces
2. Los principales problemas encontrados (en términos simples)
3. Sugerencias para el usuario (ej: verificar formato de datos, columnas, etc.)
"""

    respuesta = llm.invoke(prompt).content
    print(f"\n🤖 Respuesta Final:\n{respuesta}")
    
    # ACTUALIZAR historial conversacional
    new_conversation_entry = {
        "query": state["query"],
        "response": respuesta,
        "success": success,
        "timestamp": pd.Timestamp.now().isoformat(),
        "iterations_needed": state["iteration_count"]
    }
    
    # Agregar nueva entrada al historial
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append(new_conversation_entry)
    
    # Actualizar contador total
    state["total_queries"] = state.get("total_queries", 0) + 1
    
    # NUEVO: GUARDAR ESTADO INMEDIATAMENTE
    config = {"configurable": {"thread_id": state.get("thread_id", PERSISTENT_THREAD_ID)}}
    save_success = save_conversation_state(state, config)
    
    if save_success:
        print(f"💾 Historial guardado exitosamente - {state['total_queries']} consultas totales")
    else:
        print("⚠️ Error guardando historial en base de datos")
    
    # Log final
    state["history"].append(f"Responder → Finalizado con {'éxito' if success else 'error'}")
    
    return state

def route_after_validation(state: AgentState):
    """Determina la siguiente ruta basada en la validación"""
    success = state.get("success", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    print(f"\n🔧 DEBUG route_after_validation:")
    print(f"   Success: {success}")
    print(f"   Iteration: {iteration_count}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Next node from state: {state.get('next_node', 'N/A')}")
    
    if success:
        print("   → Routing to: responder (success)")
        return "responder"
    elif iteration_count >= max_iterations:
        print("   → Routing to: responder (max iterations reached)")
        return "responder"
    else:
        print("   → Routing to: clasificar (continue iteration)")
        return "clasificar"

# ======================
# 6. Construir el grafo
# ======================
def create_graph():
    graph = StateGraph(AgentState)

    # Nodos
    graph.add_node("clasificar", node_clasificar)
    graph.add_node("ejecutar_python", node_ejecutar_python)
    graph.add_node("validar", node_validar_y_decidir)
    graph.add_node("responder", node_responder)

    # Punto de entrada
    graph.set_entry_point("clasificar")

    # Flujo principal
    graph.add_edge("clasificar", "ejecutar_python")
    graph.add_edge("ejecutar_python", "validar")
    
    # Enrutamiento condicional desde validación
    graph.add_conditional_edges("validar", route_after_validation)
    
    # Fin del grafo
    graph.add_edge("responder", END)

    # Compilar con o sin checkpointer según esté disponible
    if checkpoint_saver is not None:
        return graph.compile(checkpointer=checkpoint_saver)
    else:
        print("⚠️ Compilando grafo sin checkpoints")
        return graph.compile()

def main():
    app = create_graph()
    
    print("✅ Base de datos PostgreSQL configurada correctamente")
    print("✅ Sistema de checkpoints activado")
    print(f"🔄 Límite de conversaciones: {MAX_CONVERSATIONS_PER_THREAD} por thread")
    print("🚀 Sistema de Análisis de Datos con Memoria Persistente")
    print("   Basado en LangGraph con historial conversacional limitado")
    print("   Escribe 'salir' para terminar")
    print("   Escribe 'limpiar' para forzar limpieza de conversaciones\n")
    
    # El sistema de historial ya está configurado, no necesitamos forzar creación
    if history_connection:
        print("🔧 Sistema de historial listo")
    else:
        print("⚠️ Sistema de historial no disponible")
    
    # Cargar contexto ANTES del bucle para inicialización correcta
    print("🔍 Verificando historial existente...")
    thread_exists = check_thread_exists()
    previous_context = None
    
    if thread_exists:
        previous_context = load_previous_context()
        if previous_context:
            summary = get_conversation_summary()
            print(f"💭 Memoria encontrada: {summary}")
            print("   Continuando conversación existente...\n")
            
            # Verificar y limpiar si es necesario al inicio
            cleanup_old_conversations(PERSISTENT_THREAD_ID, MAX_CONVERSATIONS_PER_THREAD)
        else:
            print("⚠️ Thread existe pero no se pudo cargar contexto\n")
            thread_exists = False
    else:
        print("🆕 Iniciando nueva conversación\n")
    
    while True:
        query = input("Pregunta sobre el dataset (o 'salir' / 'limpiar'): ")
        
        if query.lower() == "salir":
            break
        elif query.lower() == "limpiar":
            print("🧹 Ejecutando limpieza manual...")
            cleanup_old_conversations(PERSISTENT_THREAD_ID, MAX_CONVERSATIONS_PER_THREAD)
            continue

        # Crear estado inicial basado en contexto pre-cargado
        if previous_context and isinstance(previous_context, dict):
            # Continuar desde estado previo
            initial_state = {
                "query": query,
                "action": "",
                "result": None,
                "thought": "",
                "history": [],
                "execution_history": [],
                "iteration_count": 0,
                "max_iterations": 3,
                "df_info": previous_context.get("df_info", {}),
                "success": False,
                "final_error": None,
                "next_node": "clasificar",
                "thread_id": PERSISTENT_THREAD_ID,
                # Cargar historial existente
                "conversation_history": previous_context.get("conversation_history", []),
                "session_start_time": previous_context.get("session_start_time", str(pd.Timestamp.now())),
                "total_queries": previous_context.get("total_queries", 0)
            }
        else:
            # Nuevo estado inicial solo si NO hay contexto previo
            initial_state = {
                "query": query,
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
                "thread_id": PERSISTENT_THREAD_ID,
                # Nuevo historial
                "conversation_history": [],
                "session_start_time": str(pd.Timestamp.now()),
                "total_queries": 0
            }

        # Configuración para el checkpointer con thread fijo
        config = {"configurable": {"thread_id": PERSISTENT_THREAD_ID}}

        print(f"\n{'='*60}")
        print(f"🔄 Procesando consulta: {query}")
        print(f"💭 Total de consultas en sesión: {initial_state['total_queries']}")
        print(f"{'='*60}")
        
        try:
            final_state = app.invoke(initial_state, config=config)
            
            # Actualizar contexto para próximas consultas en la misma sesión
            previous_context = final_state.copy()
            thread_exists = True
            
            print(f"\n📊 RESUMEN DE EJECUCIÓN:")
            print(f"   Iteraciones totales: {final_state['iteration_count']}")
            print(f"   Éxito: {final_state.get('success', False)}")
            print(f"   Total consultas en historial: {final_state.get('total_queries', 0)}")
            if not final_state.get('success', False):
                print(f"   Error final: {final_state.get('final_error', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error crítico en el sistema: {e}")
            
        print(f"\n{'-'*60}\n")

if __name__ == "__main__":
    main()

