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
import json
from langgraph.checkpoint.postgres import PostgresSaver
import uuid
from datetime import datetime
import re

def clean_data_for_json(data):
    """Función simplificada solo para datasets"""
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
    
def clean_state_for_serialization(state):
    """
    Limpia el estado para que sea serializable por msgpack.
    Remueve o convierte objetos no serializables como DataFrames.
    """
    cleaned_state = state.copy()
    
    # Limpiar sql_results si contiene DataFrame
    if "sql_results" in cleaned_state and hasattr(cleaned_state["sql_results"], 'to_dict'):
        df = cleaned_state["sql_results"]
        cleaned_state["sql_results"] = {
            "data": df.head(100).to_dict('records'),  # Limitar a 100 filas
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "serialized": True
        }
    
    # Limpiar otros campos que puedan contener objetos no serializables
    if "df_info" in cleaned_state and "sample" in cleaned_state["df_info"]:
        # Ya está limpio por clean_data_for_json, pero verificar
        pass
    
    # Limpiar execution_history de posibles objetos no serializables
    if "execution_history" in cleaned_state:
        for record in cleaned_state["execution_history"]:
            if "result" in record and hasattr(record["result"], 'to_dict'):
                # Si el resultado es un DataFrame, convertirlo
                record["result"] = f"DataFrame con shape {record['result'].shape}"
    
    return cleaned_state

# Suprimir warnings de Google Cloud
import warnings
warnings.filterwarnings('ignore', message='.*ALTS.*')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Variable para controlar el guardado automático del Excel a PostgreSQL
ENABLE_AUTO_SAVE_TO_DB = False  # Cambiar a False para desactivar

# Configuración del archivo Excel y tabla PostgreSQL
DATASETS_TO_PROCESS = [
    {
        "excel_path": "./Data/ncr_ride_bookings.xlsx",
        "table_name": "dataset_rides",
        "table_schema": "public"
    },
    {
        "excel_path": "./Data/crocodile_dataset.xlsx",
        "table_name": "crocodile_dataset", 
        "table_schema": "public"
    }
]

DATASET_CONFIG = DATASETS_TO_PROCESS[0]  # Mantener ncr_ride_bookings como principal

# Variable global para PostgresSaver
postgres_saver = None

SINGLE_USER_THREAD_ID = "single_user_persistent_thread"
SINGLE_USER_ID = "default_user"

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

def setup_postgres_saver():
    """
    Configura e inicializa PostgresSaver para memoria de conversaciones.
    CORREGIDO: Usa autocommit para evitar problemas con índices concurrentes
    """
    print("🧠 Configurando PostgresSaver para memoria conversacional...")
    
    try:
        db_config = load_db_config()
        
        # Crear connection string para PostgresSaver
        postgres_uri = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        # CORRECCIÓN: Crear con autocommit para evitar error de índices concurrentes
        conn = psycopg.connect(postgres_uri, autocommit=True)
        checkpointer = PostgresSaver(conn)
        
        # Configurar las tablas automáticamente
        try:
            checkpointer.setup()
            print("✅ PostgresSaver configurado exitosamente")
            print("📊 Tablas de memoria creadas: checkpoints, checkpoint_blobs, checkpoint_writes")
            return checkpointer
        except Exception as setup_error:
            print(f"⚠️ Error en setup: {setup_error}")
            # Fallback a método alternativo
            return setup_postgres_saver_alternative()
        
    except Exception as e:
        print(f"❌ Error configurando PostgresSaver: {e}")
        # Intentar método alternativo
        return setup_postgres_saver_alternative()

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

# Variable global para conexión de datos
data_connection = None

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
    
def setup_data_connection():
    """
    Configura una conexión independiente para operaciones de datos.
    """
    global data_connection
    
    print("🔧 Configurando conexión para datos...")
    
    try:
        db_config = load_db_config()
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        data_connection = psycopg.connect(connection_string)
        print("✅ Conexión de datos configurada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error configurando conexión de datos: {e}")
        data_connection = None
        return False
    
def initialize_dataset_on_startup():
    """
    Inicializa todos los datasets en BD SOLO si ENABLE_AUTO_SAVE_TO_DB es True.
    Se ejecuta automáticamente al iniciar el programa.
    """
    global ENABLE_AUTO_SAVE_TO_DB
    
    if not ENABLE_AUTO_SAVE_TO_DB:
        print("💾 Guardado automático desactivado - No se crearán tablas al inicio")
        return True
    
    print("💾 Guardado automático activado - Verificando/creando tablas de datasets...")
    
    # Crear conexión para verificar/crear tablas
    dataset_connection = None
    if dataset_connection is None:
        try:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            dataset_connection = psycopg.connect(connection_string)
            print("🔗 Conexión creada para inicialización de datasets")
        except Exception as e:
            print(f"❌ Error creando conexión para datasets: {e}")
            return False
    
    success_count = 0
    total_files = len(DATASETS_TO_PROCESS)
    
    try:
        for dataset_config in DATASETS_TO_PROCESS:
            excel_path = dataset_config['excel_path']
            table_name = dataset_config['table_name']
            table_schema = dataset_config['table_schema']
            
            print(f"\n🔍 Procesando: {os.path.basename(excel_path)} → {table_name}")
            
            # Verificar si la tabla ya existe
            if check_dataset_table_exists(dataset_connection, table_name, table_schema):
                print(f"✅ Tabla '{table_name}' ya existe en BD")
                success_count += 1
                continue
            
            # Verificar que el archivo existe
            if not os.path.exists(excel_path):
                print(f"❌ Archivo Excel no encontrado: {excel_path}")
                continue
            
            print(f"📂 Cargando Excel para crear tabla en BD...")
            df_temp = pd.read_excel(excel_path)
            
            # Crear tabla
            success, column_mapping = create_dataset_table_from_df(df_temp, dataset_connection, table_name, table_schema)
            if not success:
                print(f"❌ No se pudo crear la tabla {table_name}")
                continue
            
            # Insertar datos
            insert_success = insert_dataframe_to_table(df_temp, column_mapping, dataset_connection, table_name, table_schema)
            if insert_success:
                print(f"✅ Dataset '{table_name}' inicializado correctamente en BD")
                success_count += 1
            else:
                print(f"❌ Error insertando datos en {table_name}")
        
        print(f"\n📋 Resumen: {success_count}/{total_files} datasets inicializados correctamente")
        return success_count > 0
            
    except Exception as e:
        print(f"❌ Error en inicialización de datasets: {e}")
        return False
    finally:
        # Cerrar conexión temporal si se creó
        if dataset_connection:
            dataset_connection.close()

    
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

def check_dataset_table_exists(connection=None, table_name=None, table_schema=None):
    """
    Verifica si una tabla específica existe en PostgreSQL.
    """
    conn = connection
    
    if conn is None:
        print("⚠️ No se puede verificar tabla: no hay conexión disponible")
        return False
    
    # Usar valores por defecto si no se especifican
    table_name = table_name or DATASET_CONFIG["table_name"]
    table_schema = table_schema or DATASET_CONFIG["table_schema"]
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_name = %s
                )
            """, (table_schema, table_name))
            
            exists = cursor.fetchone()[0]
            print(f"🔍 Tabla '{table_name}' {'existe' if exists else 'no existe'} en BD")
            return exists
            
    except Exception as e:
        print(f"⚠️ Error verificando tabla {table_name}: {e}")
        return False

def get_dataset_table_info(connection=None):
    """
    Obtiene información de la tabla del dataset.
    Acepta una conexión opcional.
    """
    conn = connection
    
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

def get_dataset_table_info_by_name(table_name, connection=None):
    """
    Obtiene información de una tabla específica por nombre.
    """
    conn = connection
    
    if conn is None:
        try:
            db_config = load_db_config()
            temp_connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(temp_connection_string)
            temp_connection = True
        except Exception as e:
            return None
    else:
        temp_connection = False
    
    try:
        with conn.cursor() as cursor:
            # Obtener información de columnas
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns_info = cursor.fetchall()
            
            # Obtener conteo de filas
            cursor.execute(f'SELECT COUNT(*) FROM public."{table_name}"')
            row_count = cursor.fetchone()[0]
            
            # Formatear información
            columns = [col[0] for col in columns_info]
            dtypes = {col[0]: col[1] for col in columns_info}
            
            return {
                "columns": columns,
                "dtypes": dtypes,
                "row_count": row_count,
                "table_name": table_name
            }
            
    except Exception as e:
        print(f"⚠️ Error obteniendo información de tabla {table_name}: {e}")
        return None
    finally:
        if temp_connection:
            conn.close()
    
def list_stored_tables(connection=None):
    """
    Lista todas las tablas almacenadas en la BD, excluyendo tablas del sistema.
    """
    # Intentar múltiples fuentes de conexión
    conn = connection
    
    if conn is None:
        # Crear conexión temporal si no hay ninguna disponible
        try:
            db_config = load_db_config()
            temp_connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(temp_connection_string)
            temp_connection = True
            print("🔗 Conexión temporal creada para listar tablas")
        except Exception as e:
            print(f"⚠️ No se pudo crear conexión para listar tablas: {e}")
            return []
    else:
        temp_connection = False
    
    try:
        with conn.cursor() as cursor:
            # Primera consulta: verificar todas las tablas en el esquema public
            cursor.execute("""
                SELECT table_name, table_type
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name
            """)
            
            all_tables = cursor.fetchall()
            
            # Filtrar tablas del sistema y tablas de checkpoint
            excluded_tables = {
                'checkpoint_blobs', 
                'checkpoint_migrations', 
                'checkpoint_writes', 
                'checkpoints'
            }

            dataset_tables = []

            for table_name, table_type in all_tables:
                # Solo agregar tablas que no sean del sistema ni de checkpoints
                if table_name not in excluded_tables:
                    dataset_tables.append(table_name)
            
            return dataset_tables
            
    except Exception as e:
        print(f"⚠️ Error listando tablas: {e}")
        return []
    finally:
        # Cerrar conexión temporal si se creó
        if temp_connection:
            conn.close()

def show_stored_files():
    """
    Muestra los archivos almacenados en la BD de forma amigable.
    """
    print("🔍 Buscando tablas en la base de datos...")
    stored_tables = list_stored_tables()
    
    if not stored_tables:
        print("📁 No se encontraron tablas de dataset en la base de datos")
        print("💡 Verifica que las tablas se hayan creado correctamente")
        
        # Mostrar información adicional para debugging
        print("\n🔧 Para verificar manualmente, puedes ejecutar en PostgreSQL:")
        print("   SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        return
    
    print(f"📁 Los siguientes archivos se encuentran en mi BD. Puedes preguntar sobre ellos ({len(stored_tables)} encontrados):")
    for i, table_name in enumerate(stored_tables, 1):
        print(f"   {i}. {table_name}")

def show_conversation_memory(thread_id: str):
    """
    Muestra un resumen de la memoria de conversación para debugging.
    """
    # Primero hacer debugging de la estructura
    # debug_checkpoint_structure(thread_id)
    
    conversation_history, user_context = load_conversation_history(thread_id)
    
    if conversation_history:
        print(f"🧠 Memoria encontrada:")
        print(f"   📚 {len(conversation_history)} conversaciones previas")
        print(f"   📊 Datasets usados: {user_context.get('common_datasets', [])}")
        print(f"   🎯 Estrategia preferida: {user_context.get('preferred_analysis_type', 'N/A')}")
        print(f"   ⚠️ Patrones de error: {len(user_context.get('error_patterns', []))}")
        
        # Mostrar última conversación
        if conversation_history:
            last_conv = conversation_history[-1]
            print(f"   🕒 Última consulta: {last_conv.get('query', 'N/A')[:50]}...")
            print(f"   ✅ Fue exitosa: {last_conv.get('success', False)}")
    else:
        print("🧠 No se encontró memoria previa")
    
    print()

def create_dataset_table_from_df(df: pd.DataFrame, connection=None, table_name=None, table_schema=None):
    """
    Crea una tabla en PostgreSQL desde un DataFrame.
    """
    conn = connection
    
    if conn is None:
        print("⚠️ No se puede crear tabla: no hay conexión disponible")
        return False, {}
    
    # Usar valores por defecto si no se especifican
    table_name = table_name or DATASET_CONFIG["table_name"]
    table_schema = table_schema or DATASET_CONFIG["table_schema"]
    
    try:
        postgres_types = get_postgres_data_types()
        
        # Limpiar nombres de columnas y crear mapeo
        original_columns = list(df.columns)
        clean_columns = [sanitize_column_name(col) for col in original_columns]
        column_mapping = dict(zip(original_columns, clean_columns))
        
        print(f"📝 Creando tabla '{table_name}' con {len(df.columns)} columnas...")
        
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
                CREATE TABLE {table_schema}.{table_name} (
                    id SERIAL PRIMARY KEY,
                    {', '.join(column_definitions)},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            
            cursor.execute(create_table_sql)
            conn.commit()
            
            print(f"✅ Tabla '{table_name}' creada exitosamente")
            return True, column_mapping
            
    except Exception as e:
        print(f"❌ Error creando tabla {table_name}: {e}")
        if conn:
            conn.rollback()
        return False, {}

def insert_dataframe_to_table(df: pd.DataFrame, column_mapping: dict, connection=None, table_name=None, table_schema=None):
    """
    Inserta los datos del DataFrame en la tabla PostgreSQL.
    """
    conn = connection
    
    if conn is None:
        print("⚠️ No se puede insertar datos: no hay conexión disponible")
        return False
    
    # Usar valores por defecto si no se especifican
    table_name = table_name or DATASET_CONFIG["table_name"]
    table_schema = table_schema or DATASET_CONFIG["table_schema"]
    
    try:
        # Renombrar columnas según el mapeo
        df_clean = df.rename(columns=column_mapping)
        
        print(f"📥 Insertando {len(df_clean)} filas en la tabla '{table_name}'...")
        
        # Preparar datos para inserción
        columns_list = list(column_mapping.values())
        placeholders = ', '.join(['%s'] * len(columns_list))
        columns_str = ', '.join([f'"{col}"' for col in columns_list])
        
        insert_sql = f"""
            INSERT INTO {table_schema}.{table_name} 
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
            cursor.execute(f"SELECT COUNT(*) FROM {table_schema}.{table_name}")
            inserted_count = cursor.fetchone()[0] - 1  # Restar 1 por el ID serial
            
            print(f"✅ {inserted_count} filas insertadas correctamente en '{table_name}'")
            return True
            
    except Exception as e:
        print(f"❌ Error insertando datos en {table_name}: {e}")
        if conn:
            conn.rollback()
        return False

def load_excel_to_postgres(connection=None, force_load=False):
    """
    Función principal que carga el archivo Excel a PostgreSQL si no existe.
    Acepta una conexión opcional y un flag para forzar carga.
    """
    global ENABLE_AUTO_SAVE_TO_DB

    # Verificar si ya existe la tabla SIEMPRE (independiente de ENABLE_AUTO_SAVE_TO_DB)
    if check_dataset_table_exists(connection):
        print("✅ Dataset ya está cargado en PostgreSQL")
        table_info = get_dataset_table_info(connection)
        if table_info:
            return table_info, True  # True indica que se cargó desde BD
        else:
            print("⚠️ Error obteniendo info de tabla existente")
    
    # Si no existe, cargar Excel y crear tabla (solo si está activado)
    print(f"📂 Cargando archivo Excel: {DATASET_CONFIG['excel_path']}")
    
    try:
        # Verificar que el archivo existe
        if not os.path.exists(DATASET_CONFIG['excel_path']):
            print(f"❌ Archivo Excel no encontrado: {DATASET_CONFIG['excel_path']}")
            return None, False
        
        # Cargar DataFrame
        df_temp = pd.read_excel(DATASET_CONFIG['excel_path'])
        print(f"📊 Excel cargado: {df_temp.shape[0]} filas, {df_temp.shape[1]} columnas")
        
        # Si el guardado está desactivado, retornar solo info del Excel
        if not ENABLE_AUTO_SAVE_TO_DB:
            table_info = {
                "columns": list(df_temp.columns),
                "dtypes": {col: str(dtype) for col, dtype in df_temp.dtypes.items()},
                "row_count": len(df_temp),
                "table_name": "excel_only_mode"
            }
            return table_info, False
        
        # Crear tabla en PostgreSQL (solo si está activado)
        success, column_mapping = create_dataset_table_from_df(df_temp, connection)
        if not success:
            print("❌ No se pudo crear la tabla")
            return None, False
        
        # Insertar datos
        insert_success = insert_dataframe_to_table(df_temp, column_mapping, connection)
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

def load_conversation_history(thread_id: str):
    """
    Recupera el historial de conversaciones desde PostgresSaver para un thread específico.
    CORREGIDO: Maneja correctamente la estructura del checkpoint
    """
    global postgres_saver
    
    if not postgres_saver:
        print("⚠️ PostgresSaver no disponible para recuperar historial")
        return [], {}
    
    try:
        # Obtener el checkpoint más reciente para este thread
        config = {"configurable": {"thread_id": thread_id}}
        
        # Intentar obtener el estado más reciente
        checkpoint = postgres_saver.get(config)
        
        if checkpoint:
            # El checkpoint es un diccionario con la estructura correcta
            # Buscar en los valores del estado
            state_values = None
            
            # Intentar diferentes formas de acceder a los datos
            if isinstance(checkpoint, dict):
                # Caso 1: El checkpoint es directamente el diccionario de estado
                if "conversation_history" in checkpoint:
                    state_values = checkpoint
                # Caso 2: Los valores están en una clave específica
                elif "values" in checkpoint:
                    state_values = checkpoint["values"]
                elif "state" in checkpoint:
                    state_values = checkpoint["state"]
                # Caso 3: Buscar en las claves del checkpoint
                else:
                    # Buscar cualquier clave que contenga conversation_history
                    for key, value in checkpoint.items():
                        if isinstance(value, dict) and "conversation_history" in value:
                            state_values = value
                            break
            
            if state_values and "conversation_history" in state_values:
                # Recuperar historial de conversaciones
                conversation_history = state_values.get("conversation_history", [])
                user_context = state_values.get("user_context", {
                    "preferred_analysis_type": None,
                    "common_datasets": [],
                    "visualization_preferences": [],
                    "error_patterns": [],
                    "last_interaction": None
                })
                
                print(f"📚 Historial recuperado: {len(conversation_history)} conversaciones previas")
                print(f"👤 Contexto de usuario recuperado: {len(user_context.get('common_datasets', []))} datasets comunes")
                
                return conversation_history, user_context
            else:
                print("📭 No se encontró historial en la estructura del checkpoint")
                # Para debugging: mostrar las claves disponibles
                if isinstance(checkpoint, dict):
                    print(f"🔍 Claves disponibles en checkpoint: {list(checkpoint.keys())}")
                return [], {}
        else:
            print("📭 No se encontró checkpoint previo para este thread")
            return [], {}
            
    except Exception as e:
        print(f"⚠️ Error recuperando historial: {e}")
        # Para debugging: mostrar más información del error
        print(f"🔍 Tipo de error: {type(e).__name__}")
        if hasattr(e, 'args'):
            print(f"🔍 Detalles: {e.args}")
        return [], {}
    
def debug_checkpoint_structure(thread_id: str):
    """
    Función de debugging para inspeccionar la estructura del checkpoint.
    """
    global postgres_saver
    
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

def extract_learned_patterns_from_history(conversation_history: List[dict]) -> List[str]:
    """
    Extrae patrones aprendidos del historial de conversaciones existente.
    """
    patterns = []
    
    # Analizar conversaciones exitosas
    successful_conversations = [conv for conv in conversation_history if conv.get("success", False)]
    
    if successful_conversations:
        # Estrategias más exitosas
        strategies = [conv["strategy_used"] for conv in successful_conversations if conv.get("strategy_used")]
        if strategies:
            most_common = max(set(strategies), key=strategies.count)
            patterns.append(f"Estrategia más exitosa: {most_common}")
        
        # Datasets más utilizados
        datasets = [conv["dataset_used"] for conv in successful_conversations if conv.get("dataset_used") != "unknown"]
        if datasets:
            most_common_dataset = max(set(datasets), key=datasets.count)
            patterns.append(f"Dataset más usado: {most_common_dataset}")
        
        # Patrones de iteraciones
        iterations = [conv["iterations"] for conv in successful_conversations if conv.get("iterations", 0) > 1]
        if iterations:
            avg_iterations = sum(iterations) / len(iterations)
            patterns.append(f"Promedio iteraciones complejas: {avg_iterations:.1f}")
    
    return patterns[-5:]  # Solo los 5 más relevantes
    
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

# ======================
# FUNCIONES PARA GESTIÓN MÚLTIPLE DE DATASETS
# ======================

def get_all_available_datasets(connection=None):
    """
    Obtiene metadatos completos de todos los datasets disponibles (BD + archivos Excel).
    Combina información de tablas en BD y archivos Excel disponibles.
    """
    conn = connection
    available_datasets = {}
    
    # 1. Obtener tablas de la BD
    stored_tables = list_stored_tables(conn)
    
    for table_name in stored_tables:
        table_info = get_dataset_table_info_by_name(table_name, conn)
        if table_info:
            # Buscar configuración correspondiente
            dataset_config = None
            for config in DATASETS_TO_PROCESS:
                if config["table_name"] == table_name:
                    dataset_config = config
                    break
            
            available_datasets[table_name] = {
                "source": "database",
                "table_name": table_name,
                "friendly_name": get_friendly_dataset_name(table_name),
                "columns": table_info["columns"][:10],  # Primeras 10 columnas
                "row_count": table_info["row_count"],
                "main_columns": identify_key_columns(table_info["columns"]),
                "description": generate_dataset_description(table_name, table_info["columns"]),
                "excel_path": dataset_config["excel_path"] if dataset_config else None,
                "keywords": generate_dataset_keywords(table_name, table_info["columns"])
            }
    
    # 2. Agregar archivos Excel que no estén en BD
    for config in DATASETS_TO_PROCESS:
        table_name = config["table_name"]
        if table_name not in available_datasets and os.path.exists(config["excel_path"]):
            try:
                # Leer solo las primeras filas para metadatos
                df_sample = pd.read_excel(config["excel_path"], nrows=5)
                available_datasets[table_name] = {
                    "source": "excel_file",
                    "table_name": table_name,
                    "friendly_name": get_friendly_dataset_name(table_name),
                    "columns": list(df_sample.columns)[:10],
                    "row_count": "Estimado: " + str(len(df_sample) * 100),  # Estimación
                    "main_columns": identify_key_columns(list(df_sample.columns)),
                    "description": generate_dataset_description(table_name, list(df_sample.columns)),
                    "excel_path": config["excel_path"],
                    "keywords": generate_dataset_keywords(table_name, list(df_sample.columns))
                }
            except Exception as e:
                print(f"⚠️ Error leyendo Excel {config['excel_path']}: {e}")
    
    return available_datasets

def get_friendly_dataset_name(table_name):
    """Convierte nombres de tabla a nombres amigables"""
    name_mapping = {
        "dataset_rides": "Dataset de Viajes NCR",
        "crocodile_dataset": "Dataset de Cocodrilos",
        "ncr_ride_bookings": "Reservas de Viajes NCR"
    }
    return name_mapping.get(table_name, table_name.replace("_", " ").title())

def identify_key_columns(columns):
    """Identifica las columnas más importantes basándose en nombres comunes"""
    key_patterns = {
        "id": ["id", "identifier", "key"],
        "date": ["date", "time", "created", "updated", "timestamp"],
        "location": ["location", "city", "address", "place", "destination"],
        "amount": ["price", "cost", "amount", "value", "fare", "payment"],
        "category": ["type", "category", "method", "status", "class"],
        "user": ["user", "customer", "client", "passenger", "driver"]
    }
    
    identified = []
    for col in columns[:8]:  # Solo primeras 8 columnas
        col_lower = col.lower()
        for category, patterns in key_patterns.items():
            if any(pattern in col_lower for pattern in patterns):
                identified.append(f"{col} ({category})")
                break
        else:
            identified.append(col)
    
    return identified[:5]  # Máximo 5 columnas clave

def generate_dataset_description(table_name, columns):
    """Genera una descripción inteligente del dataset"""
    descriptions = {
        "dataset_rides": "Contiene información de viajes y reservas de transporte, incluyendo fechas, ubicaciones, costos y métodos de pago",
        "crocodile_dataset": "Dataset biológico con información sobre cocodrilos, posiblemente incluyendo medidas, ubicaciones y características",
        "ncr_ride_bookings": "Sistema de reservas de viajes con detalles de pasajeros, rutas, precios y estados de booking"
    }
    
    if table_name in descriptions:
        return descriptions[table_name]
    
    # Generar descripción automática basada en columnas
    col_hints = []
    columns_lower = [c.lower() for c in columns]
    
    if any("date" in c or "time" in c for c in columns_lower):
        col_hints.append("información temporal")
    if any("price" in c or "cost" in c or "amount" in c for c in columns_lower):
        col_hints.append("datos financieros")
    if any("location" in c or "city" in c for c in columns_lower):
        col_hints.append("datos geográficos")
    if any("user" in c or "customer" in c for c in columns_lower):
        col_hints.append("información de usuarios")
    
    if col_hints:
        return f"Dataset que incluye {', '.join(col_hints)}"
    else:
        return f"Dataset con {len(columns)} columnas de datos"

def generate_dataset_keywords(table_name, columns):
    """Genera palabras clave para identificación automática"""
    keywords = [table_name.replace("_", " ")]
    
    # Agregar keywords basados en nombre
    if "ride" in table_name or "booking" in table_name:
        keywords.extend(["viajes", "transporte", "reservas", "rides", "bookings"])
    if "crocodile" in table_name:
        keywords.extend(["cocodrilos", "animales", "biología", "crocodiles"])
    
    # Agregar keywords basados en columnas
    col_keywords = []
    for col in columns[:10]:
        col_lower = col.lower()
        if "payment" in col_lower:
            col_keywords.extend(["pago", "payment"])
        if "vehicle" in col_lower:
            col_keywords.extend(["vehículo", "vehicle"])
        if "date" in col_lower:
            col_keywords.extend(["fecha", "date"])
        if "location" in col_lower or "city" in col_lower:
            col_keywords.extend(["ubicación", "location"])
    
    keywords.extend(list(set(col_keywords)))
    return keywords

def identify_dataset_from_query(query: str, available_datasets: dict) -> str:
    """
    Identifica qué dataset es más relevante basándose en la consulta del usuario.
    Retorna el nombre de la tabla más apropiada.
    """
    query_lower = query.lower()
    
    # Búsqueda por referencias directas
    for table_name, info in available_datasets.items():
        # Buscar por nombre amigable
        friendly_name = info["friendly_name"].lower()
        if friendly_name in query_lower:
            return table_name
        
        # Buscar por keywords
        for keyword in info["keywords"]:
            if keyword.lower() in query_lower:
                return table_name
    
    # Búsqueda por patrones específicos
    if any(word in query_lower for word in ["viaje", "ride", "booking", "reserva", "transporte"]):
        for table_name in available_datasets:
            if "ride" in table_name or "booking" in table_name:
                return table_name
    
    if any(word in query_lower for word in ["cocodril", "animal", "biolog"]):
        for table_name in available_datasets:
            if "crocodile" in table_name:
                return table_name
    
    # Búsqueda por números (archivo 1, dataset 1, etc.)
    if "archivo 1" in query_lower or "dataset 1" in query_lower or "primer" in query_lower:
        dataset_names = list(available_datasets.keys())
        if dataset_names:
            return dataset_names[0]  # Primer dataset
    
    if "archivo 2" in query_lower or "dataset 2" in query_lower or "segundo" in query_lower:
        dataset_names = list(available_datasets.keys())
        if len(dataset_names) > 1:
            return dataset_names[1]  # Segundo dataset
    
    # Si no se encuentra coincidencia, retornar el primero disponible
    if available_datasets:
        return list(available_datasets.keys())[0]
    
    return None

def load_specific_dataset(table_name: str, connection=None):
    """
    Carga un dataset específico en memoria.
    Retorna (df, dataset_info, success)
    """
    global df, dataset_info, dataset_loaded  # MOVER ESTO AL INICIO
    
    # Buscar configuración del dataset
    dataset_config = None
    for config in DATASETS_TO_PROCESS:
        if config["table_name"] == table_name:
            dataset_config = config
            break
    
    if not dataset_config:
        print(f"❌ Configuración no encontrada para dataset: {table_name}")
        return None, None, False
    
    conn = connection
    
    # Intentar cargar desde BD primero
    if check_dataset_table_exists(conn, table_name, dataset_config["table_schema"]):
        print(f"🔄 Cargando {table_name} desde PostgreSQL...")
        try:
            with conn.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {dataset_config['table_schema']}.{table_name}")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(rows, columns=columns)
                
                dataset_info = get_dataset_table_info_by_name(table_name, conn)
                dataset_loaded = True
                print(f"✅ Dataset {table_name} cargado desde BD: {df.shape}")
                return df, dataset_info, True
        except Exception as e:
            print(f"❌ Error cargando desde BD: {e}")
    
    # Fallback: cargar desde Excel
    if os.path.exists(dataset_config["excel_path"]):
        print(f"📂 Cargando {table_name} desde Excel...")
        try:
            df = pd.read_excel(dataset_config["excel_path"])
            dataset_info = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "table_name": table_name
            }
            dataset_loaded = True
            print(f"✅ Dataset {table_name} cargado desde Excel: {df.shape}")
            return df, dataset_info, True
        except Exception as e:
            print(f"❌ Error cargando desde Excel: {e}")
    
    print(f"❌ No se pudo cargar dataset: {table_name}")
    return None, None, False

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

# Configurar sistema de conexión de datos
setup_data_connection()

# NUEVO: Inicializar dataset en BD si ENABLE_AUTO_SAVE_TO_DB está activado
print("🔄 Inicializando sistema de dataset...")
if not initialize_dataset_on_startup():
    print("⚠️ Advertencia: Error en inicialización del dataset, continuando con funcionalidad limitada")

print("🔄 Sistema preparado para carga de dataset bajo demanda...")

# Variables globales para el dataset
dataset_info = None
df = None
dataset_loaded = False

def ensure_dataset_loaded(state=None):
    """
    Función para cargar el dataset solo cuando sea necesario.
    Ahora respeta el dataset seleccionado en el estado.
    """
    global dataset_info, df, dataset_loaded
    
    # Determinar qué dataset cargar
    if state and state.get("selected_dataset"):
        target_dataset = state["selected_dataset"]
        print(f"🎯 Cargando dataset seleccionado: {target_dataset}")
    else:
        print("❌ No se especificó dataset y no hay fallback por defecto")
        return False
    
    # Verificar si ya está cargado el dataset correcto
    if dataset_loaded and df is not None and dataset_info:
        current_dataset = dataset_info.get("table_name", "")
        if current_dataset == target_dataset:
            print("✅ Dataset correcto ya está cargado en memoria")
            return True
        else:
            print(f"🔄 Dataset actual ({current_dataset}) no coincide, recargando...")
            dataset_loaded = False
    
    print(f"🔄 Cargando dataset: {target_dataset}")
    
    # Buscar configuración del dataset objetivo
    dataset_config = None
    for config in DATASETS_TO_PROCESS:
        if config["table_name"] == target_dataset:
            dataset_config = config
            break
    
    if not dataset_config:
        print(f"❌ Configuración no encontrada para: {target_dataset}")
        return False
    
    # Crear conexión temporal
    dataset_connection = None
    try:
        db_config = load_db_config()
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        dataset_connection = psycopg.connect(connection_string)
        print("🔗 Conexión temporal creada para dataset")
    except Exception as e:
        print(f"⚠️ Error creando conexión: {e}")
        dataset_connection = None
    
    # Intentar cargar desde BD primero
    if dataset_connection and check_dataset_table_exists(
        dataset_connection, 
        dataset_config["table_name"], 
        dataset_config["table_schema"]
    ):
        print(f"🔄 Cargando {target_dataset} desde PostgreSQL...")
        try:
            with dataset_connection.cursor() as cursor:
                cursor.execute(f"SELECT * FROM {dataset_config['table_schema']}.{dataset_config['table_name']}")
                rows = cursor.fetchall()
                columns = [desc[0] for desc in cursor.description]
                df = pd.DataFrame(rows, columns=columns)
                
                dataset_info = {
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "row_count": len(df),
                    "table_name": target_dataset
                }
                dataset_loaded = True
                print(f"✅ Dataset {target_dataset} cargado desde BD: {df.shape}")
                
                if dataset_connection:
                    dataset_connection.close()
                return True
        except Exception as e:
            print(f"❌ Error cargando desde BD: {e}")
    else:
        print(f"🔍 Tabla '{target_dataset}' no existe en BD")
    
    # Fallback: cargar desde Excel
    if os.path.exists(dataset_config["excel_path"]):
        print(f"📂 Cargando archivo Excel: {dataset_config['excel_path']}")
        try:
            df = pd.read_excel(dataset_config["excel_path"])
            dataset_info = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "table_name": target_dataset
            }
            dataset_loaded = True
            print(f"📊 Excel cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            if dataset_connection:
                dataset_connection.close()
            return True
        except Exception as e:
            print(f"❌ Error cargando Excel: {e}")
    else:
        print(f"❌ Archivo Excel no encontrado: {dataset_config['excel_path']}")
    
    if dataset_connection:
        dataset_connection.close()
    
    print(f"❌ No se pudo cargar el dataset: {target_dataset}")
    return False

# Configurar API y directorios
api_key = os.getenv("GOOGLE_API_KEY")
os.makedirs("./outputs", exist_ok=True)

# Inicializar LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key, temperature=0)
# llm = ChatOllama(model="gemma3", temperature=0)

# ======================
# 2. Definir intérprete Python con acceso a df
# ======================
python_repl = PythonREPLTool()

def run_python_with_df(code: str, error_context: Optional[str] = None):
    """
    Ejecuta código Python con acceso al DataFrame `df` ya cargado.
    IMPORTANTE: Asume que el dataset correcto ya fue cargado por node_ejecutar_python.
    """
    # Verificar que hay un dataset cargado (NO recargar)
    if df is None or not dataset_loaded:
        return {
            "success": False,
            "result": None,
            "error": "No hay dataset cargado en memoria",
            "error_type": "dataset_not_loaded"
        }

    # NO recargar dataset aquí - debe estar ya cargado por node_ejecutar_python
    # Solo verificar que existe
    
    # Contexto completo de ejecución con todos los módulos necesarios
    local_vars = {
        "df": df, 
        "pd": pd, 
        "plt": plt, 
        "sns": sns, 
        "os": os,
        "np": pd.np if hasattr(pd, 'np') else None  # numpy si está disponible
    }
    
    # También agregar a globals para funciones definidas en el código
    global_vars = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "sns": sns,
        "os": os
    }

    prohibited_patterns = [
        "pd.DataFrame(",
        "pandas.DataFrame(",
        "DataFrame(",
        "= pd.read_csv",
        "= pd.read_excel",
        "# Datos de ejemplo",
        "datos de ejemplo",
        "reemplaza con tu DataFrame"
    ]
    
    # Validar patrones prohibidos de forma más inteligente
    for pattern in prohibited_patterns:
        # Usar regex para detectar creación real de DataFrames
        if pattern in ["pd.DataFrame(", "pandas.DataFrame(", "DataFrame("]:
            if re.search(r'(pd\.|pandas\.)?DataFrame\s*\(', code):
                return {
                    "success": False,
                    "result": None,
                    "error": f"Código bloqueado. Detectado intento de crear DataFrame. Usa SOLO el df existente.",
                    "error_type": "prohibited_pattern"
                }
        elif pattern.lower() in code.lower():
            return {
                "success": False,
                "result": None,
                "error": f"Código bloqueado. Detectado patrón prohibido: '{pattern}'",
                "error_type": "prohibited_pattern"
            }

    try:
        import ast
        parsed = ast.parse(code)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            last_expr = parsed.body.pop()
            exec(compile(ast.Module(parsed.body, type_ignores=[]), filename="<ast>", mode="exec"), global_vars, local_vars)
            result = eval(compile(ast.Expression(last_expr.value), filename="<ast>", mode="eval"), global_vars, local_vars)
        else:
            exec(code, global_vars, local_vars)
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

    # Verificar que hay dataset cargado
    if df is None or not dataset_loaded:
        return "Error: No hay dataset cargado. Use ensure_dataset_loaded primero."

    # Verificar que hay dataset cargado (NO recargar)
    if df is None or not dataset_loaded:
        return "Error: No hay dataset cargado en memoria"
    
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

# def plot_booking_value_by_vehicle_type(_):
#     """Genera un boxplot de Booking Value según Vehicle Type."""
#     if "Booking Value" not in df.columns or "Vehicle Type" not in df.columns:
#         return "No existen las columnas necesarias (Booking Value, Vehicle Type)."
    
#     plt.figure(figsize=(12,6))
#     import seaborn as sns
#     sns.boxplot(data=df, x="Vehicle Type", y="Booking Value", palette="Set2")
#     plt.title("Distribución de Booking Value por tipo de vehículo", fontsize=16)
#     plt.xlabel("Tipo de Vehículo", fontsize=12)
#     plt.ylabel("Booking Value", fontsize=12)
#     plt.xticks(rotation=30, ha="right")
#     plt.grid(axis="y", alpha=0.5)

#     file_path = "./outputs/booking_value_by_vehicle_type.png"
#     plt.savefig(file_path, dpi=300, bbox_inches="tight")
#     plt.show()
#     return f"✅ Boxplot generado y guardado en {file_path}"

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
    # Tool(name="plot_booking_value_by_vehicle_type", func=plot_booking_value_by_vehicle_type, description="Genera un boxplot de Booking Value por tipo de vehículo"),
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
5. NO generes comentarios, Type hints o anotaciones en el código. Responde SOLO con código Python ejecutable, sin explicaciones ni markdown

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
        
    base_prompt += """

REGLAS CRÍTICAS DE FORMATO:
1. NO definas funciones - escribe código directo
2. NO uses if __name__ == '__main__'
3. NO incluyas comentarios sobre cargar datos
4. El DataFrame 'df' YA ESTÁ DISPONIBLE
5. Todos los módulos (pd, plt, sns, os) YA ESTÁN IMPORTADOS

CÓDIGO DEBE SER EJECUTABLE DIRECTAMENTE, ejemplo:
✅ CORRECTO:
plt.figure(figsize=(10,6))
df['columna'].hist()
plt.show()

❌ INCORRECTO:
def plot_histogram(df):
    plt.hist(df['columna'])
plot_histogram(df)
"""
    
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
    # NUEVOS campos para gestión múltiple de datasets
    available_datasets: dict         # Metadatos de todos los datasets disponibles
    selected_dataset: str           # Dataset seleccionado para la consulta actual
    active_dataframe: str           # DataFrame actualmente cargado en memoria
    dataset_context: dict           # Contexto específico del dataset seleccionado
    # NUEVOS campos para que el LLM use consultas SQL
    data_strategy: str               # "sql" o "dataframe"
    sql_feasible: bool              # Si SQL puede resolver la consulta
    table_metadata: dict            # Metadatos cacheados sin cargar datos
    strategy_history: List[str]     # Historial de cambios de estrategia
    sql_results: Any                # Resultados de consultas SQL
    strategy_switched: bool         # Si se cambió de estrategia (para logs)
    needs_fallback: bool           # Si necesita fallback a DataFrame
    strategy_reason: str           # Razón de la estrategia elegida
    sql_error: Optional[str]       # Error específico de SQL si ocurre
    # NUEVOS CAMPOS PARA MEMORIA PERSISTENTE
    conversation_history: List[dict]      # Historial de conversaciones completas
    user_context: dict                   # Contexto del usuario (preferencias, patrones)
    memory_summary: str                  # Resumen de conversaciones previas
    learned_patterns: List[str]          # Patrones aprendidos del usuario
    session_metadata: dict               # Metadatos de la sesión actual

# ======================
# Función para gestionar historial conversacional
# ======================

# ======================
# 5. Nodos del grafo
# ======================
def nodo_estrategia_datos(state: AgentState):
    """
    MODIFICADO: Ahora recupera historial desde PostgresSaver y actualiza contexto
    """
    print("🧠 Iniciando análisis con recuperación de memoria...")
    
    # PASO 6A: Recuperar historial desde PostgresSaver - CORREGIDO
    if not state.get("conversation_history"):
        # Intentar cargar desde PostgresSaver usando el thread_id
        thread_id = state.get("session_metadata", {}).get("thread_id", SINGLE_USER_THREAD_ID)
        conversation_history, user_context = load_conversation_history(thread_id)
        
        state["conversation_history"] = conversation_history
        state["user_context"] = user_context if user_context else {
            "preferred_analysis_type": None,
            "common_datasets": [],
            "visualization_preferences": [],
            "error_patterns": [],
            "last_interaction": None
        }
    
    # Generar resumen de conversaciones previas para contexto
    if state["conversation_history"]:
        memory_summary = generate_memory_summary(state["conversation_history"])
        state["memory_summary"] = memory_summary
        print(f"💭 Memoria recuperada: {len(state['conversation_history'])} conversaciones previas")
        print(f"📝 Resumen: {memory_summary[:100]}...")
        
        # Cargar patrones aprendidos desde el historial
        if not state.get("learned_patterns"):
            state["learned_patterns"] = extract_learned_patterns_from_history(state["conversation_history"])
    else:
        state["memory_summary"] = "Primera conversación con el usuario"
        print("🆕 Primera interacción - sin historial previo")
    
    # NUEVO: Detectar consultas sobre memoria ANTES de analizar estrategia
    if is_memory_query(state["query"]):
        print("🧠 Consulta sobre memoria detectada - respuesta directa")
        state["data_strategy"] = "memory"
        state["strategy_reason"] = "Consulta sobre historial de conversaciones - respuesta directa desde memoria cargada"
        state["result"] = generate_memory_response(state)
        state["success"] = True
        state["history"].append(f"Memoria → Respuesta directa sobre historial")
        return state
    
    # Resto del código original para consultas normales...
    print("🔍 Analizando estrategia de acceso a datos...")
    
    if not state.get("available_datasets"):
        state["available_datasets"] = get_all_available_datasets()
    
    if not state.get("selected_dataset"):
        # MEJORADO: Usar contexto histórico para selección de dataset
        selected_dataset = identify_dataset_from_query_with_memory(
            state["query"], 
            state["available_datasets"],
            state["user_context"]
        )
        if selected_dataset:
            state["selected_dataset"] = selected_dataset
            state["dataset_context"] = state["available_datasets"][selected_dataset]
    
    # Obtener metadatos y analizar estrategia (código original)
    table_metadata = get_table_metadata_light(state["selected_dataset"])
    state["table_metadata"] = table_metadata
    
    # Analizar consulta con contexto histórico
    strategy_prompt = f"""
Analiza esta consulta considerando el historial del usuario:

CONSULTA ACTUAL: {state['query']}
MEMORIA DEL USUARIO: {state['memory_summary']}
CONTEXTO HISTÓRICO: {state['user_context']}

METADATOS DE TABLA DISPONIBLE:
- Tabla: {state['selected_dataset']}
- Columnas: {table_metadata.get('columns', [])[:10]}
- Filas estimadas: {table_metadata.get('row_count', 'N/A')}

PATRONES APRENDIDOS:
{', '.join(state.get('learned_patterns', []))}

CRITERIOS PARA SQL:
- Consultas de conteo simple
- Filtros básicos
- Agregaciones simples
- Consultas similares a las exitosas anteriormente

CRITERIOS PARA DATAFRAME:
- Análisis estadísticos complejos
- Visualizaciones (considerando preferencias previas)
- Análisis avanzados
- Si el usuario ha tenido problemas con SQL antes

Responde:
Strategy: sql|dataframe
Reason: <explicación considerando el historial>
SQL_Feasible: true|false
"""
    
    response = llm.invoke(strategy_prompt).content.strip()
    
    # Extraer decisión (código original)
    strategy = "dataframe"
    sql_feasible = False
    reason = ""
    
    for line in response.splitlines():
        if line.lower().startswith("strategy:"):
            strategy = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("sql_feasible:"):
            sql_feasible = "true" in line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
    
    state["data_strategy"] = strategy
    state["sql_feasible"] = sql_feasible
    state["strategy_reason"] = reason
    
    print(f"📊 Estrategia seleccionada: {strategy.upper()}")
    print(f"🔍 Razón (con memoria): {reason}")
    
    state["history"].append(f"Estrategia → {strategy.upper()} - {reason}")
    
    return state

def node_clasificar_modificado(state: AgentState):
    """MODIFICADO: Se enfoca solo en dataset selection y tool selection"""
    
    # La estrategia ya fue definida por nodo_estrategia_datos
    data_strategy = state.get("data_strategy", "dataframe")
    selected_dataset = state.get("selected_dataset")
    
    print(f"🎯 Clasificando con estrategia: {data_strategy.upper()}")
    print(f"📊 Dataset seleccionado: {state.get('dataset_context', {}).get('friendly_name', 'N/A')}")
    
    # Seleccionar herramientas según estrategia
    if data_strategy == "sql":
        tools_context = """
HERRAMIENTAS DISPONIBLES (Modo SQL):
- SQL_Executor: Ejecuta consultas SQL directas en la base de datos
- Herramientas básicas de metadatos si SQL no es suficiente
"""
        recommended_action = "SQL_Executor"
    else:
        tools_context = f"""
HERRAMIENTAS DISPONIBLES (Modo DataFrame):
{get_tools_summary(tools)}
"""
        recommended_action = "Python_Interpreter"
    
    prompt = f"""
Analiza esta consulta para seleccionar la herramienta más apropiada:

CONSULTA: {state['query']}
ESTRATEGIA DEFINIDA: {data_strategy.upper()}
DATASET: {state.get('dataset_context', {}).get('friendly_name', 'N/A')}

{tools_context}

INSTRUCCIONES:
- La estrategia de datos ya fue decidida por el nodo anterior
- Selecciona la herramienta MÁS específica para esta consulta
- Si la estrategia es SQL, prioriza SQL_Executor salvo que sea inadecuado
- Si la estrategia es DataFrame, usa las herramientas especializadas o Python_Interpreter

Responde:
Thought: <análisis de la consulta y selección de herramienta>
Action: <nombre exacto de la herramienta>
"""

    response = llm.invoke(prompt).content.strip()
    
    # Extraer decisiones
    thought, action = "", recommended_action
    
    for line in response.splitlines():
        if line.lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()
    
    state["thought"] = thought
    state["action"] = action
    
    print(f"🧠 Thought: {thought}")
    print(f"➡️ Action: {action}")
    
    state["history"].append(f"Clasificar (Mod) → {action} - {thought[:100]}")
    
    return state

def nodo_sql_executor(state: AgentState):
    """NUEVO: Ejecuta consultas SQL directamente en la base de datos"""
    
    print("🗃️ Ejecutando consulta SQL...")
    
    # Obtener conexión
    conn = data_connection  # Usar la conexión global de datos
    if conn is None:
        # Fallback: crear conexión temporal
        try:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            temp_connection = True
        except Exception as e:
            print(f"❌ Error creando conexión SQL: {e}")
            state["sql_error"] = str(e)
            state["success"] = False
            return state
    else:
        temp_connection = False
    
    # Generar consulta SQL
    sql_prompt = f"""
Genera una consulta SQL para resolver esta petición:

CONSULTA: {state['query']}

INFORMACIÓN DE TABLA:
- Tabla: {state['selected_dataset']}
- Esquema: public
- Columnas disponibles: {state.get('table_metadata', {}).get('columns', [])}

REGLAS:
1. Usa SOLO la tabla: public.{state['selected_dataset']}
2. Usa comillas dobles para nombres de columnas si tienen espacios
3. Limita resultados a máximo 100 filas si no se especifica
4. Para agregaciones, usa funciones SQL estándar (COUNT, SUM, AVG, etc.)
5. Si hay fechas, asume formato TIMESTAMP
6. NO uses funciones específicas de PostgreSQL complejas

EJEMPLOS:
- Conteo: SELECT COUNT(*) FROM public.{state['selected_dataset']}
- Top 10: SELECT * FROM public.{state['selected_dataset']} LIMIT 10
- Agregación: SELECT "Payment Method", COUNT(*) FROM public.{state['selected_dataset']} GROUP BY "Payment Method"

Responde SOLO con la consulta SQL, sin explicaciones:
"""
    
    try:
        sql_query = llm.invoke(sql_prompt).content.strip()
        
        # Limpiar la consulta
        if sql_query.startswith("```"):
            sql_query = sql_query.strip("`").replace("sql", "").strip()
        
        print(f"🔍 SQL generado:\n{sql_query}")
        
        # Ejecutar consulta
        with conn.cursor() as cursor:
            cursor.execute(sql_query)
            
            # Inicializar variables
            columns = []
            rows = []
            result_df = None
            has_results = cursor.description is not None
            
            # Obtener resultados
            if has_results:  # Si hay resultados
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                # Convertir a DataFrame para compatibilidad
                if rows:
                    result_df = pd.DataFrame(rows, columns=columns)
                    if result_df is not None and not result_df.empty:
                        # Convertir DataFrame a formato serializable
                        state["sql_results"] = {
                            "data": result_df.to_dict('records'),
                            "columns": result_df.columns.tolist(),
                            "shape": result_df.shape
                        }
                    else:
                        state["sql_results"] = None
                    state["result"] = f"Consulta SQL ejecutada exitosamente. {len(result_df)} filas obtenidas."
                else:
                    state["sql_results"] = pd.DataFrame()
                    state["result"] = "Consulta SQL ejecutada exitosamente. Sin resultados."
            else:
                # Consulta sin resultados (INSERT, UPDATE, etc.)
                rowcount = cursor.rowcount
                state["result"] = f"Consulta SQL ejecutada. Filas afectadas: {rowcount}"

            state["success"] = True
            print(f"✅ SQL ejecutado exitosamente")

            # Mostrar resultados en consola
            if has_results:  # Si hay resultados
                if rows:
                    print(f"\n📊 RESULTADOS DE LA CONSULTA:")
                    print(f"   Filas obtenidas: {len(rows)}")
                    print(f"   Columnas: {len(columns)}")
                    print(f"\n{result_df.to_string(max_rows=20, max_cols=10)}")
                    
                    if len(rows) > 20:
                        print(f"\n... (mostrando primeras 20 de {len(rows)} filas)")
                else:
                    print(f"\n📊 CONSULTA EJECUTADA - Sin resultados")
            else:
                print(f"\n📊 CONSULTA EJECUTADA - Filas afectadas: {rowcount}")
        
    except Exception as e:
        print(f"❌ Error ejecutando SQL: {e}")
        state["sql_error"] = str(e)
        state["success"] = False
        state["result"] = None
        
        # Marcar para fallback a DataFrame
        state["needs_fallback"] = True
        
    finally:
        if temp_connection and conn:
            conn.close()
    
    state["history"].append(f"SQL Executor → {'Éxito' if state.get('success', False) else 'Error'}")
    return state

def node_ejecutar_python(state: AgentState):
    """Ejecuta código Python con manejo robusto de errores y contexto"""
    
    print(f"⚙️ Ejecutando Python - Intento {state['iteration_count'] + 1}")
    
    # Asegurar que el dataset esté cargado
    if not ensure_dataset_loaded(state):
        state["success"] = False
        state["result"] = "Error: No se pudo cargar el dataset"
        return state
    
    # Verificar y construir df_info si es necesario
    if not state.get("df_info") or 'columns' not in state["df_info"]:
        sample_clean = df.head(2).fillna("NULL").to_dict()
        state["df_info"] = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "sample": sample_clean
        }
    
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

def node_validar_y_decidir_modificado(state: AgentState):
    """MODIFICADO: Maneja fallbacks entre SQL y DataFrame"""
    
    state["iteration_count"] += 1
    success = state.get("success", False)
    max_iterations = state.get("max_iterations", 3)
    needs_fallback = state.get("needs_fallback", False)
    current_strategy = state.get("data_strategy", "dataframe")
    
    print(f"\n🔍 Validación - Iteración {state['iteration_count']}")
    print(f"   Éxito: {success}")
    print(f"   Estrategia actual: {current_strategy.upper()}")
    print(f"   Necesita fallback: {needs_fallback}")
    
    # Decidir próxima acción
    if success:
        state["next_node"] = "responder"
        print("   ➡️ Decisión: Proceder a responder (éxito)")
        
    elif needs_fallback and current_strategy == "sql":
        # Cambiar estrategia de SQL a DataFrame
        state["data_strategy"] = "dataframe"
        state["needs_fallback"] = False
        state["strategy_switched"] = True
        state["next_node"] = "clasificar"
        print("   ➡️ Decisión: Fallback a DataFrame")
        
    elif state["iteration_count"] >= max_iterations:
        state["next_node"] = "responder"
        print("   ➡️ Decisión: Proceder a responder (máximo iteraciones)")
        
    else:
        state["next_node"] = "clasificar"
        print("   ➡️ Decisión: Nueva iteración")
    
    # Actualizar historial con información de fallback
    fallback_info = " (con fallback)" if needs_fallback else ""
    state["history"].append(f"Validar (Mod) → Iter {state['iteration_count']}, {current_strategy.upper()}{fallback_info}, Siguiente: {state['next_node']}")
    
    return state

def node_responder(state: AgentState):
    """
    MODIFICADO: Genera respuestas interpretativas con datos específicos obtenidos
    """
    success = state.get("success", False)
    
    if success:
        # Obtener información de la última ejecución exitosa
        last_execution = None
        if state["execution_history"]:
            last_execution = state["execution_history"][-1]
        
        # Verificar si es una visualización
        is_visualization = False
        is_data_query = False
        code_executed = ""
        
        if last_execution and last_execution["success"]:
            code_executed = last_execution.get("code", "")
            
            # Detectar visualizaciones
            is_visualization = any(keyword in code_executed.lower() for keyword in [
                "plt.", "plot", "hist", "scatter", "bar", "show()", "savefig"
            ])
            
            # Detectar consultas de datos (análisis, conteos, consultas SQL, etc.)
            is_data_query = any(keyword in code_executed.lower() for keyword in [
                "count", "sum", "mean", "describe", "value_counts", "groupby", "agg",
                "select", "where", "group by", "order by", "len(", "shape", "info()",
                "nunique", "unique", "max", "min", "std", "var"
            ]) or state.get("sql_results") is not None
        
        if is_visualization:
            # Para visualizaciones: comentar el resultado, NO mostrar código
            prompt = f"""
La consulta del usuario fue: {state['query']}

Se ejecutó exitosamente código de visualización que generó un gráfico.

CÓDIGO EJECUTADO (PARA CONTEXTO INTERNO - NO MOSTRAR AL USUARIO):
{code_executed}

RESULTADO OBTENIDO: {state['result']}

Tu tarea es generar un comentario breve e interpretativo sobre lo que muestra el gráfico generado, SIN incluir código ni explicaciones técnicas.

Enfócate en:
1. Qué tipo de visualización se generó
2. Qué información muestra al usuario
3. Insights breves sobre los datos visualizados (si es posible inferirlos)
4. Confirmar dónde se guardó el archivo

NO incluyas código Python, explicaciones técnicas ni instrucciones.
"""
        
        elif is_data_query or state.get("sql_results"):
            # Para consultas que obtuvieron datos específicos
            datos_obtenidos = ""
            
            # Extraer datos de resultados SQL si existen
            if state.get("sql_results"):
                sql_data = state["sql_results"]
                if isinstance(sql_data, dict) and "data" in sql_data:
                    datos_obtenidos = f"Datos SQL: {sql_data['data'][:3]}..."  # Primeros 3 registros
                else:
                    datos_obtenidos = f"Resultados SQL: {str(sql_data)[:200]}..."
            
            # O extraer del resultado de código Python
            elif last_execution and last_execution.get("result"):
                result_data = last_execution["result"]
                if isinstance(result_data, str) and len(result_data) > 10:
                    datos_obtenidos = result_data
                else:
                    datos_obtenidos = str(result_data)
            
            prompt = f"""
La consulta del usuario fue: {state['query']}

Se ejecutó exitosamente un análisis de datos que obtuvo información específica.

DATOS OBTENIDOS:
{datos_obtenidos}

CÓDIGO EJECUTADO (PARA CONTEXTO INTERNO - NO MOSTRAR AL USUARIO):
{code_executed}

Tu tarea es generar una respuesta que:
1. Confirme qué análisis se realizó
2. INCLUYA los datos específicos obtenidos en la respuesta
3. Interprete brevemente qué significan esos datos
4. Sea clara y directa

IMPORTANTE:
- SÍ incluye los números, conteos, o datos específicos obtenidos
- NO incluyas código Python
- NO expliques cómo funciona el código
- Enfócate en el resultado y su interpretación

Ejemplo: "He analizado los datos y encontré que hay 1,247 registros en total, de los cuales 623 corresponden a la categoría X y 624 a la categoría Y, mostrando una distribución equilibrada."
"""
        
        else:
            # Para otros análisis: respuesta normal mejorada
            prompt = f"""
Pregunta del usuario: {state['query']}
Resultado obtenido: {state['result']}
Iteraciones necesarias: {state['iteration_count']}
Contexto histórico: {state.get('memory_summary', 'N/A')}

Genera una respuesta clara sobre el análisis realizado, incluyendo cualquier dato específico que se haya obtenido.
"""
    else:
        # Manejo de errores (código original)
        errors_summary = []
        for record in state["execution_history"]:
            if not record["success"]:
                errors_summary.append(f"- {record['error_type']}: {record['error']}")
        
        prompt = f"""
Pregunta del usuario: {state['query']}
Después de {state['iteration_count']} iteraciones, no se pudo completar la tarea.
Contexto histórico: {state.get('memory_summary', 'N/A')}

Errores encontrados:
{chr(10).join(errors_summary)}

Genera una respuesta empática explicando los problemas encontrados y sugerencias.
"""

    respuesta = llm.invoke(prompt).content
    print(f"\n🤖 Respuesta Final:\n{respuesta}")
    
    # Resto del código original para actualizar memoria
    conversation_record = {
        "timestamp": datetime.now().isoformat(),
        "query": state["query"],
        "success": success,
        "strategy_used": state.get("data_strategy", "unknown"),
        "dataset_used": state.get("selected_dataset", "unknown"),
        "iterations": state["iteration_count"],
        "errors": [record for record in state["execution_history"] if not record["success"]],
        "response": respuesta
    }
    
    if not state.get("conversation_history"):
        state["conversation_history"] = []
    state["conversation_history"].append(conversation_record)
    
    update_user_context(state, conversation_record)
    update_learned_patterns(state, conversation_record)
    
    print("💾 Memoria actualizada con nueva conversación")

    cleaned_state = clean_state_for_serialization(state)
    
    for key, value in cleaned_state.items():
        state[key] = value
    
    state["history"].append(f"Responder → Finalizado con {'éxito' if success else 'error'} + memoria actualizada")
    
    return state

def route_after_classification(state: AgentState):
    """Determina si ir a SQL_Executor, Python_Interpreter o responder directamente"""
    action = state.get("action", "Python_Interpreter")
    data_strategy = state.get("data_strategy", "dataframe")
    
    print(f"\n🔧 Routing después de clasificación:")
    print(f"   Action: {action}")
    print(f"   Strategy: {data_strategy}")
    
    # NUEVO: Manejar consultas de memoria
    if data_strategy == "memory":
        print("   → Routing to: responder (memoria)")
        return "responder"
    
    # Mapeo de acciones a nodos para consultas normales
    if action == "SQL_Executor" or (data_strategy == "sql" and action not in ["Python_Interpreter"]):
        print("   → Routing to: sql_executor")
        return "sql_executor"
    else:
        print("   → Routing to: ejecutar_python")
        return "ejecutar_python"

def route_after_validation_modificado(state: AgentState):
    """Routing modificado que maneja fallbacks"""
    success = state.get("success", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    needs_fallback = state.get("needs_fallback", False)
    current_strategy = state.get("data_strategy", "dataframe")
    
    print(f"\n🔧 DEBUG route_after_validation_modificado:")
    print(f"   Success: {success}")
    print(f"   Iteration: {iteration_count}")
    print(f"   Strategy: {current_strategy}")
    print(f"   Needs fallback: {needs_fallback}")
    
    if success:
        print("   → Routing to: responder (success)")
        return "responder"
    elif needs_fallback and current_strategy == "sql":
        print("   → Routing to: estrategia_datos (fallback)")
        return "estrategia_datos"  # Volver a evaluar estrategia
    elif iteration_count >= max_iterations:
        print("   → Routing to: responder (max iterations)")
        return "responder"
    else:
        print("   → Routing to: clasificar (continue)")
        return "clasificar"

def get_table_metadata_light(table_name: str):
    """Obtiene metadatos básicos de una tabla sin cargar datos"""
    
    conn = data_connection  # Usar la conexión global de datos
    if conn is None:
        try:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            temp_connection = True
        except:
            return {}
    else:
        temp_connection = False
    
    try:
        with conn.cursor() as cursor:
            # Información de columnas
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns_info = cursor.fetchall()
            
            # Conteo de filas
            cursor.execute(f'SELECT COUNT(*) FROM public."{table_name}"')
            row_count = cursor.fetchone()[0]
            
            return {
                "columns": [col[0] for col in columns_info],
                "dtypes": {col[0]: col[1] for col in columns_info},
                "row_count": row_count,
                "nullable": {col[0]: col[2] == 'YES' for col in columns_info}
            }
            
    except Exception as e:
        print(f"⚠️ Error obteniendo metadatos de {table_name}: {e}")
        return {}
    finally:
        if temp_connection and conn:
            conn.close()

# ======================
# 6. Construir el grafo
# ======================

def create_graph_with_sql():
    """Crea el grafo con los nuevos nodos SQL y PostgresSaver"""
    global postgres_saver
    
    graph = StateGraph(AgentState)

    # Nodos existentes + nuevos
    graph.add_node("estrategia_datos", nodo_estrategia_datos)
    graph.add_node("clasificar", node_clasificar_modificado)
    graph.add_node("sql_executor", nodo_sql_executor)
    graph.add_node("ejecutar_python", node_ejecutar_python)
    graph.add_node("validar", node_validar_y_decidir_modificado)
    graph.add_node("responder", node_responder)

    # Punto de entrada
    graph.set_entry_point("estrategia_datos")

    # Flujo principal
    graph.add_edge("estrategia_datos", "clasificar")
    graph.add_conditional_edges("clasificar", route_after_classification)
    graph.add_edge("sql_executor", "validar")
    graph.add_edge("ejecutar_python", "validar")
    graph.add_conditional_edges("validar", route_after_validation_modificado)
    graph.add_edge("responder", END)

    # NUEVO: Compilar con checkpointer si está disponible
    if postgres_saver:
        print("🧠 Compilando grafo con memoria persistente (PostgresSaver)")
        return graph.compile(checkpointer=postgres_saver)
    else:
        print("⚠️ Compilando grafo sin memoria persistente")
        return graph.compile()

def generate_memory_summary(conversation_history: List[dict]) -> str:
    """
    Genera un resumen conciso de las conversaciones previas.
    """
    if not conversation_history:
        return "Sin historial previo"
    
    recent_conversations = conversation_history[-5:]  # Últimas 5 conversaciones
    
    summary_parts = []
    successful_queries = sum(1 for conv in recent_conversations if conv["success"])
    total_queries = len(recent_conversations)
    
    summary_parts.append(f"Últimas {total_queries} consultas: {successful_queries} exitosas")
    
    # Datasets más utilizados
    datasets_used = [conv["dataset_used"] for conv in recent_conversations if conv["dataset_used"] != "unknown"]
    if datasets_used:
        most_common_dataset = max(set(datasets_used), key=datasets_used.count)
        summary_parts.append(f"Dataset preferido: {most_common_dataset}")
    
    # Estrategias exitosas
    successful_strategies = [conv["strategy_used"] for conv in recent_conversations if conv["success"]]
    if successful_strategies:
        most_successful_strategy = max(set(successful_strategies), key=successful_strategies.count)
        summary_parts.append(f"Estrategia exitosa: {most_successful_strategy}")
    
    return "; ".join(summary_parts)

def identify_dataset_from_query_with_memory(query: str, available_datasets: dict, user_context: dict) -> str:
    """
    Versión mejorada que considera el historial del usuario.
    """
    # Usar la función original como base
    base_result = identify_dataset_from_query(query, available_datasets)
    
    # Considerar datasets comunes del usuario
    common_datasets = user_context.get("common_datasets", [])
    if common_datasets and base_result in common_datasets:
        print(f"✅ Dataset confirmado por historial: {base_result}")
        return base_result
    
    # Si hay ambigüedad, preferir el dataset más usado históricamente
    if not base_result and common_datasets:
        preferred = common_datasets[0]  # El más usado
        print(f"🔄 Usando dataset preferido por historial: {preferred}")
        return preferred
    
    return base_result

def is_memory_query(query: str) -> bool:
    """
    Detecta si la consulta es sobre memoria/historial de conversaciones.
    """
    memory_keywords = [
        "recuerda", "recuerdas", "memoria", "historial", "anteriormente", "antes",
        "pregunta anterior", "consulta anterior", "conversación anterior", 
        "hablamos", "dijiste", "respondiste", "pregunte", "pregunté", "charlamos",
        "intercambio", "diálogo", "sesión anterior", "que te dije", "que me dijiste"
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in memory_keywords)

def generate_memory_response(state: AgentState) -> str:
    """
    Genera una respuesta directa sobre la memoria/historial sin usar SQL o herramientas.
    """
    conversation_history = state.get("conversation_history", [])
    query = state["query"]
    
    if not conversation_history:
        return "No tengo memoria de conversaciones anteriores en esta sesión."
    
    # Crear resumen de conversaciones previas
    recent_conversations = conversation_history[-3:]  # Últimas 3 para ser más específico
    
    response_parts = [
        f"Sí, recuerdo nuestras {len(conversation_history)} conversaciones anteriores:"
    ]
    
    for i, conv in enumerate(recent_conversations, 1):
        query_text = conv.get("query", "N/A")[:80] + ("..." if len(conv.get("query", "")) > 80 else "")
        success_status = "exitosa" if conv.get("success", False) else "no exitosa"
        response_parts.append(f"{i}. Preguntaste: \"{query_text}\" - Consulta {success_status}")
    
    # Agregar información de contexto
    user_context = state.get("user_context", {})
    datasets_used = user_context.get("common_datasets", [])
    if datasets_used:
        response_parts.append(f"\nHas trabajado principalmente con: {', '.join(datasets_used)}")
    
    preferred_strategy = user_context.get("preferred_analysis_type")
    if preferred_strategy:
        response_parts.append(f"Tu estrategia de análisis preferida es: {preferred_strategy}")
    
    return "\n".join(response_parts)

def update_user_context(state: AgentState, conversation_record: dict):
    """
    Actualiza el contexto del usuario basado en la conversación actual.
    """
    user_context = state["user_context"]
    
    # Actualizar último timestamp de interacción
    user_context["last_interaction"] = conversation_record["timestamp"]
    
    # Actualizar datasets comunes
    dataset_used = conversation_record["dataset_used"]
    if dataset_used != "unknown":
        if dataset_used not in user_context["common_datasets"]:
            user_context["common_datasets"].append(dataset_used)
        else:
            # Mover al frente (más reciente)
            user_context["common_datasets"].remove(dataset_used)
            user_context["common_datasets"].insert(0, dataset_used)
    
    # Mantener solo los 3 más usados
    user_context["common_datasets"] = user_context["common_datasets"][:3]
    
    # Actualizar análisis preferido
    if conversation_record["success"]:
        strategy = conversation_record["strategy_used"]
        if not user_context["preferred_analysis_type"]:
            user_context["preferred_analysis_type"] = strategy
        elif user_context["preferred_analysis_type"] != strategy:
            # Alternar basado en éxito reciente
            user_context["preferred_analysis_type"] = strategy
    
    # Registrar patrones de error
    if conversation_record["errors"]:
        error_types = [error["error_type"] for error in conversation_record["errors"]]
        for error_type in error_types:
            if error_type not in user_context["error_patterns"]:
                user_context["error_patterns"].append(error_type)

def update_learned_patterns(state: AgentState, conversation_record: dict):
    """
    Actualiza los patrones aprendidos del comportamiento del usuario.
    """
    if not state.get("learned_patterns"):
        state["learned_patterns"] = []
    
    patterns = state["learned_patterns"]
    
    # Patrón de éxito
    if conversation_record["success"]:
        success_pattern = f"Exitoso: {conversation_record['strategy_used']} en {conversation_record['dataset_used']}"
        if success_pattern not in patterns:
            patterns.append(success_pattern)
    
    # Patrón de múltiples iteraciones
    if conversation_record["iterations"] > 1:
        iteration_pattern = f"Requiere {conversation_record['iterations']} iteraciones para consultas complejas"
        if iteration_pattern not in patterns and conversation_record["iterations"] <= 3:
            patterns.append(iteration_pattern)
    
    # Mantener solo los 5 patrones más recientes
    state["learned_patterns"] = patterns[-5:]

def main():
    global postgres_saver
    
    # Configurar PostgresSaver
    postgres_saver = setup_postgres_saver()
    
    app = create_graph_with_sql()
    
    print("✅ Base de datos PostgreSQL configurada correctamente")
    print(f"💾 Guardado automático a BD: {'ACTIVADO' if ENABLE_AUTO_SAVE_TO_DB else 'DESACTIVADO'}")
    print(f"🧠 Memoria conversacional: {'ACTIVADA' if postgres_saver else 'DESACTIVADA'}")
    
    # CORREGIDO: Thread ID automático para usuario único
    thread_id = get_automatic_thread_id()
    
    print("🚀 Sistema de Análisis de Datos con Memoria Persistente")
    print("   Memoria automática activada para usuario único")
    print("   Dataset se cargará al hacer la primera consulta")
    print("   Escribe 'salir' para terminar\n")
    
    # Mostrar archivos almacenados
    show_stored_files()

    # Mostrar memoria de conversación
    show_conversation_memory(thread_id)

    print()
    
    while True:
        query = input("Pregunta sobre el dataset (o 'salir'): ")
        
        if query.lower() == "salir":
            break

        # Estado inicial CON CAMPOS DE MEMORIA AUTOMÁTICA
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
            
            # CAMPOS DE MEMORIA AUTOMÁTICA - SE CARGARÁN EN nodo_estrategia_datos
            "conversation_history": [],  # Se cargará desde PostgresSaver
            "user_context": {},          # Se cargará desde PostgresSaver
            "memory_summary": "",        # Se generará desde historial cargado
            "learned_patterns": [],      # Se extraerán desde historial cargado
            "session_metadata": {
                "thread_id": thread_id,
                "session_start": datetime.now().isoformat(),
                "user_id": SINGLE_USER_ID
            }
        }

        print(f"\n{'='*60}")
        print(f"🔄 Procesando consulta: {query}")
        print(f"🧠 Thread automático: {thread_id}")
        print(f"{'='*60}")
        
        try:
            # CORREGIDO: Configurar thread automático para memoria persistente
            config = {"configurable": {"thread_id": thread_id}} if postgres_saver else {}
            
            # Invocar con configuración de thread
            final_state = app.invoke(initial_state, config=config)

            # Limpiar estado final para evitar problemas de serialización futuros
            final_state = clean_state_for_serialization(final_state)
            
            print(f"\n📊 RESUMEN DE EJECUCIÓN:")
            print(f"   Iteraciones totales: {final_state['iteration_count']}")
            print(f"   Éxito: {final_state.get('success', False)}")
            print(f"   Conversaciones en memoria: {len(final_state.get('conversation_history', []))}")
            print(f"   Patrones aprendidos: {len(final_state.get('learned_patterns', []))}")
            if postgres_saver:
                print(f"   Estado persistido automáticamente")
            if not final_state.get('success', False):
                print(f"   Error final: {final_state.get('final_error', 'N/A')}")
            
        except Exception as e:
            print(f"❌ Error crítico en el sistema: {e}")
            import traceback
            print("🔍 Detalles del error:")
            traceback.print_exc()
            
        print(f"\n{'-'*60}\n")

if __name__ == "__main__":
    main()