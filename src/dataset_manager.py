import os
import pandas as pd
import psycopg
from typing import Optional
from database import load_db_config, data_connection
from config import DATASETS_TO_PROCESS, DATASET_CONFIG, ENABLE_AUTO_SAVE_TO_DB

# Variables globales para el dataset
dataset_info = None
df = None
dataset_loaded = False

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

def create_dataset_table_from_df(df: pd.DataFrame, connection=None, table_name=None, table_schema=None, semantic_description=None):
    """
    Crea una tabla en PostgreSQL desde un DataFrame.
    MODIFICADO: Ahora incluye descripción semántica en metadatos.
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
            
            # Crear tabla CON columna de descripción semántica
            create_table_sql = f"""
                CREATE TABLE {table_schema}.{table_name} (
                    id SERIAL PRIMARY KEY,
                    {', '.join(column_definitions)},
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    semantic_description TEXT
                )
            """
            
            cursor.execute(create_table_sql)
            
            # Si se proporcionó descripción, agregarla como comentario de tabla
            # CORREGIDO: Usar f-string en lugar de placeholder para DDL
            if semantic_description:
                # Escapar comillas simples en la descripción
                escaped_description = semantic_description.replace("'", "''")
                comment_sql = f"""
                    COMMENT ON TABLE {table_schema}.{table_name} IS '{escaped_description}'
                """
                cursor.execute(comment_sql)
            
            conn.commit()
            
            print(f"✅ Tabla '{table_name}' creada exitosamente")
            if semantic_description:
                print(f"🧠 Descripción semántica almacenada")
            
            return True, column_mapping
            
    except Exception as e:
        print(f"❌ Error creando tabla {table_name}: {e}")
        if conn:
            conn.rollback()
        return False, {}

def insert_dataframe_to_table(df: pd.DataFrame, column_mapping: dict, connection=None, table_name=None, table_schema=None, semantic_description=None):
    """
    Inserta los datos del DataFrame en la tabla PostgreSQL.
    MODIFICADO: Ahora inserta la descripción semántica en cada fila.
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
        
        # Preparar datos para inserción (con descripción semántica)
        columns_list = list(column_mapping.values())
        if semantic_description:
            columns_list.append('semantic_description')
            placeholders = ', '.join(['%s'] * (len(columns_list)))
        else:
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
            for col in column_mapping.values():
                value = row[col]
                if pd.isna(value):
                    row_data.append(None)
                elif isinstance(value, (pd.Timestamp, pd.Timedelta)):
                    row_data.append(value.to_pydatetime() if hasattr(value, 'to_pydatetime') else str(value))
                else:
                    row_data.append(value)
            
            # Agregar descripción semántica al final de cada fila
            if semantic_description:
                row_data.append(semantic_description)
            
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

def generate_semantic_description_with_llm(df: pd.DataFrame, table_name: str, excel_path: str) -> str:
    """
    Genera una descripción semántica del dataset usando LLM.
    Analiza estructura, contenido y propósito del dataset.
    """
    from nodes import llm

    try:
        # Obtener muestra de datos (primeras 5 filas)
        sample_data = df.head(5).to_string()
        
        # Información estructural
        columns_info = ", ".join(df.columns.tolist())
        dtypes_info = df.dtypes.to_string()
        row_count = len(df)
        
        # Estadísticas básicas para columnas numéricas
        numeric_stats = ""
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().to_string()
        
        prompt = f"""
Analiza este dataset y genera una descripción semántica clara y concisa.

INFORMACIÓN DEL DATASET:
- Nombre de archivo: {os.path.basename(excel_path)}
- Nombre de tabla: {table_name}
- Cantidad de filas: {row_count}
- Columnas ({len(df.columns)}): {columns_info}

TIPOS DE DATOS:
{dtypes_info}

MUESTRA DE DATOS (primeras 5 filas):
{sample_data}

{f"ESTADÍSTICAS NUMÉRICAS:{numeric_stats}" if numeric_stats else ""}

TAREA:
Genera una descripción semántica de 2-3 oraciones que explique:
1. Qué tipo de datos contiene este dataset
2. Para qué análisis o consultas podría ser útil
3. Características principales (temporal, geográfico, transaccional, etc.)

La descripción debe ser clara, directa y útil para que un LLM pueda decidir si este dataset es relevante para una consulta de usuario.

Responde SOLO con la descripción, sin formato adicional.
"""
        
        response = llm.invoke(prompt).content.strip()
        
        print(f"📝 Descripción generada para '{table_name}':")
        print(f"   {response[:150]}...")
        
        return response
        
    except Exception as e:
        print(f"⚠️ Error generando descripción semántica: {e}")
        # Fallback a descripción básica
        return f"Dataset con {len(df.columns)} columnas y {len(df)} filas. Columnas principales: {', '.join(df.columns.tolist()[:5])}"

def initialize_dataset_on_startup():
    """
    Inicializa todos los datasets en BD SOLO si ENABLE_AUTO_SAVE_TO_DB es True.
    Se ejecuta automáticamente al iniciar el programa.
    MODIFICADO: Ahora genera descripciones semánticas automáticamente.
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
            
            # 🧠 NUEVO: Generar descripción semántica con LLM
            print(f"🤖 Generando descripción semántica con LLM...")
            semantic_description = generate_semantic_description_with_llm(df_temp, table_name, excel_path)
            
            # Crear tabla con descripción semántica
            success, column_mapping = create_dataset_table_from_df(
                df_temp, 
                dataset_connection, 
                table_name, 
                table_schema,
                semantic_description=semantic_description
            )
            if not success:
                print(f"❌ No se pudo crear la tabla {table_name}")
                continue
            
            # Insertar datos (incluyendo descripción en la primera fila si es necesario)
            insert_success = insert_dataframe_to_table(
                df_temp, 
                column_mapping, 
                dataset_connection, 
                table_name, 
                table_schema,
                semantic_description=semantic_description
            )
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

def load_specific_dataset(table_name: str, connection=None):
    """
    Carga un dataset específico en memoria.
    Retorna (df, dataset_info, success)
    """
    global df, dataset_info, dataset_loaded
    
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