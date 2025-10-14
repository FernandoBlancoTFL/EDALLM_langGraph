import os
import pandas as pd
import psycopg
from typing import Optional
from database import load_db_config, data_connection
from config import ENABLE_AUTO_SAVE_TO_DB

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

def check_dataset_table_exists(connection=None, table_name=None, table_schema='public'):
    """
    Verifica si una tabla específica existe en PostgreSQL.
    MODIFICADO: Ya no usa valores por defecto de config, requiere table_name explícito.
    """
    conn = connection
    
    if conn is None:
        print("⚠️ No se puede verificar tabla: no hay conexión disponible")
        return False
    
    if table_name is None:
        print("⚠️ No se especificó nombre de tabla para verificar")
        return False
    
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
            # print(f"🔍 Tabla '{table_name}' {'existe' if exists else 'no existe'} en BD")
            return exists
            
    except Exception as e:
        print(f"⚠️ Error verificando tabla {table_name}: {e}")
        return False

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
            # print("🔗 Conexión temporal creada para listar tablas")
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
                'checkpoints',
                'document_registry'
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

def create_dataset_table_from_df(df: pd.DataFrame, connection=None, table_name=None, table_schema='public', semantic_description=None):
    """
    Crea una tabla en PostgreSQL desde un DataFrame.
    MODIFICADO: Requiere table_name explícito, no usa valores por defecto de config.
    """
    conn = connection
    
    if conn is None:
        print("⚠️ No se puede crear tabla: no hay conexión disponible")
        return False, {}
    
    if table_name is None:
        print("⚠️ No se especificó nombre de tabla")
        return False, {}
    
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

def insert_dataframe_to_table(df: pd.DataFrame, column_mapping: dict, connection=None, table_name=None, table_schema='public', semantic_description=None):
    """
    Inserta los datos del DataFrame en la tabla PostgreSQL.
    MODIFICADO: Requiere table_name explícito, no usa valores por defecto de config.
    """
    conn = connection
    
    if conn is None:
        print("⚠️ No se puede insertar datos: no hay conexión disponible")
        return False
    
    if table_name is None:
        print("⚠️ No se especificó nombre de tabla")
        return False
    
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
            inserted_count = cursor.fetchone()[0]
            
            print(f"✅ {inserted_count} filas insertadas correctamente en '{table_name}'")
            return True
            
    except Exception as e:
        print(f"❌ Error insertando datos en {table_name}: {e}")
        if conn:
            conn.rollback()
        return False

def generate_semantic_description_with_llm(df: pd.DataFrame, table_name: str, filename: str = None) -> str:
    """
    Genera una descripción semántica del dataset usando LLM.
    MODIFICADO: Ya no requiere ruta de archivo, usa filename opcional.
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
        
        filename_str = f"- Nombre de archivo: {filename}" if filename else ""
        
        prompt = f"""
Analiza este dataset y genera una descripción semántica clara y concisa.

INFORMACIÓN DEL DATASET:
{filename_str}
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

def ensure_dataset_loaded(state=None):
    """
    Función para cargar el dataset solo cuando sea necesario.
    MEJORADO: Ahora mapea automáticamente nombres parciales al nombre completo de la tabla.
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
    
    # Crear conexión temporal
    dataset_connection = None
    try:
        db_config = load_db_config()
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        dataset_connection = psycopg.connect(connection_string)
        print("🔗 Conexión temporal creada para dataset")
    except Exception as e:
        print(f"⚠️ Error creando conexión: {e}")
        return False
    
    try:
        # MEJORADO: Buscar la tabla real (puede ser nombre parcial)
        actual_table_name = target_dataset
        
        with dataset_connection.cursor() as cursor:
            # Verificar si existe exactamente como se pasó
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """, (target_dataset,))
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                # Buscar tabla que empiece con el nombre dado
                print(f"🔍 Buscando tabla que coincida con '{target_dataset}'...")
                
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE %s
                    AND table_name NOT IN ('document_registry', 'checkpoints', 'checkpoint_writes')
                    LIMIT 1
                """, (target_dataset + '%',))
                
                result = cursor.fetchone()
                if result:
                    actual_table_name = result[0]
                    print(f"🔄 Mapeado: '{target_dataset}' → '{actual_table_name}'")
                    # IMPORTANTE: Actualizar el estado con el nombre real
                    if state:
                        state["selected_dataset"] = actual_table_name
                else:
                    print(f"❌ Tabla '{target_dataset}' no existe en la BD")
                    return False
        
        # Cargar desde la BD usando el nombre real
        print(f"🔄 Cargando '{actual_table_name}' desde PostgreSQL...")
        
        with dataset_connection.cursor() as cursor:
            # Cargar todos los datos de la tabla
            cursor.execute(f'SELECT * FROM public."{actual_table_name}"')
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(rows, columns=columns)
            
            # Eliminar columnas del sistema si existen
            system_columns = ['created_at', 'semantic_description']
            for col in system_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])
            
            dataset_info = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "table_name": actual_table_name  # Usar el nombre real
            }
            dataset_loaded = True
            print(f"✅ Dataset '{actual_table_name}' cargado desde BD: {df.shape}")
            return True
            
    except Exception as e:
        print(f"❌ Error cargando dataset desde BD: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if dataset_connection:
            dataset_connection.close()
