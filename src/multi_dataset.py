import os
import pandas as pd
import psycopg
from typing import Any
from dataset_manager import list_stored_tables, get_dataset_table_info_by_name
from database import load_db_config
from config import DATASETS_TO_PROCESS

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

def get_semantic_descriptions_from_db(connection=None):
    """
    Recupera las descripciones semánticas de todas las tablas desde la BD.
    Retorna un diccionario {table_name: semantic_description}
    """
    conn = connection
    if conn is None:
        try:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            should_close = True
        except Exception as e:
            print(f"⚠️ Error conectando para obtener descripciones: {e}")
            return {}
    else:
        should_close = False
    
    descriptions = {}
    
    try:
        with conn.cursor() as cursor:
            # Obtener todas las tablas que tienen columna semantic_description
            stored_tables = list_stored_tables(conn)
            
            for table_name in stored_tables:
                try:
                    # Obtener la descripción desde la primera fila (todas tienen la misma)
                    query = f"""
                        SELECT semantic_description 
                        FROM public.{table_name} 
                        WHERE semantic_description IS NOT NULL 
                        LIMIT 1
                    """
                    cursor.execute(query)
                    result = cursor.fetchone()
                    
                    if result and result[0]:
                        descriptions[table_name] = result[0]
                        print(f"📖 Descripción recuperada para '{table_name}'")
                except Exception as e:
                    print(f"⚠️ Error obteniendo descripción de {table_name}: {e}")
                    # Fallback: intentar obtener desde comentario de tabla
                    try:
                        comment_query = """
                            SELECT obj_description(oid) 
                            FROM pg_class 
                            WHERE relname = %s AND relnamespace = 'public'::regnamespace
                        """
                        cursor.execute(comment_query, (table_name,))
                        comment_result = cursor.fetchone()
                        if comment_result and comment_result[0]:
                            descriptions[table_name] = comment_result[0]
                    except:
                        pass
        
        return descriptions
        
    except Exception as e:
        print(f"❌ Error general obteniendo descripciones semánticas: {e}")
        return {}
    finally:
        if should_close and conn:
            conn.close()

def identify_dataset_with_llm(query: str, available_datasets: dict, semantic_descriptions: dict, user_context: dict) -> str:
    """
    Usa LLM para seleccionar el dataset más apropiado basándose en descripciones semánticas.
    """
    if not available_datasets:
        print("⚠️ No hay datasets disponibles")
        return None
    
    # Construir lista de datasets con sus descripciones
    datasets_info = []
    for table_name, info in available_datasets.items():
        semantic_desc = semantic_descriptions.get(table_name, info.get("description", "Sin descripción"))
        datasets_info.append(f"""
Dataset: {table_name}
Nombre amigable: {info.get('friendly_name', table_name)}
Descripción: {semantic_desc}
Columnas principales: {', '.join(info.get('main_columns', [])[:5])}
Cantidad de filas: {info.get('row_count', 'N/A')}
        """)
    
    # Considerar historial del usuario
    common_datasets_info = ""
    if user_context.get("common_datasets"):
        common_datasets_info = f"\nDATASETS MÁS USADOS POR EL USUARIO: {', '.join(user_context['common_datasets'][:3])}"
    
    prompt = f"""
Analiza la consulta del usuario y selecciona el dataset MÁS apropiado.

CONSULTA DEL USUARIO:
{query}
{common_datasets_info}

DATASETS DISPONIBLES:
{chr(10).join(datasets_info)}

INSTRUCCIONES:
- Selecciona el dataset cuya descripción mejor coincida con la intención de la consulta
- Considera el contexto semántico, no solo palabras clave exactas
- Si el usuario menciona análisis previos, considera los datasets más usados
- Si hay ambigüedad, elige el dataset más relevante semánticamente

Responde SOLO con el nombre exacto de la tabla (table_name), sin explicaciones adicionales.
Ejemplo de respuesta válida: dataset_rides
"""
    
    try:
        from nodes import llm
        response = llm.invoke(prompt).content.strip()
        
        # Limpiar respuesta (puede venir con comillas, espacios, etc.)
        selected = response.replace('"', '').replace("'", "").strip()
        
        # Verificar que la respuesta sea válida
        if selected in available_datasets:
            print(f"🤖 LLM seleccionó dataset: {selected}")
            print(f"   Razón: Mejor coincidencia semántica con la consulta")
            return selected
        else:
            print(f"⚠️ LLM respondió con dataset inválido: {selected}")
            print(f"   Fallback: usando primer dataset disponible")
            return list(available_datasets.keys())[0]
            
    except Exception as e:
        print(f"❌ Error en selección con LLM: {e}")
        # Fallback a método tradicional
        return identify_dataset_from_query(query, available_datasets)

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

def identify_dataset_from_query_with_memory(query: str, available_datasets: dict, user_context: dict) -> str:
    """
    Versión mejorada que usa LLM con descripciones semánticas.
    MODIFICADO: Ahora prioriza selección por LLM usando descripciones semánticas.
    """
    if not available_datasets:
        return None
    
    # Obtener descripciones semánticas de la BD
    semantic_descriptions = get_semantic_descriptions_from_db()
    
    # Si hay descripciones semánticas disponibles, usar LLM para selección inteligente
    if semantic_descriptions:
        print("🧠 Usando LLM para selección de dataset (basado en descripciones semánticas)")
        return identify_dataset_with_llm(query, available_datasets, semantic_descriptions, user_context)
    else:
        print("⚠️ No se encontraron descripciones semánticas, usando método tradicional")
        # Fallback al método original
        base_result = identify_dataset_from_query(query, available_datasets)
        
        # Considerar datasets comunes del usuario
        common_datasets = user_context.get("common_datasets", [])
        if common_datasets and base_result in common_datasets:
            print(f"✅ Dataset confirmado por historial: {base_result}")
            return base_result
        
        # Si hay ambigüedad, preferir el dataset más usado históricamente
        if not base_result and common_datasets:
            preferred = common_datasets[0]
            print(f"🔄 Usando dataset preferido por historial: {preferred}")
            return preferred
        
        return base_result