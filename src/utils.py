import os
import re
import pandas as pd
from datetime import datetime
from dataset_manager import list_stored_tables
from checkpoints import load_conversation_history
from datetime import datetime

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

def show_stored_files():
    """
    Muestra los archivos almacenados en la BD de forma amigable.
    """
    # print("🔍 Buscando tablas en la base de datos...")
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

def generate_unique_plot_filename(base_name: str) -> str:
    """
    Genera un nombre único para un archivo de gráfico con timestamp.
    
    Args:
        base_name: Nombre base del archivo (ej: "histogram_edad")
    
    Returns:
        Nombre único con timestamp (ej: "histogram_edad_20231008_143022.png")
    """
    # Limpiar el nombre base (remover .png si existe)
    base_name = base_name.replace('.png', '')
    
    # Generar timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Construir nombre único
    unique_name = f"{base_name}_{timestamp}.png"
    
    return unique_name

def extract_plot_filename_from_result(result_text: str) -> str:
    """
    Extrae el nombre del archivo de gráfico desde el texto de resultado.
    MEJORADO: Ahora captura más patrones.
    
    Args:
        result_text: Texto que contiene la ruta del archivo guardado
    
    Returns:
        Nombre del archivo (ej: "histogram_edad_20231008_143022.png") o None
    """
    if not result_text:
        return None
    
    result_str = str(result_text)
    
    # Patrón 1: "outputs/nombre_archivo.png"
    match = re.search(r'outputs[/\\]([^\s\'"]+\.png)', result_str)
    if match:
        return match.group(1)
    
    # Patrón 2: "./src/outputs/nombre_archivo.png"
    match = re.search(r'\.\/src\/outputs[/\\]([^\s\'"]+\.png)', result_str)
    if match:
        return match.group(1)
    
    # Patrón 3: "src/outputs/nombre_archivo.png"
    match = re.search(r'src[/\\]outputs[/\\]([^\s\'"]+\.png)', result_str)
    if match:
        return match.group(1)
    
    # Patrón 4: Solo el nombre del archivo con .png (sin path)
    # Ejemplo: "histogram_booking_value.png" o "`histogram_booking_value.png`"
    match = re.search(r'[`\']?([a-zA-Z0-9_\-]+\.png)[`\']?', result_str)
    if match:
        filename = match.group(1)
        # Verificar que el archivo existe en outputs
        filepath = os.path.join("./src/outputs", filename)
        if os.path.exists(filepath):
            return filename
    
    # Patrón 5: Buscar cualquier .png mencionado
    match = re.search(r'([a-zA-Z0-9_\-]+_\d{8}_\d{6}\.png)', result_str)
    if match:
        return match.group(1)
    
    return None

def get_plot_metadata(filename: str) -> dict:
    """
    Obtiene metadata de un archivo de gráfico.
    
    Args:
        filename: Nombre del archivo (ej: "histogram_edad_20231008_143022.png")
    
    Returns:
        Diccionario con metadata del gráfico
    """
    filepath = os.path.join("./src/outputs", filename)
    
    metadata = {
        "filename": filename,
        "exists": os.path.exists(filepath),
        "created_at": None,
        "size_bytes": None
    }
    
    if metadata["exists"]:
        stat = os.stat(filepath)
        metadata["created_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
        metadata["size_bytes"] = stat.st_size
    
    return metadata

