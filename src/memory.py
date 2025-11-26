from typing import List
from datetime import datetime
from checkpoints import load_conversation_history
from state import AgentState


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
        # Filtrar None y valores vacíos
        valid_datasets = [ds for ds in datasets_used if ds is not None and ds != ""]
        if valid_datasets:
            response_parts.append(f"\nHas trabajado principalmente con: {', '.join(valid_datasets)}")
    
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
            # Mover al frente
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

def is_visualization_query(query: str) -> tuple[bool, str]:
    """
    Detecta si la consulta requiere obligatoriamente DataFrame por visualización/análisis complejo.
    Retorna (es_visualización, razón)
    """
    query_lower = query.lower()
    
    # Palabras clave para visualizaciones
    visualization_keywords = {
        'graficar', 'gráfico', 'grafica', 'gráfica', 'grafique',
        'histograma', 'histogram',
        'plot', 'plotear', 'plotea',
        'diagrama', 'chart',
        'scatter', 'dispersión', 'dispersion',
        'barras', 'bar chart',
        'líneas', 'lineas', 'line chart',
        'boxplot', 'box plot', 'caja',
        'heatmap', 'mapa de calor',
        'visualizar', 'visualiza', 'visualización', 'visualizacion',
        'generar gráfico', 'generar grafico', 'crear gráfico', 'crear grafico',
        'mostrar gráfico', 'mostrar grafico',
        'pie chart', 'torta', 'pastel'
    }
    
    # Palabras clave para análisis complejos que requieren DataFrame
    complex_analysis_keywords = {
        'correlación', 'correlacion', 'correlation',
        'regresión', 'regresion', 'regression',
        'clustering', 'agrupar', 'agrupamiento',
        'normalizar', 'normalización', 'normalization',
        'transformar', 'transformación', 'transformation',
        'pivot', 'pivotar', 'tabla pivote',
        'melt', 'unpivot',
        'merge', 'combinar dataframes',
        'estadísticas descriptivas complejas',
        'análisis exploratorio', 'analisis exploratorio',
        'distribución', 'distribucion', 'distribution'
    }
    
    # Verbos de generación que implican visualización
    generation_verbs = {
        'genera', 'generar', 'generá', 'generame',
        'crea', 'crear', 'creá', 'creame',
        'hace', 'hacer', 'hacé', 'haceme',
        'muestra', 'mostrar', 'mostrá', 'mostrame',
        'dibuja', 'dibujar', 'dibujá', 'dibujame'
    }
    
    # Verificar si hay un verbo de generación + término de visualización cercano
    words = query_lower.split()
    for i, word in enumerate(words):
        if any(verb in word for verb in generation_verbs):
            # Verificar las siguientes 3-4 palabras
            context = ' '.join(words[i:i+5])
            for viz_keyword in visualization_keywords:
                if viz_keyword in context:
                    return True, f"Detectado verbo de generación '{word}' + visualización '{viz_keyword}'"
    
    # Verificar palabras clave de visualización directamente
    for keyword in visualization_keywords:
        if keyword in query_lower:
            return True, f"Palabra clave de visualización detectada: '{keyword}'"
    
    # Verificar análisis complejos
    for keyword in complex_analysis_keywords:
        if keyword in query_lower:
            return True, f"Análisis complejo detectado: '{keyword}'"
    
    return False, ""

def is_greeting_query(query: str) -> tuple[bool, str]:
    """
    Detecta si la consulta es un saludo simple.
    Retorna (es_saludo, respuesta_saludo)
    """
    query_lower = query.lower().strip()
    
    # Saludos comunes
    greetings = {
        'hola': '¡Hola! 👋 Soy tu asistente de análisis de datos. ¿En qué puedo ayudarte hoy?',
        'buenos dias': '¡Buenos días! ☀️ Estoy listo para ayudarte con el análisis de tus datos.',
        'buenos días': '¡Buenos días! ☀️ Estoy listo para ayudarte con el análisis de tus datos.',
        'buenas tardes': '¡Buenas tardes! 🌤️ ¿Qué análisis necesitas realizar?',
        'buenas noches': '¡Buenas noches! 🌙 ¿En qué puedo asistirte?',
        'buen dia': '¡Buen día! ☀️ Estoy aquí para ayudarte con tus datos.',
        'buen día': '¡Buen día! ☀️ Estoy aquí para ayudarte con tus datos.',
        'hey': '¡Hola! 👋 ¿Qué análisis quieres hacer hoy?',
        'saludos': '¡Saludos! 👋 ¿En qué puedo ayudarte?',
        'hi': 'Hi! 👋 How can I help you with your data today?',
        'hello': 'Hello! 👋 Ready to analyze your data!',
    }
    
    # Verificar saludos exactos
    for greeting, response in greetings.items():
        if query_lower == greeting or query_lower == greeting + '!' or query_lower == greeting + '.':
            return True, response
    
    # Verificar saludos con palabras adicionales simples (ej: "hola, como estás")
    first_word = query_lower.split()[0] if query_lower.split() else ""
    if first_word in greetings and len(query_lower.split()) <= 4:
        return True, greetings[first_word]
    
    return False, ""


def is_help_query(query: str) -> tuple[bool, str]:
    """
    Detecta si el usuario pregunta qué puede hacer el sistema o pide ayuda.
    Retorna (es_ayuda, respuesta_ayuda)
    """
    query_lower = query.lower().strip()
    
    # Palabras clave de ayuda/capacidades
    help_patterns = [
        'que puedes hacer', 'qué puedes hacer',
        'que sabes hacer', 'qué sabes hacer',
        'que haces', 'qué haces',
        'como funciona', 'cómo funciona',
        'como te uso', 'cómo te uso',
        'que ofreces', 'qué ofreces',
        'cuales son tus funciones', 'cuáles son tus funciones',
        'para que sirves', 'para qué sirves',
        'ayuda', 'help',
        'que puedo preguntar', 'qué puedo preguntar',
        'como puedo usarte', 'cómo puedo usarte',
        'que me puedes decir', 'qué me puedes decir'
    ]
    
    for pattern in help_patterns:
        if pattern in query_lower:
            return True, generate_help_response()
    
    return False, ""


def generate_help_response() -> str:
    """
    Genera una respuesta explicando las capacidades del sistema.
    """
    response = """¡Hola! 👋 Soy tu asistente de análisis de datos con IA. Aquí está lo que puedo hacer por ti:

📊 **Análisis de Datos:**
   • Consultas SQL rápidas (conteos, filtros, agregaciones)
   • Análisis estadísticos complejos con Python/Pandas
   • Exploración de datasets completos

📈 **Visualizaciones:**
   • Histogramas, gráficos de barras, dispersión
   • Gráficos de líneas, boxplots, heatmaps
   • Cualquier tipo de visualización con matplotlib/seaborn

🧠 **Memoria Conversacional:**
   • Recuerdo tus consultas anteriores
   • Aprendo tus preferencias de análisis
   • Puedes preguntarme sobre nuestras conversaciones pasadas

💡 **Ejemplos de preguntas:**
   • "¿Cuántas filas tiene el dataset de viajes?"
   • "Genera un histograma de la columna edad"
   • "Muéstrame las columnas del dataset de cocodrilos"
   • "¿Cuál es el promedio de ventas por mes?"
   • "Visualiza la correlación entre precio y cantidad"

📁 **Datasets disponibles:** Puedo trabajar con todos los datasets cargados en tu base de datos.

📤 **Formatos soportados:** Puedes subir archivos en formato .csv, .xls y .xlsx

¿Qué análisis te gustaría hacer? 🚀"""
    
    return response


def is_simple_conversation_query(query: str) -> tuple[bool, str]:
    """
    Detecta preguntas conversacionales simples que no requieren análisis de datos.
    Retorna (es_conversación, respuesta)
    """
    query_lower = query.lower().strip()
    
    # Patrones conversacionales simples
    simple_conversations = {
        'como estas': '¡Estoy funcionando perfectamente! 🤖 Listo para analizar tus datos. ¿Qué necesitas?',
        'cómo estás': '¡Estoy funcionando perfectamente! 🤖 Listo para analizar tus datos. ¿Qué necesitas?',
        'como te llamas': 'Soy tu asistente de análisis de datos con IA. Puedes llamarme simplemente "Asistente". ¿En qué puedo ayudarte?',
        'cómo te llamas': 'Soy tu asistente de análisis de datos con IA. Puedes llamarme simplemente "Asistente". ¿En qué puedo ayudarte?',
        'cual es tu nombre': 'Soy tu asistente de análisis de datos. Mi función es ayudarte a analizar y visualizar datos. ¿Qué análisis necesitas?',
        'cuál es tu nombre': 'Soy tu asistente de análisis de datos. Mi función es ayudarte a analizar y visualizar datos. ¿Qué análisis necesitas?',
        'gracias': '¡De nada! 😊 Si necesitas más análisis, aquí estaré.',
        'muchas gracias': '¡Es un placer ayudarte! 😊 ¿Algo más que pueda hacer por ti?',
        'perfecto': '¡Genial! 👍 ¿Hay algo más en lo que pueda ayudarte?',
        'excelente': '¡Me alegra que te sea útil! 🎉 ¿Necesitas algo más?',
        'ok': '👍 ¿Hay algo más que quieras analizar?',
        'vale': '👍 ¿Necesitas otro análisis?',
    }
    
    # Verificar coincidencias exactas
    for pattern, response in simple_conversations.items():
        if query_lower == pattern or query_lower == pattern + '!' or query_lower == pattern + '.':
            return True, response
    
    return False, ""


def is_general_query(query: str) -> tuple[bool, str, str]:
    """
    Función principal que detecta cualquier tipo de consulta general (no de datos).
    Retorna (es_general, tipo, respuesta)
    Tipos: 'greeting', 'help', 'conversation'
    """
    # Verificar en orden de prioridad
    
    # 1. Saludos
    is_greeting, greeting_response = is_greeting_query(query)
    if is_greeting:
        return True, 'greeting', greeting_response
    
    # 2. Ayuda/Capacidades
    is_help, help_response = is_help_query(query)
    if is_help:
        return True, 'help', help_response
    
    # 3. Conversación simple
    is_conversation, conv_response = is_simple_conversation_query(query)
    if is_conversation:
        return True, 'conversation', conv_response
    
    return False, '', ''