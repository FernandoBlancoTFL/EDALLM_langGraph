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

def generate_memory_response_with_search(state: AgentState) -> str:
    """
    Busca activamente en el historial para responder la consulta.
    Si no encuentra información relevante, lo comunica honestamente.
    """
    conversation_history = state.get("conversation_history", [])
    query = state["query"]
    
    if not conversation_history:
        return "No tengo memoria de conversaciones anteriores en esta sesión, así que no puedo responder a tu pregunta sobre el historial."
    
    # Crear contexto del historial para búsqueda
    history_context = []
    for i, conv in enumerate(conversation_history, 1):
        query_text = conv.get("query", "N/A")
        response_text = conv.get("response", "N/A")[:200]
        success = conv.get("success", False)
        
        history_context.append(f"""
Conversación {i}:
- Usuario preguntó: {query_text}
- Estado: {'Exitosa' if success else 'No exitosa'}
- Respuesta: {response_text}
        """)
    
    # Prompt para búsqueda inteligente
    search_prompt = f"""
El usuario te pregunta: "{query}"

Tienes acceso al historial de {len(conversation_history)} conversaciones previas:

{chr(10).join(history_context)}

TAREA:
1. Busca en el historial si hay información que responda a la pregunta del usuario
2. Si ENCUENTRAS información relevante: Responde con esa información específica
3. Si NO ENCUENTRAS información: Responde honestamente que no tienes esa información en el historial

IMPORTANTE:
- Sé específico con la información que encuentres
- Si el usuario pregunta su nombre y lo mencionó en alguna conversación, dile su nombre
- Si pregunta sobre análisis previos, menciona qué análisis hizo
- Si NO hay información, sé honesto: "No tengo esa información en nuestro historial"

Genera una respuesta natural y útil.
"""
    
    try:
        from nodes import llm
        response = llm.invoke(search_prompt).content.strip()
        print(f"🔍 Búsqueda en memoria completada")
        return response
    except Exception as e:
        print(f"❌ Error buscando en memoria: {e}")
        # Fallback a respuesta simple
        return f"Tengo {len(conversation_history)} conversaciones en memoria, pero tuve problemas buscando la información específica que solicitaste."

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
    CORREGIDO: Maneja valores None en datasets_used
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
    
    # CORRECCIÓN: Filtrar valores None antes de hacer join
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