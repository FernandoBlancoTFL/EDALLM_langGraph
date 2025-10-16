"""
Servicio para gestionar el historial de conversaciones y checkpoints.
"""
from typing import List, Dict, Optional
from datetime import datetime
from src.config import SINGLE_USER_THREAD_ID
from checkpoints import get_postgres_saver


def get_conversation_history(
    thread_id: str = None,
    limit: int = 30,
    include_incomplete: bool = False
) -> List[Dict]:
    """
    Obtiene el historial de conversaciones del usuario desde los checkpoints.
    
    Args:
        thread_id: ID del thread (usa el default si no se provee)
        limit: Número máximo de conversaciones a retornar
        include_incomplete: Si True, incluye checkpoints sin response_metadata
    
    Returns:
        Lista de conversaciones ordenadas de más reciente a más antigua
    """
    try:
        postgres_saver = get_postgres_saver()
        
        if thread_id is None:
            thread_id = SINGLE_USER_THREAD_ID
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Obtener checkpoints (pedimos más para filtrar después)
        checkpoints_list = list(postgres_saver.list(config, limit=limit * 3))
        
        conversations = []
        seen_queries = set()  # Para evitar duplicados de la misma query
        
        for checkpoint_tuple in checkpoints_list:
            checkpoint_data = checkpoint_tuple[1]
            
            # Obtener valores del checkpoint
            channel_values = checkpoint_data.get('channel_values') or checkpoint_data.get('values', {})

            # Filtrar: solo checkpoints útiles
            has_response_metadata = 'response_metadata' in channel_values
            has_llm_response = 'llm_response' in channel_values
            is_success = channel_values.get('success', False)

            # IMPORTANTE: Usar query desde response_metadata (más confiable)
            response_metadata = channel_values.get('response_metadata', {})
            query = response_metadata.get('query') or channel_values.get('query')
            
            # Criterio de inclusión
            if include_incomplete:
                # Incluir cualquier checkpoint con query
                should_include = query is not None
            else:
                # Solo checkpoints completos
                should_include = (
                    has_response_metadata and 
                    has_llm_response and 
                    is_success and 
                    query is not None
                )
            
            if not should_include:
                continue
            
            # Evitar duplicados usando el timestamp de response_metadata
            # (es el mismo para todos los checkpoints de una conversación)
            # response_metadata = channel_values.get('response_metadata', {})
            metadata_timestamp = response_metadata.get('timestamp')

            if metadata_timestamp:
                # Usar timestamp de metadata (más confiable)
                query_key = f"{query}_{metadata_timestamp}"
            else:
                # Fallback: usar timestamp del checkpoint
                query_key = f"{query}_{checkpoint_data.get('ts', '')[:19]}"

            if query_key in seen_queries:
                continue
            seen_queries.add(query_key)
            
            # Construir objeto de conversación
            # IMPORTANTE: Usar query desde response_metadata para consistencia
            response_metadata = channel_values.get('response_metadata', {})
            conversation = {
                "checkpoint_id": checkpoint_data.get('id'),
                "timestamp": checkpoint_data.get('ts'),
                "query": response_metadata.get('query') or query,  # Priorizar metadata
                "llm_response": channel_values.get('llm_response'),
                "success": is_success,
                "response_metadata": response_metadata
            }
            
            conversations.append(conversation)
            
            # Límite alcanzado
            if len(conversations) >= limit:
                break
        
        return conversations
    
    except Exception as e:
        print(f"❌ Error obteniendo historial de conversaciones: {e}")
        import traceback
        traceback.print_exc()
        return []
