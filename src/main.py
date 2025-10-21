import os
import sys
from datetime import datetime
from config import *
from database import create_database_if_not_exists, test_target_database_connection, setup_data_connection
from dataset_manager import initialize_dataset_on_startup
from checkpoints import setup_postgres_saver, get_automatic_thread_id, postgres_saver
from graph import create_graph_with_sql
from utils import show_stored_files, show_conversation_memory, clean_state_for_serialization
from state import AgentState
from database import create_database_if_not_exists, test_target_database_connection, setup_data_connection

def main():
    global postgres_saver
    
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
    
    # configurar PostgresSaver (después de que existe la BD)
    postgres_saver = setup_postgres_saver()
    
    app = create_graph_with_sql()
    
    print("✅ Base de datos PostgreSQL configurada correctamente")
    print(f"💾 Guardado automático a BD: {'ACTIVADO' if ENABLE_AUTO_SAVE_TO_DB else 'DESACTIVADO'}")
    print(f"🧠 Memoria conversacional: {'ACTIVADA' if postgres_saver else 'DESACTIVADA'}")
    
    # Thread ID automático para usuario único
    thread_id = get_automatic_thread_id()
    
    print("🚀 Sistema de Análisis de Datos con Memoria Persistente")
    print("   Escribe 'salir' para terminar\n")
    
    # Mostrar archivos almacenados
    show_stored_files()

    # Mostrar memoria de conversación
    show_conversation_memory(thread_id)

    print()
    
    while True:
        query = input("Pregunta sobre el dataset (o 'salir'): ")

        if not query or query.strip() == "":
            print("⚠️ Por favor, escribe una consulta válida para continuar.")
            print("   No puedo procesar mensajes vacíos.\n")
            continue
        
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
            # NO inicializar estos campos - dejar que el grafo los maneje
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
            # Configurar thread automático para memoria persistente
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