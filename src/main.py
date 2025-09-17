from graph import create_graph

def main():
    app = create_graph()
    print("🚀 Sistema de Análisis de Datos")
    print("Escribe 'salir' para terminar\n")

    while True:
        query = input("Pregunta sobre el dataset (o 'salir'): ")
        if query.lower() == "salir":
            break

        # Estado inicial con configuración completa
        initial_state = {
            "query": query,
            "action": "",
            "result": None,
            "thought": "",
            "history": [],
            "execution_history": [],      # Historial detallado de ejecuciones
            "iteration_count": 0,         # Contador de iteraciones
            "max_iterations": 3,          # Máximo de iteraciones
            "df_info": {},                # Info del DataFrame (se llena automáticamente)
            "success": False,             # Flag de éxito
            "final_error": None,          # Error final si aplica
            "next_node": "clasificar"     # Control de flujo
        }

        try:
            final_state = app.invoke(initial_state)
            print("\n📊 RESUMEN:")
            print(f"   Iteraciones: {final_state['iteration_count']}")
            print(f"   Éxito: {final_state.get('success', False)}")
            if not final_state.get('success', False):
                print(f"   Error final: {final_state.get('final_error', 'N/A')}")
            print(f"   Historial: {len(final_state['history'])} eventos")
            
        except Exception as e:
            print(f"❌ Error crítico: {e}")

if __name__ == "__main__":
    main()
