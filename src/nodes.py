import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from datetime import datetime
from state import AgentState
from config import API_KEY, GROQ_KEY, SINGLE_USER_THREAD_ID
from database import data_connection, load_db_config
from dataset_manager import df
from tools import run_python_with_df, get_tools_summary, tools
from prompts import build_code_prompt
from memory import *
from multi_dataset import get_all_available_datasets, identify_dataset_from_query_with_memory
from database import get_table_metadata_light
from utils import clean_state_for_serialization
import psycopg
from dataset_manager import ensure_dataset_loaded
import dataset_manager

# Inicializar LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=API_KEY, temperature=0)
# llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_KEY, temperature=0)
# llm = ChatOllama(model="gemma3", temperature=0)

def nodo_estrategia_datos(state: AgentState):
    """
    Recupera historial desde PostgresSaver y actualiza contexto
    """
    print("🧠 Iniciando análisis con recuperación de memoria...")

    # DETECCIÓN 0: Consultas generales (saludos, ayuda, conversación) - ANTES DE TODO
    is_general, general_type, general_response = is_general_query(state["query"])
    if is_general:
        print(f"💬 Consulta general detectada ({general_type}) - Respuesta instantánea")
        state["data_strategy"] = "general"
        state["strategy_reason"] = f"Consulta general de tipo '{general_type}' - no requiere análisis de datos"
        state["result"] = general_response
        state["success"] = True
        state["history"].append(f"General → Respuesta instantánea ({general_type})")
        
        # No actualizar memoria para consultas triviales
        return state

    # Recuperar historial (solo si no es consulta general)
    if not state.get("conversation_history") or len(state.get("conversation_history", [])) == 0:
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
        
        if not state.get("learned_patterns"):
            state["learned_patterns"] = extract_learned_patterns_from_history(state["conversation_history"])
    else:
        state["memory_summary"] = "Primera conversación con el usuario"
        print("🆕 Primera interacción - sin historial previo")
    
    # DETECCIÓN 1: Consultas sobre memoria
    if is_memory_query(state["query"]):
        print("🧠 Consulta sobre memoria detectada - respuesta directa")
        state["data_strategy"] = "memory"
        state["strategy_reason"] = "Consulta sobre historial de conversaciones - respuesta directa desde memoria cargada"
        state["result"] = generate_memory_response(state)
        state["success"] = True
        state["history"].append(f"Memoria → Respuesta directa sobre historial")
        return state
    
    print("🔍 Analizando estrategia de acceso a datos...")
    
    if not state.get("available_datasets"):
        state["available_datasets"] = get_all_available_datasets()
    
    if not state.get("selected_dataset"):
        selected_dataset = identify_dataset_from_query_with_memory(
            state["query"], 
            state["available_datasets"],
            state["user_context"]
        )
        state["selected_dataset"] = selected_dataset
        if selected_dataset:
            state["selected_dataset"] = selected_dataset
            state["dataset_context"] = state["available_datasets"][selected_dataset]
    
    # Obtener metadatos y analizar estrategia
    table_metadata = get_table_metadata_light(state["selected_dataset"])
    state["table_metadata"] = table_metadata

    # DETECCIÓN 2: Visualizaciones - forzar DATAFRAME
    is_viz, viz_reason = is_visualization_query(state["query"])
    if is_viz:
        print(f"📊 Visualización detectada - FORZANDO estrategia DATAFRAME")
        print(f"🔍 Razón: {viz_reason}")
        state["data_strategy"] = "dataframe"
        state["sql_feasible"] = False
        state["strategy_reason"] = f"Estrategia DATAFRAME forzada automáticamente: {viz_reason}"
        state["history"].append(f"Estrategia → DATAFRAME (auto-detección: {viz_reason})")
        
        return state
    
    # DETECCIÓN 3: Análisis normal - usar LLM
    print("🔍 Usando LLM para determinar estrategia...")
    
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
    
    # Extraer decisión
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
    
    print(f"📊 Estrategia seleccionada: {strategy.upper()} (LLM)")
    print(f"🔍 Razón (usando memoria): {reason}")
    
    state["history"].append(f"Estrategia → {strategy.upper()} - {reason}")
    
    return state

def node_clasificar_modificado(state: AgentState):
    """Se enfoca solo en dataset selection y tool selection"""
    
    # La estrategia ya fue definida por nodo_estrategia_datos
    data_strategy = state.get("data_strategy", "dataframe")
    
    # Skip LLM para consultas de memoria
    if data_strategy == "memory":
        state["thought"] = "Consulta de memoria - no requiere procesamiento de datos"
        state["action"] = "memory_query"
        state["history"].append("Clasificar (Mod) → Memory Query (skip LLM)")
        return state
    
    # Skip LLM para consultas generales (saludos, ayuda, conversación)
    if data_strategy == "general":
        state["thought"] = "Consulta general - respuesta directa sin análisis de datos"
        state["action"] = "general_response"
        print(f"💬 Consulta general - sin clasificación de herramientas")
        state["history"].append("Clasificar (Mod) → General Query (skip LLM)")
        return state
    
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

    # Obtener metadatos y nombre real de tabla
    table_metadata = get_table_metadata_light(state['selected_dataset'])
    
    # NUEVO: Usar el nombre real de la tabla si está disponible
    actual_table_name = table_metadata.get('actual_table_name', state['selected_dataset'])
    
    if actual_table_name != state['selected_dataset']:
        print(f"🔄 Usando tabla real: {actual_table_name}")
        state['selected_dataset'] = actual_table_name
    
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
    
    # En lugar de usar 'df' directamente
    if not state.get("df_info") or 'columns' not in state["df_info"]:
        sample_clean = dataset_manager.df.head(2).fillna("NULL").to_dict()
        state["df_info"] = {
            "columns": list(dataset_manager.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in dataset_manager.df.dtypes.items()},
            "shape": dataset_manager.df.shape,
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
    """Maneja fallbacks entre SQL y DataFrame"""
    
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
    Genera respuestas interpretativas con datos específicos obtenidos
    """
    success = state.get("success", False)
    data_strategy = state.get("data_strategy", "dataframe")
    
    # Detectar si es una consulta que NO debe guardarse en memoria
    skip_memory = data_strategy in ["general", "greeting", "help", "conversation"]
    
    if success:
        # Verificar si ya hay una respuesta directa (consultas generales, memoria)
        if state.get("result") and data_strategy in ["memory", "general"]:
            # Ya tiene respuesta generada, solo mostrarla
            respuesta = state["result"]
            print(f"\n🤖 Respuesta Final:\n{respuesta}")
            # NUEVO: Guardar respuesta LLM
            state["llm_response"] = respuesta
            
            # Para consultas de memoria, SÍ guardar (son consultas relevantes)
            if data_strategy == "memory":
                skip_memory = False
            else:
                # Para consultas generales, NO guardar y terminar aquí
                state["history"].append(f"Responder → Consulta general - sin actualización de memoria")
                return state
        
        else:
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
            
            respuesta = llm.invoke(prompt).content
            print(f"\n🤖 Respuesta Final:\n{respuesta}")

            # NUEVO: Guardar respuesta LLM
            state["llm_response"] = respuesta
    
    else:
        # Manejo de errores
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

        # NUEVO: Guardar la respuesta del LLM en el estado
        state["llm_response"] = respuesta
    
    # ACTUALIZAR MEMORIA solo para consultas relevantes
    if not skip_memory:
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
    else:
        print("ℹ️ Consulta general - no se guarda en memoria conversacional")
        state["history"].append(f"Responder → Consulta general - sin actualización de memoria")
    
    return state