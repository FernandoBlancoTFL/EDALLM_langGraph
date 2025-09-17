from config import df, llm
from tools import tools
from utils import build_code_prompt, run_python_with_df
from typing import List
from langchain.agents import Tool
from state import AgentState

def get_tools_summary(tools: List[Tool]) -> str:
    """Devuelve un resumen con nombre y descripción de cada tool."""
    return "\n".join([f"- {t.name}: {t.description}" for t in tools])

def node_clasificar(state: AgentState):
    """El LLM decide qué acción tomar con contexto mejorado"""
    
    # Obtener información del DataFrame si no existe
    if not state.get("df_info"):
        state["df_info"] = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "sample": df.head(2).to_dict()
        }
    
    # Contexto de iteraciones previas
    iteration_context = ""
    if state["iteration_count"] > 0:
        iteration_context = f"\nEsta es la iteración #{state['iteration_count'] + 1}. Intentos previos han fallado."
        if state["execution_history"]:
            last_error = state["execution_history"][-1].get("error", "")
            iteration_context += f"\nÚltimo error: {last_error}"

    tools_summary = get_tools_summary(tools)

    prompt = f"""
        Eres un asistente de análisis de datos experto. Analiza esta consulta y decide la mejor acción.

        CONSULTA: {state['query']}
        DATAFRAME INFO: Columnas = {state['df_info']['columns']}, Shape = {state['df_info']['shape']}
        {iteration_context}

        HERRAMIENTAS DISPONIBLES:
        {tools_summary}

        DECISIÓN:
        Analiza la consulta y selecciona la herramienta más adecuada. 
        Si ninguna herramienta especializada es suficiente, usa Python_Interpreter.

        Formato de salida:
        Thought: <análisis detallado de la consulta y estrategia>
        Action: <nombre exacto de la herramienta elegida>
    """

    response = llm.invoke(prompt).content.strip()

    # Extraer thought y action
    thought, action = "", "Python_Interpreter"
    for line in response.splitlines():
        if line.lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()

    state["thought"] = thought
    state["action"] = action
    state["history"].append(f"Iteración {state['iteration_count']} - Clasificar → {thought[:100]}...")

    print(f"\n🧠 Iteración {state['iteration_count']} - Thought: {thought}")
    print(f"➡️ Action: {action}")

    return state

def node_ejecutar_python(state: AgentState):
    """Ejecuta código Python con manejo robusto de errores y contexto"""
    
    print(f"⚙️ Ejecutando Python - Intento {state['iteration_count'] + 1}")
    
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

def node_validar_y_decidir(state: AgentState):
    """Valida el resultado y decide si continuar iterando"""
    
    state["iteration_count"] += 1
    success = state.get("success", False)
    max_iterations = state.get("max_iterations", 3)
    
    print(f"\n🔍 Validación - Iteración {state['iteration_count']}")
    print(f"   Éxito: {success}")
    print(f"   Iteraciones restantes: {max_iterations - state['iteration_count']}")
    
    # Decidir próxima acción
    if success:
        state["next_node"] = "responder"
        print("   ➡️ Decisión: Proceder a responder (éxito)")
    elif state["iteration_count"] >= max_iterations:
        state["next_node"] = "responder"
        print("   ➡️ Decisión: Proceder a responder (máximo de iteraciones alcanzado)")
    else:
        state["next_node"] = "clasificar"
        print("   ➡️ Decisión: Nueva iteración")
    
    state["history"].append(f"Validar → Iteración {state['iteration_count']}, Éxito: {success}, Próximo: {state['next_node']}")
    
    return state

def node_responder(state: AgentState):
    """Genera la respuesta final basada en todo el contexto"""
    
    success = state.get("success", False)
    
    if success:
        prompt = f"""
Pregunta del usuario: {state['query']}
Resultado obtenido: {state['result']}
Número de iteraciones necesarias: {state['iteration_count']}

Genera una respuesta clara y amigable en español explicando qué se logró.
"""
    else:
        # Analizar todos los errores para dar una respuesta informativa
        errors_summary = []
        for record in state["execution_history"]:
            if not record["success"]:
                errors_summary.append(f"- {record['error_type']}: {record['error']}")
        
        prompt = f"""
Pregunta del usuario: {state['query']}
Después de {state['iteration_count']} iteraciones, no se pudo completar la tarea.

Errores encontrados:
{chr(10).join(errors_summary)}

Genera una respuesta empática en español explicando:
1. Que se intentó resolver la consulta múltiples veces
2. Los principales problemas encontrados (en términos simples)
3. Sugerencias para el usuario (ej: verificar formato de datos, columnas, etc.)
"""

    respuesta = llm.invoke(prompt).content
    print(f"\n🤖 Respuesta Final:\n{respuesta}")
    
    # Log final
    state["history"].append(f"Responder → Finalizado con {'éxito' if success else 'error'}")
    
    return state

def route_after_validation(state: AgentState):
    """Determina la siguiente ruta basada en la validación"""
    success = state.get("success", False)
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    
    print(f"\n🔧 DEBUG route_after_validation:")
    print(f"   Success: {success}")
    print(f"   Iteration: {iteration_count}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Next node from state: {state.get('next_node', 'N/A')}")
    
    if success:
        print("   → Routing to: responder (success)")
        return "responder"
    elif iteration_count >= max_iterations:
        print("   → Routing to: responder (max iterations reached)")
        return "responder"
    else:
        print("   → Routing to: clasificar (continue iteration)")
        return "clasificar"

# Copia aquí tus funciones node_clasificar, node_ejecutar_python, node_validar_y_decidir,
# node_responder y route_after_validation, ajustando imports para usar utils/config.
