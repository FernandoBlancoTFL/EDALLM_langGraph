import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools import PythonREPLTool  # intérprete de Python de Camel AI
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any

# ======================
# 1. Configuración inicial
# ======================
load_dotenv() # Cargar variables de entorno (.env debe contener GOOGLE_API_KEY)
api_key = os.getenv("GOOGLE_API_KEY")
os.makedirs("./Outputs", exist_ok=True) # Crear carpeta para guardar gráficos si no existe

# Cargar dataset con pandas
df = pd.read_excel("./Data/ncr_ride_bookings.xlsx")

# Inicializar LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
# llm = ChatOllama(model="gemma3", temperature=0)

# ======================
# 2. Definir intérprete Python con acceso a df
# ======================
python_repl = PythonREPLTool()

def run_python_with_df(code: str):
    """
    Ejecuta código Python con acceso al DataFrame `df` ya cargado.
    No permite redefinir `df` para evitar simulaciones inventadas.
    """
    local_vars = {"df": df, "pd": pd, "plt": plt, "sns": sns}

    # Bloquear intentos de volver a definir df o crear datos de ejemplo
    prohibited_patterns = [
        "pd.DataFrame",
        "df = ",
        "data = {",
        "= pd.DataFrame",
        "DataFrame(",
        "# Datos de ejemplo",
        "datos de ejemplo",
        "reemplaza con tu DataFrame"
    ]
    
    code_lower = code.lower()
    for pattern in prohibited_patterns:
        if pattern.lower() in code_lower:
            return f"❌ Error: Código bloqueado. Detectado intento de crear DataFrame o datos de ejemplo: '{pattern}'. Usa SOLO el df existente."

    try:
        exec(code, {}, local_vars)
        return "✅ Código ejecutado con éxito."
    except Exception as e:
        return f"⚠️ Error ejecutando código: {e}"


# ======================
# 3. Tools
# ======================
def get_dataframe(_):
    """
    Devuelve el DataFrame completo al LLM.
    Este tool permite que el agente acceda a 'df' directamente para cualquier análisis.
    """
    return df

def get_summary(_):
    """Devuelve un resumen general del dataset"""
    return str(df.describe(include="all"))

def get_columns(_):
    """Devuelve las columnas del dataset"""
    return str(df.columns.tolist())

def get_missing_values(_):
    """Devuelve la cantidad de valores nulos por columna"""
    return str(df.isnull().sum())

def get_dtypes_and_uniques(_):
    """Devuelve los tipos de datos de cada columna y la cantidad de valores únicos."""
    return str(pd.DataFrame({
        "dtype": df.dtypes,
        "unique_values": df.nunique()
}))

def get_categorical_distribution(column: str):
    """Devuelve la distribución de frecuencias de una columna categórica."""
    if column not in df.columns:
        return f"Columna {column} no encontrada."
    return str(df[column].value_counts(dropna=False).head(20))

def get_numeric_dispersion(_):
    """Devuelve rango, varianza y desviación estándar de variables numéricas."""
    numeric_cols = df.select_dtypes(include=["number"])
    return str(numeric_cols.agg(["min", "max", "var", "std"]))

def get_correlations(_):
    """Devuelve la matriz de correlaciones entre variables numéricas."""
    numeric_cols = df.select_dtypes(include=["number"])
    return str(numeric_cols.corr())

def detect_outliers(column: str):
    """Devuelve los valores atípicos (según IQR) de una columna numérica."""
    if column not in df.columns:
        return f"Columna {column} no encontrada."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"La columna {column} no es numérica."
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)][column]
    return str(outliers.head(50))  # solo mostramos algunos

def get_time_series_summary(_):
    """Devuelve la cantidad de viajes por fecha (si existe columna de fecha)."""
    if "Date" not in df.columns:
        return "No existe columna Date."
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return str(df.groupby(df["Date"].dt.date).size().head(30))

# Gráficas

def plot_histogram(column: str):
    """Genera un histograma de una columna numérica, lo guarda en carpeta y lo muestra en ventana."""
    if column not in df.columns:
        return f"Columna {column} no encontrada."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"La columna {column} no es numérica."
    
    plt.figure(figsize=(10,6))
    df[column].dropna().hist(bins=30, edgecolor="black", alpha=0.7)
    plt.title(f"Histograma de {column}", fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.grid(axis="y", alpha=0.5)

    file_path = f"./Outputs/histogram_{column}.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Histograma generado y guardado en {file_path}"

def plot_correlation_heatmap(_):
    """Genera un heatmap de correlaciones entre variables numéricas."""
    numeric_cols = df.select_dtypes(include=["number"])
    if numeric_cols.empty:
        return "No hay columnas numéricas para correlacionar."

    import seaborn as sns
    plt.figure(figsize=(12,8))
    sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Mapa de calor de correlaciones", fontsize=16)

    file_path = "./Outputs/correlation_heatmap.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Heatmap de correlaciones generado y guardado en {file_path}"

def plot_time_series(_):
    """Genera una serie temporal de la cantidad de viajes por día (si existe columna Date)."""
    if "Date" not in df.columns:
        return "No existe columna Date."
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    ts = df.groupby(df["Date"].dt.date).size()

    plt.figure(figsize=(14,6))
    ts.plot(kind="line", marker="o", alpha=0.7)
    plt.title("Cantidad de viajes por día", fontsize=16)
    plt.xlabel("Fecha", fontsize=12)
    plt.ylabel("Cantidad de viajes", fontsize=12)
    plt.grid(True, alpha=0.5)

    file_path = "./Outputs/time_series.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Serie temporal generada y guardada en {file_path}"

def plot_payment_method_distribution(_):
    """Genera un gráfico de barras de los métodos de pago ordenados por frecuencia."""
    if "Payment Method" not in df.columns:
        return "No existe columna Payment Method."
    
    counts = df["Payment Method"].value_counts().sort_values(ascending=False)
    
    plt.figure(figsize=(10,6))
    counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Métodos de Pago más frecuentes", fontsize=16)
    plt.xlabel("Método de Pago", fontsize=12)
    plt.ylabel("Frecuencia", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", alpha=0.5)

    file_path = "./Outputs/payment_method_distribution.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Gráfico de métodos de pago generado y guardado en {file_path}"

def plot_booking_value_by_vehicle_type(_):
    """Genera un boxplot de Booking Value según Vehicle Type."""
    if "Booking Value" not in df.columns or "Vehicle Type" not in df.columns:
        return "No existen las columnas necesarias (Booking Value, Vehicle Type)."
    
    plt.figure(figsize=(12,6))
    import seaborn as sns
    sns.boxplot(data=df, x="Vehicle Type", y="Booking Value", palette="Set2")
    plt.title("Distribución de Booking Value por tipo de vehículo", fontsize=16)
    plt.xlabel("Tipo de Vehículo", fontsize=12)
    plt.ylabel("Booking Value", fontsize=12)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.5)

    file_path = "./Outputs/booking_value_by_vehicle_type.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.show()
    return f"✅ Boxplot generado y guardado en {file_path}"

tools = [
    Tool(name="get_summary", func=get_summary, description="Muestra un resumen estadístico del dataset"),
    Tool(name="get_columns", func=get_columns, description="Muestra las columnas del dataset"),
    Tool(name="get_missing_values", func=get_missing_values, description="Muestra los valores nulos en el dataset"),
    Tool(name="get_dtypes_and_uniques", func=get_dtypes_and_uniques, description="Muestra tipos de datos y cantidad de valores únicos por columna"),
    Tool(name="get_categorical_distribution", func=get_categorical_distribution, description="Muestra distribución de valores en una columna categórica"),
    Tool(name="get_numeric_dispersion", func=get_numeric_dispersion, description="Muestra rango, varianza y desviación estándar de columnas numéricas"),
    Tool(name="get_correlations", func=get_correlations, description="Muestra correlaciones entre variables numéricas"),
    Tool(name="detect_outliers", func=detect_outliers, description="Detecta valores atípicos en una columna numérica"),
    Tool(name="get_time_series_summary", func=get_time_series_summary, description="Muestra cantidad de viajes por fecha"),
    Tool(name="plot_histogram", func=plot_histogram, description="Genera un histograma de una columna numérica"),
    Tool(name="plot_correlation_heatmap", func=plot_correlation_heatmap, description="Genera un heatmap de correlaciones entre variables numéricas"),
    Tool(name="plot_time_series", func=plot_time_series, description="Genera una serie temporal de cantidad de viajes por día"),
    Tool(name="plot_payment_method_distribution", func=plot_payment_method_distribution, description="Genera un gráfico de barras de métodos de pago ordenados por frecuencia"),
    Tool(name="plot_booking_value_by_vehicle_type", func=plot_booking_value_by_vehicle_type, description="Genera un boxplot de Booking Value por tipo de vehículo"),
    Tool(
        name="Python_Interpreter",
        func=run_python_with_df,
        description="Ejecuta código Python con acceso al DataFrame `df` cargado desde Excel. Usa este df para limpiar datos, convertir columnas y generar gráficos."
    ),
]

# Mapeo para invocarlos fácilmente
tool_dict = {t.name: t for t in tools}

# ======================
# 4. Definir estado del grafo
# ======================
class AgentState(TypedDict):
    query: str
    action: str
    result: Any
    history: List[str]

# ======================
# 5. Nodos del grafo
# ======================
def node_clasificar(state: AgentState):
    """El LLM decide qué acción tomar: usar un tool disponible o Python_Interpreter"""

    # Listado dinámico de tools disponibles
    available_tools = list(tool_dict.keys())
    tools_str = ", ".join([t for t in available_tools if t != "Python_Interpreter"])

    prompt = f"""
Eres un asistente de análisis de datos. Analiza esta consulta del usuario y decide qué acción tomar.

Pregunta del usuario: {state['query']}

Tools disponibles:
{tools_str}

Descripción de tools:
- get_summary: resumen estadístico del dataset
- get_columns: lista de columnas
- get_missing_values: valores nulos
- plot_histogram: histograma de una columna numérica
- plot_correlation_heatmap: mapa de correlaciones
- etc.

Reglas de decisión:
1. Si la consulta puede resolverse directamente con un tool existente, úsalo.
2. Si necesitas mostrar datos específicos (como "primeros 5 registros", "filtrar por condición", "operaciones personalizadas"), usa "Python_Interpreter".
3. Para consultas complejas o que requieren código personalizado, usa "Python_Interpreter".

Formato de salida:
Thought: <tu razonamiento>
Action: <nombre_del_tool_o_Python_Interpreter>
"""

    # Respuesta del LLM
    response = llm.invoke(prompt).content.strip()

    # Separar Thought y Action
    thought, action = "", ""
    for line in response.splitlines():
        if line.lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip()

    # Si no detectó bien, fallback a todo el texto como action
    if not action:
        action = response.splitlines()[-1].strip()

    # Guardar en el estado
    state["thought"] = thought
    state["action"] = action

    # Mostrar en consola
    print(f"\n🧠 Thought: {thought}")
    print(f"➡️ Action elegido: {action}")

    return state

def node_tool_or_python(state: AgentState):
    """Ejecuta el tool o el intérprete según corresponda"""
    action = state["action"]
    print(f"⚙️ Ejecutando acción: {action}")

    if action in tool_dict and action != "Python_Interpreter":
        tool_func = tool_dict[action].func

        # Si la función requiere argumentos (ej: plot_histogram), tratamos de extraerlos del query
        import inspect
        sig = inspect.signature(tool_func)
        params = sig.parameters

        if len(params) == 0:
            # Tool sin argumentos -> se llama directo
            result = tool_func(None)
        else:
            # Tool con 1 argumento -> intentamos extraer columna del query
            query = state["query"]
            column = None
            for col in df.columns:
                if col.lower() in query.lower():
                    column = col
                    break
            if column:
                result = tool_func(column)
            else:
                result = tool_func(query)  # fallback: pasar el query entero

    elif action == "Python_Interpreter":
        code_generation_prompt = f"""
Convierte esta consulta en código Python ejecutable:

Consulta: {state['query']}

Contexto:
- El DataFrame se llama 'df' y YA está cargado.
- ❌ No crear nuevos DataFrames ni datos de ejemplo.
- ✅ Usa solo df existente.
Solo genera un gráfico si el usuario lo pide explícitamente. Si pide ver datos, responde con el método adecuado sin graficar. No verifiques con código usando if, toma tu la decisión.
Solo si el usuario pide un gráfico ten en cuenta lo siguiente:
- La columna temporal puede ser: datetime.time, datetime.datetime o string.
- Antes de operar, **inspecciona el tipo de la columna temporal**.
- Si la columna es solo hora (`datetime.time`) y existe columna 'Date', **combínalas** para crear un datetime completo:
    `df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))`
    `df['Hour'] = df['DateTime'].dt.hour`
- Para datetime.datetime: `df['Hour'] = df['Time'].dt.hour`
- Para string convertida a datetime: primero `df['DateTime'] = pd.to_datetime(df['Time'])`, luego `df['Hour'] = df['DateTime'].dt.hour`
- Si existe columna de estado de reserva:
    - Asegúrate de convertirla a tipo categórico:  
      `df['Booking Status'] = df['Booking Status'].astype('category')`
    - Usar este campo como color en los gráficos.
- Si no hay columna explícita de conteo de reservas, calcularla con `groupby + size()`.
- **No manejar errores con try/except**: deja que cualquier error crítico llegue al nodo de validación.
- Antes de llamar a plt.show(), **guardar el gráfico** en la carpeta ./Outputs con un nombre representativo (tu dale un nombre representativo al archivo):
    `plt.savefig(f"./Outputs/name_this_graph.png", dpi=300, bbox_inches="tight")`
- Recuerda ejecutar a plt.show() una vez guardado el gráfico en la carpeta.

Instrucciones (en el caso de que el usuario pida realizar un gráfico UNICAMENTE):
1. Detectar y convertir correctamente la columna temporal, fusionando fecha y hora si aplica.
2. Convertir la columna de estado de reserva en categoría si existe.
3. Calcular la cantidad de reservas por hora (y por estado de reserva si existe).
4. Generar un gráfico de dispersión: x = hora, y = cantidad de reservas, color = estado de reserva (si aplica).
5. Código **robusto y ejecutable** para cualquier dataset similar, pero **sin atraparlo en try/except**.

Responde SOLO con código Python ejecutable, sin explicaciones.
"""
        try:
            python_code = llm.invoke(code_generation_prompt).content.strip()

            # --- Limpiar formato de markdown ---
            if python_code.startswith("```"):
                python_code = python_code.strip("`")  # elimina los backticks
                if python_code.lower().startswith("python"):
                    python_code = python_code[len("python"):].strip()
                # También elimina un posible bloque de cierre al final
                if python_code.endswith("```"):
                    python_code = python_code[:-3].strip()

            print("🔍 Código generado por el LLM:\n", python_code)
            result = run_python_with_df(python_code)
            print(f"{result}")
        except Exception as e:
            result = f"Error al generar o ejecutar código: {str(e)}"
    else:
        result = f"Acción '{action}' no reconocida."

    state["result"] = result
    return state

def node_validar_resultado(state: AgentState):
    """
    Valida si el resultado necesita reintento.
    Maneja errores de conversión, tipos de datos y problemas comunes.
    """
    result = state["result"]

    # Lista expandida de errores que requieren reintento
    error_patterns = [
        "no es numérica",
        "typeerror",
        "valueerror",
        "convertible to datetime",
        "cannot convert",
        "invalid literal",
        "keyerror",
        "attributeerror",
        "unsupported operand",
        "no numeric data to plot"
    ]

    needs_retry = any(pattern in str(result).lower() for pattern in error_patterns)

    if isinstance(result, str) and needs_retry:
        print("⚠️ Validación: error detectado, reintentando con estrategia mejorada...")

        retry_prompt = f"""
Convierte esta consulta en código Python ejecutable:

Consulta: {state['query']}

Contexto:
- El DataFrame se llama 'df' y YA está cargado.
- ❌ No crear nuevos DataFrames ni datos de ejemplo.
- ✅ Usa solo df existente.
Solo genera un gráfico si el usuario lo pide explícitamente. Si pide ver datos, responde con el método adecuado sin graficar. No verifiques con código usando if, toma tu la decisión.
Solo si el usuario pide un gráfico ten en cuenta lo siguiente:
- La columna temporal puede ser: datetime.time, datetime.datetime o string.
- Antes de operar, **inspecciona el tipo de la columna temporal**.
- Si la columna es solo hora (`datetime.time`) y existe columna 'Date', **combínalas** para crear un datetime completo:
    `df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))`
    `df['Hour'] = df['DateTime'].dt.hour`
- Para datetime.datetime: `df['Hour'] = df['Time'].dt.hour`
- Para string convertida a datetime: primero `df['DateTime'] = pd.to_datetime(df['Time'])`, luego `df['Hour'] = df['DateTime'].dt.hour`
- Si existe columna de estado de reserva:
    - Asegúrate de convertirla a tipo categórico:  
      `df['Booking Status'] = df['Booking Status'].astype('category')`
    - Usar este campo como color en los gráficos.
- Si no hay columna explícita de conteo de reservas, calcularla con `groupby + size()`.
- **No manejar errores con try/except**: deja que cualquier error crítico llegue al nodo de validación.
- Antes de llamar a plt.show(), **guardar el gráfico** en la carpeta ./Outputs con un nombre representativo (tu dale un nombre representativo al archivo):
    `plt.savefig(f"./Outputs/name_this_graph.png", dpi=300, bbox_inches="tight")`
- Recuerda ejecutar a plt.show() una vez guardado el gráfico en la carpeta.

Instrucciones (en el caso de que el usuario pida realizar un gráfico UNICAMENTE):
1. Detectar y convertir correctamente la columna temporal, fusionando fecha y hora si aplica.
2. Convertir la columna de estado de reserva en categoría si existe.
3. Calcular la cantidad de reservas por hora (y por estado de reserva si existe).
4. Generar un gráfico de dispersión: x = hora, y = cantidad de reservas, color = estado de reserva (si aplica).
5. Código **robusto y ejecutable** para cualquier dataset similar, pero **sin atraparlo en try/except**.

Responde SOLO con código Python ejecutable, sin explicaciones.
"""
        try:
            python_code = llm.invoke(retry_prompt).content.strip()

            # --- Limpiar formato de markdown ---
            if python_code.startswith("```"):
                python_code = python_code.strip("`")
                if python_code.lower().startswith("python"):
                    python_code = python_code[len("python"):].strip()
                if python_code.endswith("```"):
                    python_code = python_code[:-3].strip()

            print("🔍 Código corregido:\n", python_code)
            result = run_python_with_df(python_code)
            state["result"] = result
        except Exception as e:
            state["result"] = f"Error al reintentar con estrategia mejorada: {str(e)}"

    return state

def node_responder(state: AgentState):
    """Genera la respuesta final para el usuario"""
    prompt = f"""
Eres un asistente en español.
Pregunta del usuario: {state['query']}
Resultado técnico: {state['result']}

Redacta una respuesta clara, en español, explicando el resultado.
"""
    respuesta = llm.invoke(prompt).content
    print("\n🤖 Respuesta:", respuesta)
    state["history"].append(respuesta)
    return state

# ======================
# 6. Construir el grafo
# ======================
graph = StateGraph(AgentState)

graph.add_node("clasificar", node_clasificar)
graph.add_node("ejecutar", node_tool_or_python)
graph.add_node("validar", node_validar_resultado)
graph.add_node("responder", node_responder)

graph.set_entry_point("clasificar")
graph.add_edge("clasificar", "ejecutar")
graph.add_edge("ejecutar", "validar")
graph.add_edge("validar", "responder")
graph.add_edge("responder", END)

app = graph.compile()

# ======================
# 7. Loop de consola
# ======================
while True:
    query = input("Pregunta sobre el dataset (o 'salir'): ")
    if query.lower() == "salir":
        break
    state = {"query": query, "action": "", "result": None, "history": []}
    app.invoke(state)



