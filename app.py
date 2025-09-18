import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain_experimental.tools import PythonREPLTool  # intérprete de Python de Camel AI
from typing import TypedDict, List, Any, Optional, Optional
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Any
import sys

import psycopg
from psycopg import sql
from langgraph.checkpoint.postgres import PostgresSaver
import uuid
import json

def clean_data_for_json(data):
    """Limpia datos para que sean serializables en JSON"""
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

# Suprimir warnings de Google Cloud
import warnings
warnings.filterwarnings('ignore', message='.*ALTS.*')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# ======================
# Thread ID persistente para memoria a corto plazo
# ======================
PERSISTENT_THREAD_ID = "persistent_chat_session"

# ======================
# 0. Configuración y creación de BD PostgreSQL
# ======================

def load_db_config():
    """Carga la configuración de la base de datos desde variables de entorno"""
    load_dotenv()
    
    db_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD'),
        'database': os.getenv('POSTGRES_DB', 'langgraph_analysis')
    }
    
    # Verificar que las credenciales estén configuradas
    if not db_config['password']:
        print("❌ Error: POSTGRES_PASSWORD no está configurada en el archivo .env")
        sys.exit(1)
    
    return db_config

def database_exists(cursor, db_name):
    """Verifica si una base de datos existe"""
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", 
        (db_name,)
    )
    return cursor.fetchone() is not None

def create_database_if_not_exists():
    """
    Crea la base de datos si no existe.
    Retorna True si la BD se creó o ya existía, False en caso de error.
    """
    db_config = load_db_config()
    target_db = db_config['database']
    
    print(f"🔍 Verificando existencia de base de datos: {target_db}")
    
    # Conectar a PostgreSQL usando la BD por defecto 'postgres'
    try:
        with psycopg.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            dbname='postgres',  # Conectar a la BD por defecto
            autocommit=True
        ) as conn:
            with conn.cursor() as cursor:
                # Verificar si la base de datos existe
                if database_exists(cursor, target_db):
                    print(f"✅ Base de datos '{target_db}' ya existe")
                    return True
                else:
                    print(f"📝 Base de datos '{target_db}' no existe. Creando...")
                    
                    # Crear la base de datos
                    cursor.execute(sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(target_db)
                    ))
                    
                    print(f"✅ Base de datos '{target_db}' creada exitosamente")
                    return True
            
    except psycopg.Error as e:
        print(f"❌ Error al gestionar la base de datos PostgreSQL:")
        print(f"   Código de error: {e.pgcode}")
        print(f"   Mensaje: {e.pgerror}")
        
        # Errores comunes y sugerencias
        if "authentication failed" in str(e).lower():
            print("   💡 Sugerencia: Verifica las credenciales en el archivo .env")
        elif "connection refused" in str(e).lower():
            print("   💡 Sugerencia: Verifica que PostgreSQL esté ejecutándose")
        elif "permission denied" in str(e).lower():
            print("   💡 Sugerencia: El usuario necesita permisos CREATEDB")
        
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def test_target_database_connection():
    """Prueba la conexión a la base de datos objetivo"""
    db_config = load_db_config()
    
    try:
        with psycopg.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            dbname=db_config['database']
        ) as conn:
            pass  # Conexión exitosa
        print(f"✅ Conexión exitosa a la base de datos '{db_config['database']}'")
        return True
    except psycopg.Error as e:
        print(f"❌ Error al conectar a la base de datos objetivo: {e}")
        return False

# ======================
# Funciones de gestión de historial persistente
# ======================

def check_thread_exists():
    """Verifica si el thread tiene historial en la tabla personalizada"""
    if checkpoint_saver is None:
        print("🔍 Debug - checkpoint_saver es None")
        return False
    
    try:
        with checkpoint_saver.conn.cursor() as cursor:
            # Primero verificar si la tabla existe
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'conversation_memory'
                )
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print("🔍 Debug - Tabla conversation_memory no existe aún")
                return False
            
            # Si la tabla existe, consultar el historial
            cursor.execute(
                "SELECT total_queries FROM conversation_memory WHERE thread_id = %s",
                (PERSISTENT_THREAD_ID,)
            )
            result = cursor.fetchone()
            
            if result:
                total_queries = result[0]
                print(f"🔍 Debug - Conversación encontrada con {total_queries} consultas")
                return True
            else:
                print(f"🔍 Debug - No hay historial para thread {PERSISTENT_THREAD_ID}")
                return False
                
    except Exception as e:
        print(f"⚠️ Error verificando historial: {e}")
        return False

def load_previous_context():
    """Carga el historial desde la tabla personalizada"""
    if checkpoint_saver is None:
        print("🔍 Debug - checkpoint_saver es None en load_previous_context")
        return None
    
    try:
        with checkpoint_saver.conn.cursor() as cursor:
            # Verificar si la tabla existe primero
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'conversation_memory'
                )
            """)
            
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print("🔍 Debug - Tabla conversation_memory no existe, no hay contexto previo")
                return None
            
            cursor.execute(
                "SELECT conversation_data FROM conversation_memory WHERE thread_id = %s",
                (PERSISTENT_THREAD_ID,)
            )
            result = cursor.fetchone()
            
            if result:
                conversation_data = result[0]  # JSONB se deserializa automáticamente
                print("🔍 Debug - Contexto cargado desde tabla personalizada")
                return conversation_data
            else:
                print("🔍 Debug - No hay contexto guardado")
                return None
                
    except Exception as e:
        print(f"⚠️ Error cargando contexto: {e}")
        return None

def get_conversation_summary():
    """Obtener resumen del historial de conversación actual"""
    previous_context = load_previous_context()
    if not previous_context:
        return "No hay historial previo."
    
    # Verificar si previous_context es un diccionario antes de usar .get()
    if not isinstance(previous_context, dict):
        return "No hay historial previo válido."
    
    conversation_history = previous_context.get("conversation_history", [])
    if not conversation_history:
        return "No hay consultas previas en el historial."
    
    total_queries = len(conversation_history)
    last_query = conversation_history[-1].get("query", "N/A") if conversation_history else "N/A"
    
    return f"Historial: {total_queries} consultas previas. Última consulta: '{last_query[:50]}...'"

def save_conversation_state(state, config):
    """Guarda el historial de conversación directamente en PostgreSQL"""
    if checkpoint_saver is None:
        print("⚠️ No se puede guardar: checkpoint_saver es None")
        return False
    
    try:
        # Guardar directamente en una tabla personalizada
        conversation_history = state.get("conversation_history", [])
        if not conversation_history:
            print("⚠️ No hay historial de conversación para guardar")
            return False
        
        # Usar la conexión del checkpoint_saver
        with checkpoint_saver.conn.cursor() as cursor:
            # Crear tabla si no existe
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory (
                    thread_id VARCHAR(255) PRIMARY KEY,
                    conversation_data JSONB,
                    total_queries INTEGER,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Preparar datos para guardar
            conversation_data = {
                "conversation_history": conversation_history,
                "session_start_time": state.get("session_start_time", pd.Timestamp.now().isoformat()),
                "total_queries": state.get("total_queries", 0),
                "df_info": clean_data_for_json(state.get("df_info", {}))
            }
            
            # Insertar o actualizar
            cursor.execute("""
                INSERT INTO conversation_memory (thread_id, conversation_data, total_queries)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_id) 
                DO UPDATE SET 
                    conversation_data = EXCLUDED.conversation_data,
                    total_queries = EXCLUDED.total_queries,
                    last_updated = CURRENT_TIMESTAMP
            """, (
                config["configurable"]["thread_id"],
                json.dumps(conversation_data, default=str),
                state.get("total_queries", 0)
            ))
            
            checkpoint_saver.conn.commit()
            
        print("💾 Historial guardado en tabla personalizada")
        return True
        
    except Exception as e:
        print(f"⚠️ Error guardando en BD personalizada: {e}")
        return False
    
def verify_checkpoint_saved(config):
    """Verifica que el checkpoint se guardó correctamente"""
    if checkpoint_saver is None:
        return False
    
    try:
        checkpoints_list = list(checkpoint_saver.list(config))
        count = len(checkpoints_list)
        print(f"🔍 Verificación: {count} checkpoints en BD")
        
        if count > 0:
            latest = checkpoints_list[0]
            if hasattr(latest, 'checkpoint') and latest.checkpoint:
                channel_values = latest.checkpoint.get("channel_values", {})
                conversations = channel_values.get("conversation_history", [])
                print(f"🔍 Conversaciones guardadas: {len(conversations)}")
                return True
        
        return False
        
    except Exception as e:
        print(f"⚠️ Error verificando checkpoint: {e}")
        return False

# ======================
# 1. Configuración inicial
# ======================
load_dotenv() # Cargar variables de entorno (.env debe contener GOOGLE_API_KEY)

# Crear BD antes de continuar
print("🚀 Inicializando sistema de análisis de datos...")
if not create_database_if_not_exists():
    print("❌ No se pudo crear o acceder a la base de datos. Terminando aplicación.")
    sys.exit(1)

# Probar conexión a la BD objetivo
if not test_target_database_connection():
    print("❌ No se pudo conectar a la base de datos objetivo. Terminando aplicación.")
    sys.exit(1)

api_key = os.getenv("GOOGLE_API_KEY")
os.makedirs("./outputs", exist_ok=True) # Crear carpeta para guardar gráficos si no existe

# Cargar dataset con pandas
df = pd.read_excel("./Data/ncr_ride_bookings.xlsx")

# Inicializar LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)
# llm = ChatOllama(model="gemma3", temperature=0)

# ======================
# 1,5. Configurar PostgresSaver para guardar checkpoints
# ======================

def get_postgres_connection_string():
    """Genera la cadena de conexión para PostgreSQL usando psycopg3"""
    db_config = load_db_config()
    return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

# Configurar PostgresSaver correctamente
connection_string = get_postgres_connection_string()

try:
    # Método 1: Crear conexión directa usando psycopg
    import psycopg
    conn = psycopg.connect(connection_string)
    
    # Crear PostgresSaver con la conexión directa
    checkpoint_saver = PostgresSaver(conn)
    checkpoint_saver.setup()
    
    print("✅ PostgresSaver configurado exitosamente")
    
except Exception as e:
    print(f"❌ Error configurando PostgresSaver: {e}")
    print("⚠️ Continuando sin checkpoints...")
    checkpoint_saver = None

# ======================
# 2. Definir intérprete Python con acceso a df
# ======================
python_repl = PythonREPLTool()

def run_python_with_df(code: str, error_context: Optional[str] = None):
    """
    Ejecuta código Python con acceso al DataFrame `df` ya cargado.
    Incluye contexto de errores previos para mejor debugging.
    """
    local_vars = {"df": df, "pd": pd, "plt": plt, "sns": sns, "os": os}

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
            return {
                "success": False,
                "result": None,
                "error": f"Código bloqueado. Detectado intento de crear DataFrame: '{pattern}'. Usa SOLO el df existente.",
                "error_type": "prohibited_pattern"
            }

    try:
        import ast
        parsed = ast.parse(code)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            last_expr = parsed.body.pop()
            exec(compile(ast.Module(parsed.body, type_ignores=[]), filename="<ast>", mode="exec"), {}, local_vars)
            result = eval(compile(ast.Expression(last_expr.value), filename="<ast>", mode="eval"), {}, local_vars)
        else:
            exec(code, {}, local_vars)
            result = None

        return {
            "success": True,
            "result": result if result is not None else "✅ Código ejecutado con éxito.",
            "error": None,
            "error_type": None
        }
    except Exception as e:
        error_type = type(e).__name__
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "error_type": error_type
        }
    
def get_tools_summary(tools: List[Tool]) -> str:
    """Devuelve un resumen con nombre y descripción de cada tool."""
    return "\n".join([f"- {t.name}: {t.description}" for t in tools])


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

    file_path = f"./outputs/histogram_{column}.png"
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

    file_path = "./outputs/correlation_heatmap.png"
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

    file_path = "./outputs/time_series.png"
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

    file_path = "./outputs/payment_method_distribution.png"
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

    file_path = "./outputs/booking_value_by_vehicle_type.png"
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

# ======================
# Prompt unificado para Python
# ======================
def build_code_prompt(query: str, execution_history: List[dict] = None, df_info: dict = None):
    """
    Genera un prompt contextual que incluye historial de errores y información del DataFrame.
    """
    
    # Información básica del DataFrame
    if df_info is None:
        # Limpiar sample para evitar valores NaN que no se pueden serializar
        sample_clean = df.head(2).fillna("NULL").to_dict()
        
        df_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "sample": sample_clean
        }
    
    base_prompt = f"""
Eres un experto en análisis de datos con Python y pandas.

INFORMACIÓN DEL DATAFRAME:
- Columnas disponibles: {df_info['columns']}
- Tipos de datos: {', '.join([f"{col}: {dtype}" for col, dtype in df_info['dtypes'].items()])}
- Dimensiones: {df_info['shape']}

REGLAS IMPORTANTES:
1. Usa EXCLUSIVAMENTE el DataFrame 'df' que ya está cargado
2. NO crees ni simules nuevos DataFrames ni datos de ejemplo
3. Para gráficos, guarda en './outputs/' y usa plt.show()
4. Si trabajas con columnas de tiempo, verifica su tipo primero

TAREA: {query}
"""

    # Agregar historial de errores si existe
    if execution_history:
        base_prompt += "\n\nHISTORIAL DE INTENTOS PREVIOS:\n"
        for i, attempt in enumerate(execution_history, 1):
            if not attempt['success']:
                base_prompt += f"""
Intento {i}:
Código: {attempt.get('code', 'N/A')}
Error: {attempt['error']} (Tipo: {attempt['error_type']})
"""
        
    base_prompt += "\n⚠️ IMPORTANTE: Analiza los errores anteriores y genera un código DIFERENTE que los evite. Solo genera un gráfico UNICAMENTE si el usuario lo pide.\n"
    base_prompt += "\nResponde SOLO con código Python ejecutable, sin explicaciones ni markdown:"
    
    return base_prompt

# Mapeo para invocarlos fácilmente
tool_dict = {t.name: t for t in tools}

# ======================
# 4. Definir estado del grafo
# ======================
class AgentState(TypedDict):
    query: str
    action: str
    result: Any
    thought: str
    history: List[str]
    execution_history: List[dict]  # Nuevo: historial detallado de ejecuciones
    iteration_count: int           # Nuevo: contador de iteraciones
    max_iterations: int           # Nuevo: límite de iteraciones
    df_info: dict                 # Nuevo: información cached del DataFrame
    success: bool                 # Nuevo: flag de éxito
    final_error: Optional[str]    # Nuevo: error final si no se pudo resolver
    thread_id: str  # NUEVO: ID único para identificar la conversación
    # NUEVOS campos para memoria persistente
    conversation_history: List[dict]  # Historial completo de consultas y respuestas
    session_start_time: str          # Timestamp de inicio de sesión
    total_queries: int               # Contador total acumulativo

# ======================
# Función para gestionar historial conversacional
# ======================
def format_conversation_context(conversation_history: List[dict], max_entries: int = 5) -> str:
    """Formatea el historial de conversación para incluir en prompts"""
    if not conversation_history:
        return ""
    
    # Tomar las últimas max_entries consultas
    recent_history = conversation_history[-max_entries:]
    
    context_parts = []
    for i, entry in enumerate(recent_history, 1):
        query = entry.get("query", "N/A")
        response = entry.get("response", "N/A")
        success = entry.get("success", False)
        
        context_parts.append(f"""
Consulta {i}: {query}
Respuesta: {response[:200]}{"..." if len(str(response)) > 200 else ""}
Estado: {'Éxito' if success else 'Error'}
""")
    
    return "\n".join(context_parts)

# ======================
# 5. Nodos del grafo
# ======================
def node_clasificar(state: AgentState):
    """El LLM decide qué acción tomar con contexto mejorado Y memoria conversacional"""
    
    # Obtener información del DataFrame si no existe
    if not state.get("df_info"):
        # Limpiar sample para evitar valores NaN
        sample_clean = df.head(2).fillna("NULL").to_dict()
        
        state["df_info"] = {
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "shape": df.shape,
            "sample": sample_clean
        }
    
    # Contexto de iteraciones previas
    iteration_context = ""
    if state["iteration_count"] > 0:
        iteration_context = f"\nEsta es la iteración #{state['iteration_count'] + 1}. Intentos previos han fallado."
        if state["execution_history"]:
            last_error = state["execution_history"][-1].get("error", "")
            iteration_context += f"\nÚltimo error: {last_error}"

    # NUEVO: Contexto conversacional
    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = f"""

HISTORIAL DE CONVERSACIÓN PREVIA:
{format_conversation_context(state["conversation_history"])}

IMPORTANTE: Tienes acceso al historial de consultas anteriores. Si el usuario hace referencia a preguntas previas ("la pregunta anterior", "lo que pregunté antes", etc.), usa este historial para responder.
"""

    tools_summary = get_tools_summary(tools)

    prompt = f"""
Eres un asistente de análisis de datos experto con memoria conversacional. Analiza esta consulta y decide la mejor acción.

CONSULTA ACTUAL: {state['query']}
DATAFRAME INFO: Columnas = {state['df_info']['columns']}, Shape = {state['df_info']['shape']}
{iteration_context}
{conversation_context}

HERRAMIENTAS DISPONIBLES:
{tools_summary}

DECISIÓN:
Analiza la consulta actual considerando el historial previo si es relevante. 
Selecciona la herramienta más adecuada. 
Si ninguna herramienta especializada es suficiente, usa Python_Interpreter.

Formato de salida:
Thought: <análisis detallado de la consulta y estrategia, considerando historial si aplica>
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
    """Genera la respuesta final CON contexto conversacional y actualiza el historial"""
    
    success = state.get("success", False)
    
    # Contexto conversacional para el prompt
    conversation_context = ""
    if state.get("conversation_history"):
        conversation_context = f"""

HISTORIAL DE CONVERSACIÓN:
{format_conversation_context(state["conversation_history"])}
"""

    if success:
        prompt = f"""
Pregunta del usuario: {state['query']}
Resultado obtenido: {state['result']}
Número de iteraciones necesarias: {state['iteration_count']}
{conversation_context}

Genera una respuesta clara y amigable en español explicando qué se logró.
Si la consulta hace referencia a conversaciones anteriores, asegúrate de conectar tu respuesta con ese contexto.
"""
    else:
        errors_summary = []
        for record in state["execution_history"]:
            if not record["success"]:
                errors_summary.append(f"- {record['error_type']}: {record['error']}")
        
        prompt = f"""
Pregunta del usuario: {state['query']}
Después de {state['iteration_count']} iteraciones, no se pudo completar la tarea.
{conversation_context}

Errores encontrados:
{chr(10).join(errors_summary)}

Genera una respuesta empática en español explicando:
1. Que se intentó resolver la consulta múltiples veces
2. Los principales problemas encontrados (en términos simples)
3. Sugerencias para el usuario (ej: verificar formato de datos, columnas, etc.)
"""

    respuesta = llm.invoke(prompt).content
    print(f"\n🤖 Respuesta Final:\n{respuesta}")
    
    # NUEVO: Actualizar historial conversacional
    new_conversation_entry = {
        "query": state["query"],
        "response": respuesta,
        "success": success,
        "timestamp": pd.Timestamp.now().isoformat(),
        "iterations_needed": state["iteration_count"]
    }
    
    # Agregar nueva entrada al historial
    if "conversation_history" not in state:
        state["conversation_history"] = []
    state["conversation_history"].append(new_conversation_entry)
    
    # Actualizar contador total
    state["total_queries"] = state.get("total_queries", 0) + 1
    
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

# ======================
# 6. Construir el grafo
# ======================
def create_graph():
    graph = StateGraph(AgentState)

    # Nodos
    graph.add_node("clasificar", node_clasificar)
    graph.add_node("ejecutar_python", node_ejecutar_python)
    graph.add_node("validar", node_validar_y_decidir)
    graph.add_node("responder", node_responder)

    # Punto de entrada
    graph.set_entry_point("clasificar")

    # Flujo principal
    graph.add_edge("clasificar", "ejecutar_python")
    graph.add_edge("ejecutar_python", "validar")
    
    # Enrutamiento condicional desde validación
    graph.add_conditional_edges("validar", route_after_validation)
    
    # Fin del grafo
    graph.add_edge("responder", END)

    # Compilar con o sin checkpointer según esté disponible
    if checkpoint_saver is not None:
        return graph.compile(checkpointer=checkpoint_saver)
    else:
        print("⚠️ Compilando grafo sin checkpoints")
        return graph.compile()

def main():
    app = create_graph()
    
    print("✅ Base de datos PostgreSQL configurada correctamente")
    print("✅ Sistema de checkpoints activado")
    print("🚀 Sistema de Análisis de Datos con Memoria Persistente")
    print("   Basado en LangGraph con historial conversacional")
    print("   Escribe 'salir' para terminar\n")
    
    # NUEVO: Cargar contexto ANTES del bucle para inicialización correcta
    print("🔍 Verificando historial existente...")
    thread_exists = check_thread_exists()
    previous_context = None
    
    if thread_exists:
        previous_context = load_previous_context()
        if previous_context:
            summary = get_conversation_summary()
            print(f"💭 Memoria encontrada: {summary}")
            print("   Continuando conversación existente...\n")
        else:
            print("⚠️ Thread existe pero no se pudo cargar contexto\n")
            thread_exists = False
    else:
        print("🆕 Iniciando nueva conversación\n")
    
    while True:
        query = input("Pregunta sobre el dataset (o 'salir'): ")
        if query.lower() == "salir":
            break

        # NUEVO: Crear estado inicial basado en contexto pre-cargado
        if previous_context and isinstance(previous_context, dict):
            # Continuar desde estado previo
            initial_state = {
                "query": query,
                "action": "",
                "result": None,
                "thought": "",
                "history": [],
                "execution_history": [],
                "iteration_count": 0,
                "max_iterations": 3,
                "df_info": previous_context.get("df_info", {}),
                "success": False,
                "final_error": None,
                "next_node": "clasificar",
                "thread_id": PERSISTENT_THREAD_ID,
                # Cargar historial existente
                "conversation_history": previous_context.get("conversation_history", []),
                "session_start_time": previous_context.get("session_start_time", str(pd.Timestamp.now())),
                "total_queries": previous_context.get("total_queries", 0)
            }
        else:
            # Nuevo estado inicial solo si NO hay contexto previo
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
                "thread_id": PERSISTENT_THREAD_ID,
                # Nuevo historial
                "conversation_history": [],
                "session_start_time": str(pd.Timestamp.now()),
                "total_queries": 0
            }

        # Configuración para el checkpointer con thread fijo
        config = {"configurable": {"thread_id": PERSISTENT_THREAD_ID}}

        print(f"\n{'='*60}")
        print(f"🔄 Procesando consulta: {query}")
        print(f"💭 Total de consultas en sesión: {initial_state['total_queries']}")
        print(f"{'='*60}")
        
        try:
            final_state = app.invoke(initial_state, config=config)
            
            # NUEVO: Forzar guardado manual del estado completo
            save_success = save_conversation_state(final_state, config)
            
            # Actualizar contexto para próximas consultas en la misma sesión
            previous_context = final_state.copy()
            thread_exists = True
            
            print(f"\n📊 RESUMEN DE EJECUCIÓN:")
            print(f"   Iteraciones totales: {final_state['iteration_count']}")
            print(f"   Éxito: {final_state.get('success', False)}")
            print(f"   Total consultas en historial: {final_state.get('total_queries', 0)}")
            if not final_state.get('success', False):
                print(f"   Error final: {final_state.get('final_error', 'N/A')}")
            print(f"💾 Checkpoint y memoria guardados automáticamente")
            
        except Exception as e:
            print(f"❌ Error crítico en el sistema: {e}")
            
        print(f"\n{'-'*60}\n")

if __name__ == "__main__":
    main()

