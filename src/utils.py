import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import df
from typing import Optional, List

# ======================
# Función para ejecutar código Python con df
# ======================
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

# ======================
# Prompt builder
# ======================
def build_code_prompt(query: str, execution_history: List[dict] = None, df_info: dict = None):
    """
    Genera un prompt contextual que incluye historial de errores y información del DataFrame.
    """
    
    # Información básica del DataFrame
    if df_info is None:
        df_info = {
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "shape": df.shape,
            "sample": df.head(2).to_dict()
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
        5. Para columnas tipo 'time', usa df['columna'].astype(str) o métodos específicos

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
