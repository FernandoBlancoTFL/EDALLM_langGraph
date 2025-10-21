from typing import List

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
        3. Para gráficos:
        - NUNCA uses plt.show() - está PROHIBIDO
        - SIEMPRE guarda con plt.savefig() en './src/outputs/'
        - Usa nombres descriptivos sin timestamp: 'histogram_edad.png', 'scatter_ventas.png', etc.
        - Después de guardar, usa plt.close() para liberar memoria
        - SIEMPRE retorna la ruta del archivo guardado como última línea
        - Ejemplo correcto:
            plt.figure(figsize=(10,6))
            df['edad'].hist()
            plt.savefig('./src/outputs/histogram_edad.png', dpi=300, bbox_inches='tight')
            plt.close()
            './src/outputs/histogram_edad.png'
        4. Si trabajas con columnas de tiempo, verifica su tipo primero
        5. NO generes comentarios, Type hints o anotaciones en el código. Responde SOLO con código Python ejecutable, sin explicaciones ni markdown

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
        
    base_prompt += """

        REGLAS CRÍTICAS DE FORMATO:
        1. NO definas funciones - escribe código directo
        2. NO uses if __name__ == '__main__'
        3. NO incluyas comentarios sobre cargar datos
        4. El DataFrame 'df' YA ESTÁ DISPONIBLE
        5. Todos los módulos (pd, plt, sns, os) YA ESTÁN IMPORTADOS
        6. PROHIBIDO usar plt.show() - solo plt.savefig() + plt.close()

        CÓDIGO DEBE SER EJECUTABLE DIRECTAMENTE, ejemplo:
        ✅ CORRECTO (para gráficos):
        plt.figure(figsize=(10,6))
        df['columna'].hist()
        plt.savefig('./src/outputs/histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
        './src/outputs/histogram.png'

        ✅ CORRECTO (para análisis sin gráfico):
        df['columna'].describe()

        ❌ INCORRECTO:
        plt.figure(figsize=(10,6))
        df['columna'].hist()
        plt.show()  # ❌ PROHIBIDO
    """
    
    return base_prompt