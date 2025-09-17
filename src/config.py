import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_ollama import ChatOllama  # Descomentar si quieres usar Ollama también

# ======================
# Configuración inicial
# ======================
load_dotenv()  # Cargar variables de entorno (.env debe contener GOOGLE_API_KEY)
api_key = os.getenv("GOOGLE_API_KEY")

os.makedirs("./outputs", exist_ok=True)  # Crear carpeta para guardar gráficos si no existe

# Cargar dataset con pandas
df = pd.read_excel("./data/ncr_ride_bookings.xlsx")

# Inicializar LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0
)
# Alternativa con Ollama:
# llm = ChatOllama(model="gemma3", temperature=0)
