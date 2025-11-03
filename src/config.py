import os
from dotenv import load_dotenv

load_dotenv()

# Configuración principal
ENABLE_AUTO_SAVE_TO_DB = False
SINGLE_USER_THREAD_ID = "single_user_persistent_thread"
SINGLE_USER_ID = "default_user"

# Configuración de API y directorios
API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# Suprimir warnings de Google Cloud
import warnings
warnings.filterwarnings('ignore', message='.*ALTS.*')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''

# Convertir el mal formato de fechas a números decimales de los documentos subidos
ENABLE_DATE_FORMAT = True
