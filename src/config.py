import os
from dotenv import load_dotenv

load_dotenv()

# Configuración principal
ENABLE_AUTO_SAVE_TO_DB = False
SINGLE_USER_THREAD_ID = "single_user_persistent_thread"
SINGLE_USER_ID = "default_user"

# Configuración de datasets
DATASETS_TO_PROCESS = [
    {
        "excel_path": "./src/data/ncr_ride_bookings.xlsx",
        "table_name": "dataset_rides",
        "table_schema": "public"
    },
    {
        "excel_path": "./src/data/crocodile_dataset.xlsx",
        "table_name": "crocodile_dataset", 
        "table_schema": "public"
    }
]

DATASET_CONFIG = DATASETS_TO_PROCESS[0]

# Configuración de API y directorios
API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")

# Suprimir warnings de Google Cloud
import warnings
warnings.filterwarnings('ignore', message='.*ALTS.*')
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GRPC_TRACE'] = ''