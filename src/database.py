import os
import sys
import psycopg
from psycopg import sql
from config import load_dotenv

load_dotenv()

# Variables globales
data_connection = None

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

def setup_data_connection():
    """
    Configura una conexión independiente para operaciones de datos.
    """
    global data_connection
    
    print("🔧 Configurando conexión para datos...")
    
    try:
        db_config = load_db_config()
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        data_connection = psycopg.connect(connection_string)
        print("✅ Conexión de datos configurada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error configurando conexión de datos: {e}")
        data_connection = None
        return False

def get_table_metadata_light(table_name: str):
    """Obtiene metadatos básicos de una tabla sin cargar datos"""
    
    conn = data_connection  # Usar la conexión global de datos
    if conn is None:
        try:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            temp_connection = True
        except:
            return {}
    else:
        temp_connection = False
    
    try:
        with conn.cursor() as cursor:
            # Información de columnas
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (table_name,))
            
            columns_info = cursor.fetchall()
            
            # Conteo de filas
            cursor.execute(f'SELECT COUNT(*) FROM public."{table_name}"')
            row_count = cursor.fetchone()[0]
            
            return {
                "columns": [col[0] for col in columns_info],
                "dtypes": {col[0]: col[1] for col in columns_info},
                "row_count": row_count,
                "nullable": {col[0]: col[2] == 'YES' for col in columns_info}
            }
            
    except Exception as e:
        print(f"⚠️ Error obteniendo metadatos de {table_name}: {e}")
        return {}
    finally:
        if temp_connection and conn:
            conn.close()