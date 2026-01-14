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
        'dbname': os.getenv('POSTGRES_DB', 'langgraph_analysis')
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
        if hasattr(e, 'pgcode'):
            print(f"   Código de error: {e.pgcode}")
        
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
    MODIFICADO: Usa autocommit para evitar transacciones colgadas.
    """
    global data_connection
    
    print("🔧 Configurando conexión para datos...")
    
    try:
        db_config = load_db_config()
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        
        # Usar autocommit para evitar transacciones abiertas
        data_connection = psycopg.connect(connection_string, autocommit=True)
        print("✅ Conexión de datos configurada con autocommit")
        return True
        
    except Exception as e:
        print(f"❌ Error configurando conexión de datos: {e}")
        data_connection = None
        return False

def get_table_metadata_light(table_name: str):
    """
    Obtiene metadatos básicos de una tabla sin cargar datos.
    MEJORADO: Ahora maneja nombres parciales y los mapea al nombre completo.
    """
    conn = data_connection
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
        # Intentar encontrar la tabla real si se pasó un nombre parcial
        actual_table_name = table_name
        
        with conn.cursor() as cursor:
            # Verificar si la tabla existe exactamente como se pasó
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = %s
                )
            """, (table_name,))
            exists = cursor.fetchone()[0]
            
            if not exists:
                # Buscar tabla que empiece con el nombre dado
                cursor.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE %s
                    AND table_name NOT IN ('document_registry', 'checkpoints', 'checkpoint_writes')
                    LIMIT 1
                """, (table_name + '%',))
                
                result = cursor.fetchone()
                if result:
                    actual_table_name = result[0]
                    print(f"🔄 Mapeado: '{table_name}' → '{actual_table_name}'")
                else:
                    print(f"❌ No se encontró tabla que coincida con: '{table_name}'")
                    return {}
            
            # Información de columnas
            cursor.execute("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
            """, (actual_table_name,))
            
            columns_info = cursor.fetchall()

            # Excluir columnas del sistema de los metadatos
            system_columns = ['created_at', 'semantic_description']
            filtered_columns = [col for col in columns_info if col[0] not in system_columns]

            # Conteo de filas
            cursor.execute(f'SELECT COUNT(*) FROM public."{actual_table_name}"')
            row_count = cursor.fetchone()[0]

            return {
                "columns": [col[0] for col in filtered_columns],
                "dtypes": {col[0]: col[1] for col in filtered_columns},
                "row_count": row_count,
                "nullable": {col[0]: col[2] == 'YES' for col in filtered_columns},
                "actual_table_name": actual_table_name  # Incluir nombre real
            }
            
    except Exception as e:
        print(f"⚠️ Error obteniendo metadatos de {table_name}: {e}")
        return {}
    finally:
        if temp_connection and conn:
            conn.close()

def create_document_registry_table():
    """
    Crea una tabla para registrar documentos cargados y sus hashes.
    Esta tabla actúa como un registro central de todos los documentos.
    """
    global data_connection
    
    # Usar la conexión global si existe, sino crear una temporal
    conn = data_connection
    should_close = False
    
    if conn is None:
        db_config = load_db_config()
        try:
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            should_close = True
        except Exception as e:
            print(f"❌ Error conectando a la BD: {e}")
            return False
    
    try:
        with conn.cursor() as cursor:
            # Crear tabla de registro de documentos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_registry (
                    file_id VARCHAR(8) PRIMARY KEY,
                    file_hash VARCHAR(64) UNIQUE NOT NULL,
                    original_filename VARCHAR(255) NOT NULL,
                    table_name VARCHAR(100) NOT NULL,
                    file_size_bytes BIGINT NOT NULL,
                    row_count INTEGER NOT NULL,
                    column_count INTEGER NOT NULL,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    semantic_description TEXT
                )
            """)
            
            # Crear índice para búsqueda rápida por hash
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_hash 
                ON document_registry(file_hash)
            """)
            
            # Crear índice para búsqueda por nombre de tabla
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_table_name 
                ON document_registry(table_name)
            """)
            
            conn.commit()
            print("✅ Tabla document_registry creada/verificada exitosamente")
            
            # Verificar que la tabla realmente existe
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'document_registry'
                )
            """)
            exists = cursor.fetchone()[0]
            
            if exists:
                print("✅ Verificación: document_registry existe en la BD")
                return True
            else:
                print("❌ Advertencia: document_registry no se creó correctamente")
                return False
                
    except Exception as e:
        print(f"❌ Error creando tabla document_registry: {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if should_close and conn:
            conn.close()