import os
import uuid
import pandas as pd
import psycopg
from typing import List, Tuple, Optional
from datetime import datetime
from database import load_db_config, data_connection
from utils import calculate_file_hash
from dataset_manager import (
    create_dataset_table_from_df, 
    insert_dataframe_to_table,
    generate_semantic_description_with_llm,
    list_stored_tables,
    get_dataset_table_info_by_name
)

class DocumentService:
    """Servicio para gestión de documentos/datasets"""
    
    def __init__(self):
        self.uploads_dir = "./src/data"
        self.allowed_extensions = {'.xlsx', '.xls', '.csv'}
        
        # Crear directorio si no existe
        os.makedirs(self.uploads_dir, exist_ok=True)

    def _check_duplicate_by_hash(self, file_hash: str, conn) -> Optional[dict]:
        try:
            with conn.cursor() as cursor:
                # Primero verificar que la tabla existe
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'document_registry'
                    )
                """)
                table_exists = cursor.fetchone()[0]
                
                if not table_exists:
                    print("⚠️ Tabla document_registry no existe, creándola...")
                    from database import create_document_registry_table
                    if not create_document_registry_table():
                        print("❌ No se pudo crear document_registry")
                        return None
                    print("✅ Tabla document_registry creada")
                
                cursor.execute("""
                    SELECT file_id, original_filename, table_name, 
                        row_count, column_count, upload_date
                    FROM document_registry
                    WHERE file_hash = %s
                """, (file_hash,))
                
                result = cursor.fetchone()
                
                if result:
                    return {
                        "file_id": result[0],
                        "original_filename": result[1],
                        "table_name": result[2],
                        "row_count": result[3],
                        "column_count": result[4],
                        "upload_date": result[5].isoformat() if result[5] else None
                    }
                
                return None
                
        except Exception as e:
            print(f"⚠️ Error verificando duplicados: {e}")
            conn.rollback()  # Importante: hacer rollback si hay error
            return None
    
    def _register_document(self, file_id: str, file_hash: str, filename: str, 
                          table_name: str, file_size: int, row_count: int, 
                          column_count: int, semantic_description: str, conn):
        """
        Registra un nuevo documento en la tabla document_registry.
        
        Args:
            file_id: ID único del archivo
            file_hash: Hash SHA256 del archivo
            filename: Nombre original del archivo
            table_name: Nombre de la tabla en PostgreSQL
            file_size: Tamaño del archivo en bytes
            row_count: Cantidad de filas
            column_count: Cantidad de columnas
            semantic_description: Descripción generada por LLM
            conn: Conexión a la base de datos
        """
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO document_registry 
                    (file_id, file_hash, original_filename, table_name, 
                     file_size_bytes, row_count, column_count, semantic_description)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (file_id, file_hash, filename, table_name, 
                      file_size, row_count, column_count, semantic_description))
                
                conn.commit()
                print(f"📝 Documento registrado en document_registry: {file_id}")
                
        except Exception as e:
            print(f"⚠️ Error registrando documento: {e}")
            raise
    
    def _generate_file_id(self) -> str:
        """Genera un ID único para el archivo"""
        return str(uuid.uuid4())[:8]
    
    def _generate_table_name(self, filename: str, file_id: str) -> str:
        """
        Genera un nombre de tabla válido para PostgreSQL.
        Formato: nombre_base_fileid
        """
        # Extraer nombre sin extensión
        base_name = os.path.splitext(filename)[0]
        
        # Limpiar caracteres especiales
        import re
        clean_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name)
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = clean_name.strip('_').lower()
        
        # Agregar file_id para unicidad
        table_name = f"{clean_name}_{file_id}"
        
        # Limitar longitud
        if len(table_name) > 50:
            table_name = table_name[:50]
        
        return table_name
    
    def _validate_file(self, filename: str) -> Tuple[bool, Optional[str]]:
        """
        Valida que el archivo tenga una extensión permitida.
        Returns: (is_valid, error_message)
        """
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.allowed_extensions:
            return False, f"Extensión no permitida. Solo se aceptan: {', '.join(self.allowed_extensions)}"
        return True, None
    
    async def upload_document(self, file_content: bytes, filename: str) -> dict:
        """
        Procesa y almacena un documento en la BD.
        MODIFICADO: Elimina archivo temporal después de procesarlo.
        """
        # Validar archivo
        is_valid, error = self._validate_file(filename)
        if not is_valid:
            raise ValueError(error)
        
        # Calcular hash del archivo
        file_hash = calculate_file_hash(file_content, algorithm='sha256')
        file_size = len(file_content)
        
        print(f"🔐 Hash SHA256 calculado: {file_hash[:16]}...")
        
        # Obtener conexión
        conn = data_connection
        if conn is None:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            should_close = True
        else:
            should_close = False
        
        temp_filepath = None  # NUEVO: Para rastrear el archivo temporal
        
        try:
            # Verificar si ya existe un archivo con el mismo hash
            duplicate = self._check_duplicate_by_hash(file_hash, conn)
            
            if duplicate:
                print(f"⚠️ Archivo duplicado detectado: {duplicate['original_filename']}")
                return {
                    "file_id": duplicate["file_id"],
                    "filename": filename,
                    "original_filename": duplicate["original_filename"],
                    "table_name": duplicate["table_name"],
                    "rows_imported": duplicate["row_count"],
                    "columns": duplicate["column_count"],
                    "is_duplicate": True,
                    "duplicate_of": duplicate["original_filename"],
                    "upload_date": duplicate["upload_date"]
                }
            
            # Si no es duplicado, proceder con la carga normal
            print(f"✅ Archivo nuevo detectado, procediendo con la carga...")
            
            # Generar ID único
            file_id = self._generate_file_id()
            
            # Generar nombre de tabla
            table_name = self._generate_table_name(filename, file_id)
            
            # MODIFICADO: Guardar archivo temporal solo para lectura
            temp_filepath = os.path.join(self.uploads_dir, f"{file_id}_{filename}")
            with open(temp_filepath, 'wb') as f:
                f.write(file_content)
            
            print(f"📁 Archivo temporal creado: {temp_filepath}")
            
            try:
                # Leer archivo según extensión
                ext = os.path.splitext(filename)[1].lower()
                
                if ext in ['.xlsx', '.xls']:
                    df = pd.read_excel(temp_filepath)
                elif ext == '.csv':
                    df = pd.read_csv(temp_filepath)
                else:
                    raise ValueError(f"Extensión no soportada: {ext}")
                
                print(f"📊 Archivo leído: {len(df)} filas, {len(df.columns)} columnas")
                
                # Generar descripción semántica con LLM
                print(f"🤖 Generando descripción semántica con LLM...")
                semantic_description = generate_semantic_description_with_llm(
                    df, 
                    table_name, 
                    filename
                )
                
                # Crear tabla en BD
                success, column_mapping = create_dataset_table_from_df(
                    df, 
                    conn, 
                    table_name, 
                    "public",
                    semantic_description
                )
                
                if not success:
                    raise Exception("Error al crear tabla en la base de datos")
                
                # Insertar datos
                insert_success = insert_dataframe_to_table(
                    df, 
                    column_mapping, 
                    conn, 
                    table_name, 
                    "public",
                    semantic_description
                )
                
                if not insert_success:
                    raise Exception("Error al insertar datos en la tabla")
                
                # Registrar documento en document_registry
                self._register_document(
                    file_id, 
                    file_hash, 
                    filename, 
                    table_name, 
                    file_size,
                    len(df), 
                    len(df.columns),
                    semantic_description,
                    conn
                )
                
                print(f"✅ Documento cargado exitosamente: {table_name}")
                
                return {
                    "file_id": file_id,
                    "filename": filename,
                    "table_name": table_name,
                    "rows_imported": len(df),
                    "columns": len(df.columns),
                    "semantic_description": semantic_description,
                    "is_duplicate": False,
                    "file_hash": file_hash[:16] + "..."
                }
                
            finally:
                # NUEVO: Eliminar archivo temporal después de procesarlo
                if temp_filepath and os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                        print(f"🗑️ Archivo temporal eliminado: {temp_filepath}")
                    except Exception as e:
                        print(f"⚠️ No se pudo eliminar archivo temporal: {e}")
                    
        finally:
            if should_close:
                conn.close()
    
    def list_documents(self) -> List[dict]:
        """
        Lista todos los documentos almacenados en la BD.
        Excluye tablas del sistema como document_registry.
        
        Returns:
            Lista de diccionarios con información de cada documento
        """
        conn = data_connection
        if conn is None:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            should_close = True
        else:
            should_close = False
        
        try:
            stored_tables = list_stored_tables(conn)
            documents = []
            
            # Tablas del sistema que NO son documentos de usuario
            system_tables = ['document_registry', 'checkpoints', 'checkpoint_writes']
            
            for table_name in stored_tables:
                # Saltar tablas del sistema
                if table_name in system_tables:
                    continue
                
                # Extraer file_id del nombre de tabla (último segmento después de _)
                parts = table_name.split('_')
                file_id = parts[-1] if len(parts) > 1 else "unknown"
                
                # Obtener información de la tabla
                table_info = get_dataset_table_info_by_name(table_name, conn)
                
                if table_info:
                    # Intentar obtener fecha de creación
                    try:
                        with conn.cursor() as cursor:
                            cursor.execute(f"""
                                SELECT created_at 
                                FROM public.{table_name} 
                                ORDER BY created_at DESC 
                                LIMIT 1
                            """)
                            result = cursor.fetchone()
                            created_at = result[0].isoformat() if result and result[0] else datetime.now().isoformat()
                    except:
                        created_at = datetime.now().isoformat()
                    
                    # Reconstruir nombre de archivo original (aproximado)
                    filename = table_name.replace(f"_{file_id}", "") + ".xlsx"
                    
                    documents.append({
                        "file_id": file_id,
                        "filename": filename,
                        "table_name": table_name,
                        "row_count": table_info["row_count"],
                        "column_count": len(table_info["columns"]),
                        "created_at": created_at
                    })
            
            return documents
            
        finally:
            if should_close:
                conn.close()
    
    def delete_document(self, file_id: str) -> dict:
        conn = data_connection
        if conn is None:
            db_config = load_db_config()
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
            conn = psycopg.connect(connection_string)
            should_close = True
        else:
            should_close = False
        
        try:
            # Buscar tabla que contenga el file_id
            stored_tables = list_stored_tables(conn)
            table_to_delete = None
            
            for table_name in stored_tables:
                if table_name.endswith(f"_{file_id}"):
                    table_to_delete = table_name
                    break
            
            if not table_to_delete:
                raise ValueError(f"No se encontró documento con file_id: {file_id}")
            
            # Eliminar tabla primero
            with conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS public.{table_to_delete} CASCADE")
                conn.commit()
            
            print(f"✅ Documento eliminado: {table_to_delete}")
            
            # Eliminar archivo físico si existe
            for filename in os.listdir(self.uploads_dir):
                if filename.startswith(file_id):
                    filepath = os.path.join(self.uploads_dir, filename)
                    try:
                        os.remove(filepath)
                        print(f"🗑️ Archivo eliminado: {filename}")
                    except:
                        pass
            
            # Eliminar del registro (en una transacción separada)
            try:
                with conn.cursor() as cursor:
                    # Verificar que la tabla existe antes de intentar eliminar
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_schema = 'public' 
                            AND table_name = 'document_registry'
                        )
                    """)
                    table_exists = cursor.fetchone()[0]
                    
                    if table_exists:
                        cursor.execute("DELETE FROM document_registry WHERE file_id = %s", (file_id,))
                        conn.commit()
                        print(f"📝 Registro eliminado de document_registry")
                    else:
                        print("⚠️ Tabla document_registry no existe, saltando eliminación del registro")
            except Exception as e:
                print(f"⚠️ Error al eliminar del registro (no crítico): {e}")
                conn.rollback()
            
            return {
                "file_id": file_id,
                "table_name": table_to_delete
            }
        
        finally:
            if should_close:
                conn.close()