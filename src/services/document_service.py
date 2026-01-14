import os
import uuid
import pandas as pd
import psycopg
from typing import List, Tuple, Optional
from datetime import datetime
from database import load_db_config, data_connection
from utils import calculate_file_hash
from config import ENABLE_DATE_FORMAT
from dataset_manager import (
    clean_mixed_date_columns,
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
        """
        Verifica si ya existe un documento con el mismo hash.
        MEJORADO: También verifica que la tabla realmente exista en la BD.
        """
        try:
            with conn.cursor() as cursor:
                # Verificar que la tabla document_registry existe
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

                # Buscar registro por hash
                cursor.execute("""
                    SELECT file_id, original_filename, table_name,
                        row_count, column_count, upload_date
                    FROM document_registry
                    WHERE file_hash = %s
                """, (file_hash,))

                result = cursor.fetchone()

                if result:
                    table_name = result[2]

                    # NUEVO: Verificar que la tabla realmente existe
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = %s
                        )
                    """, (table_name,))
                    actual_table_exists = cursor.fetchone()[0]

                    if not actual_table_exists:
                        # La tabla no existe, limpiar registro huérfano
                        print(f"⚠️ Registro huérfano detectado: tabla '{table_name}' no existe")
                        print(f"🗑️ Limpiando registro huérfano de document_registry...")

                        cursor.execute("""
                            DELETE FROM document_registry
                            WHERE file_hash = %s
                        """, (file_hash,))
                        conn.commit()

                        print(f"✅ Registro huérfano eliminado")
                        return None  # Permitir que el archivo se suba nuevamente

                    # La tabla existe, es un duplicado real
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
            conn.rollback()
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

    def format_date_to_display(self, date_value) -> str:
        """
        Convierte una fecha en formato ISO o datetime a formato DD/MM/YYYY.
        
        Args:
            date_value: Puede ser un string ISO (ej: '2025-11-25T15:01:51.560747') 
                        o un objeto datetime
        
        Returns:
            str: Fecha en formato DD/MM/YYYY (ej: '25/11/2025')
        """
        from datetime import datetime
        
        if isinstance(date_value, str):
            # Convertir string ISO a datetime
            date_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
        else:
            date_obj = date_value
        
        # Formatear a DD/MM/YYYY
        return date_obj.strftime('%d/%m/%Y')

    async def upload_document(self, file_content: bytes, filename: str) -> dict:
        """
        Procesa y almacena un documento en la BD.
        MODIFICADO: Elimina archivo temporal después de procesarlo.
        NUEVO: Limpia columnas con mezcla de fechas y números decimales.
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
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
            conn = psycopg.connect(connection_string)
            should_close = True
        else:
            should_close = False

        temp_filepath = None  # Para rastrear el archivo temporal

        try:
            # Verificar si ya existe un archivo con el mismo hash
            duplicate = self._check_duplicate_by_hash(file_hash, conn)

            if duplicate:
                print(f"⚠️ Archivo duplicado detectado: {duplicate['original_filename']}")
                
                formatted_date = self.format_date_to_display(duplicate['upload_date'])
                
                raise ValueError(
                    f"Este archivo ya existe en la base de datos como '{duplicate['original_filename']}' "
                    f"(cargado el {formatted_date})"
                )

            # Si no es duplicado, proceder con la carga normal
            print(f"✅ Archivo nuevo detectado, procediendo con la carga...")

            # Generar ID único
            file_id = self._generate_file_id()

            # Generar nombre de tabla
            table_name = self._generate_table_name(filename, file_id)

            # Guardar archivo temporal solo para lectura
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

                if(ENABLE_DATE_FORMAT):
                    # Limpiar columnas con mezcla de fechas y números
                    print(f"🧹 Analizando columnas para detectar mezclas de fechas y números...")
                    df = clean_mixed_date_columns(df)
                    print(f"✅ Limpieza completada")

                # Generar descripción semántica con LLM
                print(f"🤖 Generando descripción semántica con LLM...")
                semantic_description = generate_semantic_description_with_llm(
                    df,
                    table_name,
                    filename
                )

                # Variable para rastrear si se creó la tabla
                table_created = False

                try:
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
                    
                    table_created = True  # Marcar que la tabla fue creada
                    print(f"✅ Tabla '{table_name}' creada exitosamente")

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

                except Exception as e:
                    # Si se creó la tabla pero falló algo después, eliminarla
                    if table_created:
                        try:
                            print(f"🗑️ Eliminando tabla '{table_name}' debido a error en la carga...")
                            with conn.cursor() as cursor:
                                cursor.execute(f"DROP TABLE IF EXISTS public.{table_name} CASCADE")
                                conn.commit()
                            print(f"✅ Tabla '{table_name}' eliminada correctamente")
                        except Exception as drop_error:
                            print(f"⚠️ No se pudo eliminar la tabla '{table_name}': {drop_error}")
                            conn.rollback()
                    
                    # Re-lanzar la excepción original
                    raise e

            finally:
                # Eliminar archivo temporal después de procesarlo
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
            connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
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
        """
        Elimina un documento de la BD.
        MEJORADO: Usa conexión independiente para evitar bloqueos.
        """
        # 🔧 SOLUCIÓN 4: SIEMPRE crear una conexión nueva e independiente
        # Esto evita conflictos con la conexión global usada por el chat
        db_config = load_db_config()
        connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"

        try:
            conn = psycopg.connect(connection_string, autocommit=True)
            print("🔗 Conexión independiente creada para eliminación")
        except Exception as e:
            print(f"❌ Error creando conexión: {e}")
            raise ValueError(f"No se pudo conectar a la base de datos: {e}")

        try:
            # Buscar tabla que contenga el file_id
            stored_tables = list_stored_tables(conn)
            table_to_delete = None

            for table_name in stored_tables:
                if table_name.endswith(f"_{file_id}"):
                    table_to_delete = table_name
                    break

            if not table_to_delete:
                # Verificar si existe en document_registry aunque no esté la tabla
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT table_name FROM document_registry WHERE file_id = %s
                    """, (file_id,))
                    result = cursor.fetchone()

                    if result:
                        print(f"⚠️ Encontrado registro huérfano para file_id: {file_id}")
                        table_to_delete = result[0]
                    else:
                        raise ValueError(f"No se encontró documento con file_id: {file_id}")

            # Eliminar tabla si existe
            with conn.cursor() as cursor:
                cursor.execute(f"DROP TABLE IF EXISTS public.{table_to_delete} CASCADE")
                conn.commit()

            print(f"✅ Tabla eliminada: {table_to_delete}")

            # Eliminar archivo físico si existe
            for filename in os.listdir(self.uploads_dir):
                if filename.startswith(file_id):
                    filepath = os.path.join(self.uploads_dir, filename)
                    try:
                        os.remove(filepath)
                        print(f"🗑️ Archivo eliminado: {filename}")
                    except:
                        pass

            # Eliminar del registro
            try:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = 'public'
                            AND table_name = 'document_registry'
                        )
                    """)
                    registry_exists = cursor.fetchone()[0]

                    if registry_exists:
                        cursor.execute("DELETE FROM document_registry WHERE file_id = %s", (file_id,))
                        rows_deleted = cursor.rowcount
                        conn.commit()

                        if rows_deleted > 0:
                            print(f"📝 Registro eliminado de document_registry")
                        else:
                            print(f"⚠️ No se encontró registro en document_registry para {file_id}")
                    else:
                        print("⚠️ Tabla document_registry no existe")
            except Exception as e:
                print(f"⚠️ Error al eliminar del registro: {e}")
                conn.rollback()

            return {
                "file_id": file_id,
                "table_name": table_to_delete
            }

        finally:
            # 🔧 SOLUCIÓN 4: SIEMPRE cerrar la conexión independiente
            try:
                conn.close()
                print("✅ Conexión independiente cerrada")
            except:
                pass