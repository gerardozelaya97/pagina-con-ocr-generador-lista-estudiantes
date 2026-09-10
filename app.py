import os
import json
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

# 1. Inicializar la aplicación FastAPI
app = FastAPI(title="OCR Escaneo de Listas con Gemini")

# 2. Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. API Key e inicialización del cliente
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Falta la variable de entorno GEMINI_API_KEY. "
        "Defínela antes de arrancar el servidor."
    )

client = genai.Client(
    api_key=API_KEY,
    http_options={
        'headers': {
            'x-goog-api-key': API_KEY
        }
    }
)

# 4. Esquema enfocado en NIE y Nombre
class Estudiante(BaseModel):
    nie: Optional[str] = None
    nombre: str

class RespuestaEscaneo(BaseModel):
    estudiantes: List[Estudiante]

# 5. Función con reintentos automáticos para errores 503 (modelo saturado)
async def llamar_gemini_con_reintentos(contents: list, max_intentos: int = 4):
    prompt = (
        "Analiza detenidamente la(s) imagen(es) o documento(s) que contiene(n) el listado de estudiantes, "
        "examinando todas las columnas desde 'No.' hasta 'Observación'. "
        "Extrae exclusivamente dos datos por cada alumno: 'nie' y 'nombre'. "
        "Reglas: "
        "1. Recorre las listas de arriba hacia abajo sin omitir ninguna fila visible. "
        "2. Si hay varias páginas o imágenes, puede ser que una sea continuación de otra; "
        "   combina todos los estudiantes en un solo listado sin duplicar NIE. "
        "3. Devuelve los nombres completos tal como aparecen en la tabla. "
        "4. Incluye solo los dígitos numéricos para el NIE. "
        "5. Si un dato es parcialmente ilegible, extrae el texto que logres distinguir "
        "   en lugar de omitir la fila."
    )

    ultimo_error = None
    for intento in range(1, max_intentos + 1):
        try:
            print(f"[Gemini] Intento {intento}/{max_intentos}...")
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=contents + [prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=RespuestaEscaneo,
                    temperature=0.1
                )
            )
            return response

        except Exception as e:
            ultimo_error = e
            err_str = str(e)
            es_temporal = (
                "503" in err_str or
                "UNAVAILABLE" in err_str or
                "429" in err_str or
                "RESOURCE_EXHAUSTED" in err_str or
                "overloaded" in err_str.lower()
            )

            if es_temporal and intento < max_intentos:
                espera = 2 ** intento
                print(f"[Gemini] Error temporal (intento {intento}). Reintentando en {espera}s...")
                await asyncio.sleep(espera)
            else:
                raise

    raise ultimo_error

# 6. Endpoint de procesamiento (acepta imágenes Y PDFs)
@app.post("/api/scan")
async def scan_image(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se envió ningún archivo.")

    # Validar: aceptar imágenes y PDFs
    for f in files:
        ct = f.content_type or ""
        if not (ct.startswith("image/") or ct == "application/pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"El archivo '{f.filename}' debe ser imagen o PDF (recibido: {ct})."
            )

    try:
        contents = []
        for f in files:
            data = await f.read()
            contents.append(
                types.Part.from_bytes(data=data, mime_type=f.content_type)
            )

        print(f"[Scan] Procesando {len(files)} archivo(s)...")

        response = await llamar_gemini_con_reintentos(contents)

        if not response.text:
            raise ValueError("La respuesta de la API vino vacía.")

        data = json.loads(response.text)
        students_list = data.get("estudiantes", [])

        # Deduplicar por NIE manteniendo orden
        seen = set()
        unicos = []
        for s in students_list:
            nie = s.get("nie")
            if nie and nie in seen:
                continue
            if nie:
                seen.add(nie)
            unicos.append(s)

        return {
            "status": "ok",
            "total": len(unicos),
            "students": unicos
        }

    except Exception as e:
        print(f"\n--- ERROR DETECTADO EN EL SERVIDOR ---")
        print(f"Tipo de error: {type(e).__name__}")
        print(f"Detalle: {str(e)}")
        print(f"--------------------------------------\n")

        err_str = str(e)
        if "503" in err_str or "UNAVAILABLE" in err_str:
            raise HTTPException(
                status_code=503,
                detail="El modelo de IA está saturado. Intenta de nuevo en unos segundos."
            )

        raise HTTPException(status_code=500, detail=f"Error procesando el archivo: {str(e)}")

# 7. Arranque de Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
