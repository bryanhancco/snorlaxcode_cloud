from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv, find_dotenv
from google import genai
from google.genai import types
from ..recognition.service import (
    classify_image_bytes,
    classify_figure_bytes,
    classify_number_bytes,
    classify_direction_bytes,
)
import os
import base64
from typing import Any

load_dotenv(find_dotenv())

google_api_key = os.environ.get('GOOGLE_API_KEY')
llm = GoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=google_api_key)

def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.

    Args:
    text (str): The text to wrap.
    width (int): The width to wrap the text to.

    Returns:
    str: The wrapped text.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

def chat(message: str):
    """Enviar un mensaje genérico al LLM y devolver la respuesta.

    Este método no usa documentos recuperados; construye un prompt genérico
    con un `SystemMessage` que define el comportamiento (agente educativo)
    y el `HumanMessage` con el `message` recibido.
    """
    # Construimos los mensajes
    messages = [
        SystemMessage(
            content=(
                "Eres un agente educativo que ayuda a los estudiantes con edades de entre 3 y 7 años a responder preguntas de forma sencilla, evitando lenguaje inapropiado o complejo. Responde siempre de manera amigable y clara."
            )
        ),
        HumanMessage(
            content=message
        )
    ]

    # Ejecuta el modelo con los mensajes
    response = llm.invoke(messages)
    print("LLM response:", response)
    
    return word_wrap(str(response))

def tool_calling(message: str):
    # Definición de un esquema de herramienta ficticia
    # mapa de funciones locales (callables) que pueden ejecutarse cuando el LLM
    # solicita una función. Las claves deben coincidir con los nombres de las
    # declarations que se pasan al modelo.
    # Wrappers: the LLM/tool protocol sends JSON-like arguments, not raw bytes.
    # These wrappers expect a dict with key 'image_b64' containing a base64 string.
    def _wrap_image_to_bytes_call(fn):
        def _inner(args: dict[str, Any]):
            # accept either direct base64 or nested values
            if not isinstance(args, dict):
                return {"error": "invalid arguments, expected object with 'image_b64'"}
            b64 = args.get('image_b64') or args.get('image') or args.get('image_base64')
            if not b64:
                return {"error": "missing 'image_b64' in arguments"}
            try:
                img_bytes = base64.b64decode(b64)
            except Exception as e:
                return {"error": f"invalid base64 image: {e}"}
            try:
                result = fn(img_bytes)
                # normalize result to JSON-serializable form
                if isinstance(result, tuple) or isinstance(result, list):
                    return list(result)
                return result
            except Exception as e:
                return {"error": str(e)}
        return _inner

    function_map = {
        "colores": _wrap_image_to_bytes_call(lambda b: classify_image_bytes(b)),
        "figuras": _wrap_image_to_bytes_call(lambda b: classify_figure_bytes(b)),
        "numeros": _wrap_image_to_bytes_call(lambda b: classify_number_bytes(b)),
        "direccion": _wrap_image_to_bytes_call(lambda b: classify_direction_bytes(b)),
    }

    # Declaraciones de función para la API (schema que el modelo puede usar
    # para decidir hacer una llamada a función).
    # function declarations — include parameters so the model knows to send a JSON object
    colores_fn = {
        "name": "colores",
        "description": "Identificar color predominante en la imagen (espera {'image_b64': '<base64>'}).",
        "parameters": {"type": "object", "properties": {"image_b64": {"type": "string", "description": "Imagen codificada en base64"}}, "required": ["image_b64"]},
    }

    figuras_fn = {
        "name": "figuras",
        "description": "Detectar figura geométrica en la imagen (espera {'image_b64': '<base64>'}).",
        "parameters": {"type": "object", "properties": {"image_b64": {"type": "string"}}, "required": ["image_b64"]},
    }

    numeros_fn = {
        "name": "numeros",
        "description": "Reconocer dígito en la imagen (espera {'image_b64': '<base64>'}).",
        "parameters": {"type": "object", "properties": {"image_b64": {"type": "string"}}, "required": ["image_b64"]},
    }

    direccion_fn = {
        "name": "direccion",
        "description": "Determinar si la mano está a la izquierda o derecha (espera {'image_b64': '<base64>'}).",
        "parameters": {"type": "object", "properties": {"image_b64": {"type": "string"}}, "required": ["image_b64"]},
    }

    client = genai.Client()
    tool_spec = types.Tool(function_declarations=[colores_fn, figuras_fn, numeros_fn, direccion_fn])
    config = types.GenerateContentConfig(tools=[tool_spec])

    # Usamos el mensaje del usuario como contenido para el modelo
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=message,
        config=config,
    )

    # Verificar si el modelo solicitó una llamada a función
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        return function_call.name
    else:
        return None