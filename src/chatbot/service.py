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
    function_map = {
        "colores": classify_image_bytes,
        "figuras": classify_figure_bytes,
        "numeros": classify_number_bytes,
        "direccion": classify_direction_bytes,
    }

    # Declaraciones de función para la API (schema que el modelo puede usar
    # para decidir hacer una llamada a función).
    colores_fn = {
        "name": "colores",
        "description": "Identificar color predominante en la imagen (usa classify_image_bytes).",
    }

    figuras_fn = {
        "name": "figuras",
        "description": "Detectar figura geométrica en la imagen (usa classify_figure_bytes).",
    }

    numeros_fn = {
        "name": "numeros",
        "description": "Reconocer dígito en la imagen (usa classify_number_bytes).",
    }

    direccion_fn = {
        "name": "direccion",
        "description": "Determinar si la mano está a la izquierda o derecha (usa classify_direction_bytes).",
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