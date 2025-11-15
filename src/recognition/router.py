from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from .service import classify_image_bytes, classify_figure_bytes, annotate_face_center_bytes, classify_number_bytes
from .schema import ColorResponse, FigureResponse, NumberResponse, FaceResponse, FaceCoordsResponse
import base64


router = APIRouter(prefix="/recognize", tags=["recognition"])


@router.post('/color', response_model=ColorResponse)
async def recognize_color(file: UploadFile = File(...)):
    """Endpoint que acepta `multipart/form-data` con campo obligatorio `file`.
    Devuelve la etiqueta y el RGB central de la imagen.
    """
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {e}")

    try:
        etiqueta, rgb = classify_image_bytes(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la imagen: {e}")

    return JSONResponse(content={"color": etiqueta, "rgb": rgb})


@router.post('/figures', response_model=FigureResponse)
async def recognize_figure(file: UploadFile = File(...)):
    """Endpoint que acepta `multipart/form-data` con campo obligatorio `file`.
    Devuelve el nombre de la figura detectada en la imagen.
    """
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {e}")

    try:
        figura = classify_figure_bytes(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la imagen: {e}")

    return JSONResponse(content={"figure": figura})


@router.post('/numbers', response_model=NumberResponse)
async def recognize_number(file: UploadFile = File(...)):
    """Endpoint que acepta `multipart/form-data` con campo obligatorio `file`.
    Devuelve el dígito detectado y la confianza.
    """
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {e}")

    try:
        digit, confidence = classify_number_bytes(contents)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la imagen: {e}")

    return JSONResponse(content={"digit": digit, "confidence": confidence})


@router.post('/face', response_model=FaceResponse)
async def recognize_face(file: UploadFile = File(...)):
    """Endpoint que recibe una imagen y devuelve JSON con la imagen anotada (base64)
    y metadata (`found`, `center`).
    """
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {e}")

    try:
        jpeg_bytes, metadata = annotate_face_center_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la imagen: {e}")

    image_b64 = base64.b64encode(jpeg_bytes).decode('utf-8')

    return {
        'found': bool(metadata.get('found')),
        'center': metadata.get('center'),
        'image_b64': image_b64,
    }


@router.post('/face/coords', response_model=FaceCoordsResponse)
async def recognize_face_coords(file: UploadFile = File(...)):
    """Endpoint que recibe una imagen y devuelve únicamente las coordenadas x,y del centro del rostro.
    Si no encuentra rostro devuelve `found: false`.
    """
    try:
        contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error leyendo archivo: {e}")

    try:
        _, metadata = annotate_face_center_bytes(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno al procesar la imagen: {e}")

    if metadata.get('found') and metadata.get('center'):
        x, y = metadata['center']
        return {'found': True, 'x': int(x), 'y': int(y)}
    else:
        return {'found': False, 'x': None, 'y': None}
