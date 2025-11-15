
from pydantic import BaseModel
from typing import List


class ColorResponse(BaseModel):
	"""Respuesta devuelta por el servicio de reconocimiento de color."""
	color: str
	rgb: List[int]


class FigureResponse(BaseModel):
	"""Respuesta devuelta por el servicio de reconocimiento de figuras."""
	figure: str


class NumberResponse(BaseModel):
	"""Respuesta devuelta por el servicio de reconocimiento de números."""
	digit: int
	confidence: float


class FaceResponse(BaseModel):
	"""Metadata sobre la detección facial.

	Nota: el endpoint `/recognize/face` devuelve la imagen anotada (image/jpeg).
	Este modelo describe la metadata opcional que acompaña a la imagen.
	"""
	found: bool
	center: list[int] | None = None
	image_b64: str | None = None


class FaceCoordsResponse(BaseModel):
	"""Respuesta que contiene únicamente las coordenadas del centro facial."""
	found: bool
	x: int | None = None
	y: int | None = None
