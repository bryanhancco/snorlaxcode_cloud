import cv2
import numpy as np
from typing import Tuple, List
from langchain_core.tools import tool

def clasificar_color(r: int, g: int, b: int) -> str:
	"""Clasifica un color RGB en etiquetas simples (mismo comportamiento que antes)."""
	color_bgr = np.uint8([[[b, g, r]]])
	color_hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
	h, s, v = color_hsv[0][0]

	min_saturacion_para_color = 50
	if s < min_saturacion_para_color:
		return "otro color"

	if 100 <= h <= 140:
		return "azul"
	if (0 <= h <= 10) or (165 <= h <= 179):
		return "rojo"
	if 20 <= h <= 40:
		return "amarillo"
	if 140 <= h <= 165:
		return "morado"
	if 10 <= h <= 20:
		return "naranja"
	if 80 <= h <= 100:
		return "celeste"
	if 40 <= h <= 80:
		return "verde"

	# fallback RGB
	colores_referencia_rgb = {
		"rojo": (255, 0, 0),
		"azul": (0, 0, 255),
		"amarillo": (255, 255, 0),
		"morado": (128, 0, 128),
		"naranja": (255, 165, 0),
		"celeste": (0, 191, 255),
		"verde": (0, 255, 0),
	}

	min_distancia = float('inf')
	nombre_color_cercano = "otro color"
	for nombre, (cr, cg, cb) in colores_referencia_rgb.items():
		distancia = np.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
		if distancia < min_distancia:
			min_distancia = distancia
			nombre_color_cercano = nombre

	umbral_distancia = 60
	if min_distancia < umbral_distancia:
		return nombre_color_cercano
	else:
		return "otro color"


def _read_image_from_bytes(data: bytes) -> np.ndarray:
	arr = np.frombuffer(data, np.uint8)
	img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
	if img is None:
		raise ValueError("No se pudo decodificar la imagen")
	return img


def _classify_image_np(img: np.ndarray) -> Tuple[str, int, int, int]:
	h, w = img.shape[:2]
	tamaño_cuadrado = min(50, w // 4, h // 4)
	x_centro = w // 2
	y_centro = h // 2

	x1 = max(0, x_centro - tamaño_cuadrado // 2)
	y1 = max(0, y_centro - tamaño_cuadrado // 2)
	x2 = min(w, x_centro + tamaño_cuadrado // 2)
	y2 = min(h, y_centro + tamaño_cuadrado // 2)

	roi = img[y1:y2, x1:x2]
	if roi.size == 0:
		raise ValueError("ROI vacía: imagen demasiado pequeña")

	b, g, r = [int(x) for x in cv2.mean(roi)[:3]]
	etiqueta = clasificar_color(r, g, b)
	return etiqueta, r, g, b

def classify_image_bytes(data: bytes) -> Tuple[str, List[int]]:
	"""Wrapper útil para routers: recibe bytes de imagen y devuelve (etiqueta, [r,g,b])."""
	img = _read_image_from_bytes(data)
	etiqueta, r, g, b = _classify_image_np(img)
	return etiqueta, [r, g, b]


# -------------------- Figure recognition helpers --------------------
def remove_collinear_vertices(approx, angle_thresh_deg=15):
	pts = approx.reshape(-1, 2)
	if len(pts) <= 3:
		return approx.astype(np.int32)
	keep = []
	n = len(pts)
	for i in range(n):
		prev = pts[(i - 1) % n].astype(float)
		cur = pts[i].astype(float)
		nxt = pts[(i + 1) % n].astype(float)
		v1 = prev - cur
		v2 = nxt - cur
		n1 = np.linalg.norm(v1)
		n2 = np.linalg.norm(v2)
		if n1 < 1e-6 or n2 < 1e-6:
			continue
		cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
		ang = np.degrees(np.arccos(cosang))
		if ang < 180.0 - angle_thresh_deg:
			keep.append(cur.astype(int))
	if len(keep) < 3:
		return approx.astype(np.int32)
	return np.array(keep).reshape(-1, 1, 2).astype(np.int32)


def obtener_nombre_figura(approx, area):
	num_vertices = len(approx)

	def internal_angles(pts):
		pts = pts.reshape(-1, 2).astype(float)
		n = len(pts)
		angles = []
		for i in range(n):
			prev = pts[(i - 1) % n]
			cur = pts[i]
			nxt = pts[(i + 1) % n]
			v1 = prev - cur
			v2 = nxt - cur
			n1 = np.linalg.norm(v1)
			n2 = np.linalg.norm(v2)
			if n1 < 1e-6 or n2 < 1e-6:
				angles.append(180.0)
				continue
			cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
			ang = np.degrees(np.arccos(cosang))
			angles.append(ang)
		return angles

	if num_vertices == 3:
		return "triángulo"

	if num_vertices == 4:
		angles = internal_angles(approx)
		if (not cv2.isContourConvex(approx)) or any(a > 150.0 for a in angles):
			return "triángulo"

		x, y, w, h = cv2.boundingRect(approx)
		aspect_ratio = float(w) / h if h != 0 else 0
		if 0.9 <= aspect_ratio <= 1.1:
			return "cuadrado"
		else:
			return "rectángulo"

	if num_vertices > 4:
		perimeter = cv2.arcLength(approx, True)
		if perimeter > 0:
			circularity = 4 * np.pi * area / (perimeter * perimeter)
			if circularity > 0.88:
				return "círculo"

	return "desconocida"


def detect_figure_from_frame(frame: np.ndarray) -> str:
	"""Detecta una figura en la imagen y devuelve el nombre (sin mostrar ni usar cámara)."""
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
	lower_green_lemon = np.array([30, 80, 80])
	upper_green_lemon = np.array([90, 255, 255])
	mask = cv2.inRange(hsv, lower_green_lemon, upper_green_lemon)

	kernel = np.ones((3, 3), np.uint8)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	figura_detectada = "ninguna"
	mejor_contorno = None
	max_area = 0

	if contours:
		for contour in contours:
			area = cv2.contourArea(contour)
			if 5000 < area < 200000:
				hull = cv2.convexHull(contour)
				hull_area = cv2.contourArea(hull)
				solidity = float(area) / hull_area if hull_area > 0 else 0
				if solidity > 0.9 and area > max_area:
					max_area = area
					mejor_contorno = contour
	if mejor_contorno is not None:
		perimeter = cv2.arcLength(mejor_contorno, True)
		epsilons = [
			max(0.003 * perimeter, 0.5),
			max(0.006 * perimeter, 1.0),
			max(0.01 * perimeter, 1.5),
			max(0.02 * perimeter, 2.5),
		]
		approx = None
		for e in epsilons:
			a = cv2.approxPolyDP(mejor_contorno, e, True)
			a = remove_collinear_vertices(a, angle_thresh_deg=12)
			if 3 <= len(a) <= 4:
				approx = a
				break

		if approx is None:
			hull = cv2.convexHull(mejor_contorno)
			for e in epsilons:
				a = cv2.approxPolyDP(hull, e, True)
				a = remove_collinear_vertices(a, angle_thresh_deg=12)
				if 3 <= len(a) <= 4:
					approx = a
					break

		if approx is None:
			a_cont = remove_collinear_vertices(cv2.approxPolyDP(mejor_contorno, epsilons[0], True), angle_thresh_deg=12)
			a_hull = remove_collinear_vertices(cv2.approxPolyDP(cv2.convexHull(mejor_contorno), epsilons[1], True), angle_thresh_deg=12)
			approx = a_cont if len(a_cont) <= len(a_hull) else a_hull

		if approx is not None and len(approx) >= 3:
			figura_detectada = obtener_nombre_figura(approx, max_area)

	return figura_detectada

def classify_figure_bytes(data: bytes) -> str:
	"""Recibe bytes de imagen y devuelve el nombre de la figura detectada."""
	img = _read_image_from_bytes(data)
	return detect_figure_from_frame(img)


# -------------------- Number recognition helpers --------------------
import os
try:
	import tensorflow as tf
except Exception:
	tf = None

_mnist_model = None
MODEL_FILENAME = 'mnist_cnn_model.h5'


def _get_model_path() -> str:
	return os.path.join(os.path.dirname(__file__), MODEL_FILENAME)


def _load_mnist_model():
	global _mnist_model
	if _mnist_model is not None:
		return _mnist_model
	if tf is None:
		raise RuntimeError('TensorFlow no está disponible en el entorno')
	model_path = _get_model_path()
	if not os.path.exists(model_path):
		raise FileNotFoundError(f'Modelo no encontrado en {model_path}')
	# Intento de carga con tf.keras primero
	try:
		_mnist_model = tf.keras.models.load_model(model_path, compile=False)
		return _mnist_model
	except Exception as e_tf:
		# Si falla la deserialización con tf.keras (p. ej. modelo guardado con 'keras' standalone),
		# intentamos cargar con la librería `keras` si está disponible.
		try:
			import keras as standalone_keras  # type: ignore
			_mnist_model = standalone_keras.models.load_model(model_path, compile=False)
			return _mnist_model
		except Exception as e_keras:
			# Falló ambos intentos: ofrecemos un mensaje de error útil al usuario
			raise RuntimeError(
				"Fallo al cargar el modelo MNIST. Intentos:") from e_keras


def _prepare_digit_roi_from_mask(mask: np.ndarray, bbox: tuple) -> np.ndarray:
	x, y, w, h = bbox
	roi = mask[y:y + h, x:x + w]
	if roi.size == 0:
		raise ValueError('ROI vacía para número')

	target_size = 28
	padding = 4
	max_dim = max(roi.shape[0], roi.shape[1])
	scale_factor = min((target_size - 2 * padding) / max_dim, 1.0)
	resized_roi = cv2.resize(
		roi,
		(max(1, int(roi.shape[1] * scale_factor)), max(1, int(roi.shape[0] * scale_factor))),
		interpolation=cv2.INTER_AREA
	)

	final_roi = np.zeros((target_size, target_size), dtype=np.uint8)
	start_x = (target_size - resized_roi.shape[1]) // 2
	start_y = (target_size - resized_roi.shape[0]) // 2
	final_roi[start_y:start_y + resized_roi.shape[0], start_x:start_x + resized_roi.shape[1]] = resized_roi
	_, final_roi = cv2.threshold(final_roi, 127, 255, cv2.THRESH_BINARY)
	return final_roi

def classify_number_bytes(data: bytes) -> tuple[int, float]:
	"""Recibe bytes de imagen y devuelve (digit, confidence).

	Busca regiones verdes (rango por defecto), extrae la ROI más plausible,
	la procesa y la clasifica con el modelo MNIST.
	"""
	img = _read_image_from_bytes(data)

	# mask green areas (same ranges used in the original script)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_green = np.array([35, 100, 100])
	upper_green = np.array([85, 255, 255])
	mask = cv2.inRange(hsv, lower_green, upper_green)

	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		raise ValueError('No se detectaron regiones de interés para números')

	plausible_contours = []
	for contour in contours:
		area = cv2.contourArea(contour)
		x, y, w, h = cv2.boundingRect(contour)
		if 1000 < area < 150000:
			aspect_ratio = float(w) / h if h != 0 else 0
			if 0.4 < aspect_ratio < 1.6:
				plausible_contours.append((area, contour, (x, y, w, h)))

	if not plausible_contours:
		raise ValueError('No se encontraron contornos plausibles para números')

	plausible_contours.sort(key=lambda x: x[0], reverse=True)
	_, largest_contour, bbox = plausible_contours[0]

	final_roi = _prepare_digit_roi_from_mask(mask, bbox)

	# prepare input for model
	inp = final_roi.astype('float32') / 255.0
	inp = np.expand_dims(inp, axis=(0, -1))  # (1,28,28,1)

	model = _load_mnist_model()
	preds = model.predict(inp, verbose=0)
	digit = int(np.argmax(preds))
	confidence = float(np.max(preds))
	return digit, confidence


# -------------------- Direction recognition (hand side) --------------------
def classify_direction_bytes(data: bytes) -> tuple[str, List[int]]:
	"""Recibe bytes de imagen y devuelve ('Izquierda'|'Derecha', [cx,cy]).

	Basado en `first_module/features/direction_recognition/direction.py` pero
	sin acceso a cámara: procesa una sola imagen y determina si la mano
	está en la mitad izquierda o derecha del frame.
	"""
	img = _read_image_from_bytes(data)
	h, w = img.shape[:2]

	# Convertir a HSV y enmascarar color de piel (rango aproximado)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	lower_skin = np.array([0, 20, 70], dtype=np.uint8)
	upper_skin = np.array([20, 255, 255], dtype=np.uint8)
	mask = cv2.inRange(hsv, lower_skin, upper_skin)

	# Suavizar para reducir ruido
	mask = cv2.GaussianBlur(mask, (5, 5), 0)

	# Encontrar contornos
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		raise ValueError('No se detectó ninguna mano')

	# Seleccionar la mano más grande y filtrar ruido
	max_contour = max(contours, key=cv2.contourArea)
	if cv2.contourArea(max_contour) <= 1000:
		raise ValueError('Mano demasiado pequeña o ruido')

	M = cv2.moments(max_contour)
	if M.get('m00', 0) == 0:
		raise ValueError('No se pudo calcular el centro de la mano')

	cx = int(M['m10'] / M['m00'])
	cy = int(M['m01'] / M['m00'])

	side = "Izquierda" if cx < (w / 2) else "Derecha"
	return side, [int(cx), int(cy)]


# -------------------- Face detection / annotation --------------------
_face_cascade = None


def _get_face_cascade():
	global _face_cascade
	if _face_cascade is not None:
		return _face_cascade
	cascade_name = 'haarcascade_frontalface_default.xml'
	cascade_path = cv2.data.haarcascades + cascade_name
	if not cv2.os.path.exists(cascade_path):
		# fallback: try to use cv2.data.haarcascades folder join
		cascade_path = cv2.data.haarcascades + cascade_name
	_face_cascade = cv2.CascadeClassifier(cascade_path)
	return _face_cascade


def annotate_face_center_bytes(data: bytes) -> tuple[bytes, dict]:
	"""Decodifica `data` (bytes de imagen), detecta la cara más grande y
	anota un punto en su centro. Devuelve (jpeg_bytes, metadata).

	Metadata: {'found': bool, 'center': [x,y] | None}
	"""
	img = _read_image_from_bytes(data)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	face_cascade = _get_face_cascade()

	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
	metadata = {'found': False, 'center': None}

	if len(faces) > 0:
		# seleccionar la cara más grande
		x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
		cx = int(x + w / 2)
		cy = int(y + h / 2)
		# dibujar un punto rojo en el centro
		cv2.circle(img, (cx, cy), radius=6, color=(0, 0, 255), thickness=-1)
		metadata['found'] = True
		metadata['center'] = [int(cx), int(cy)]

	# codificar a JPEG
	ok, buf = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
	if not ok:
		raise RuntimeError('Error al codificar la imagen a JPEG')
	return buf.tobytes(), metadata
