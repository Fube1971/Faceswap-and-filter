from flask import Flask, request, send_file
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO
import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

app = Flask(__name__)

# ------------------------------------------------------
# Función para convertir una imagen BGR a estilo “anime”
# ------------------------------------------------------
def aplicar_estilo_anime(img):
    """
    Convierte una imagen BGR de OpenCV en un estilo tipo 'anime/cartoon'.
    """
    # 1) Convertir a escala de grises y aplicar median blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    # 2) Detectar bordes con adaptiveThreshold
    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    # 3) Suavizar colores usando bilateralFilter
    color = cv2.bilateralFilter(img, d=9, sigmaColor=300, sigmaSpace=300)

    # 4) Combinar bordes y colores para efecto “cartoon”
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# ----------------------------------------
# Inicializa InsightFace (buffalo_l + swapper)
# ----------------------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)
swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

@app.route('/')
def index():
    return "✅ InsightFace + Anime Filter Server Online"

@app.route('/procesar', methods=['POST'])
def procesar():
    # 1) Verificar que ambas imágenes llegaron
    if 'image' not in request.files or 'base' not in request.files:
        return "Faltan archivos 'image' o 'base'", 400

    # 2) Guardar temporalmente los archivos subidos
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as user_file, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:

        user_file.write(request.files['image'].read())
        base_file.write(request.files['base'].read())
        user_file.flush()
        base_file.flush()

    # 3) Leer imágenes desde disco
    img_user = cv2.imread(user_file.name)
    img_base = cv2.imread(base_file.name)

    # 4) Detectar caras en ambas imágenes
    faces_user = face_app.get(img_user)
    faces_base = face_app.get(img_base)

    if not faces_user or not faces_base:
        return "No se detectaron suficientes caras", 400

    # 5) Hacer face swap: incrustar cara del jugador en la imagen base
    swapped = swapper.get(img_base, faces_base[0], faces_user[0], paste_back=True)

    # 6) Aplicar filtro anime/cartoon sobre la imagen resultante
    anime_img = aplicar_estilo_anime(swapped)

    # 7) Codificar como JPEG y devolver la imagen
    _, buffer = cv2.imencode(".jpg", anime_img)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    # Nota: en producción se usa gunicorn; el debug solo para tests locales
    app.run(debug=True)
