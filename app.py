from flask import Flask, request, send_file
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO

import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Aquí importamos AnimeGANv2-PyTorch
from animegan2_pytorch import AnimeGANv2Pytorch

app = Flask(__name__)

# --------------------------------------------------
# 1) Inicializar InsightFace (detección + face-swap)
# --------------------------------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

# --------------------------------------------------
# 2) Inicializar AnimeGANv2 con el preentrenado “paprika”
# --------------------------------------------------
gan = AnimeGANv2Pytorch(
    model_name="paprika",
    device="cpu"
)

@app.route('/')
def index():
    return "✅ InsightFace + AnimeGANv2 Server Online"

@app.route('/procesar', methods=['POST'])
def procesar():
    # 1) Verificar llegada de archivos
    if 'image' not in request.files or 'base' not in request.files:
        return "Faltan archivos 'image' o 'base'", 400

    # 2) Guardar archivos temporales
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as user_file, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:

        user_file.write(request.files['image'].read())
        base_file.write(request.files['base'].read())
        user_file.flush()
        base_file.flush()

    # 3) Leer imágenes con OpenCV
    img_user = cv2.imread(user_file.name)
    img_base = cv2.imread(base_file.name)

    # 4) Detectar caras
    faces_user = face_app.get(img_user)
    faces_base = face_app.get(img_base)

    if not faces_user or not faces_base:
        return "No se detectaron suficientes caras", 400

    # 5) Hacer face-swap
    swapped = swapper.get(img_base, faces_base[0], faces_user[0], paste_back=True)

    # 6) Convertir BGR a RGB en [0..1] y pasar por AnimeGANv2
    swapped_rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # AnimeGANv2-PyTorch devuelve RGB float en [0..1]
    anime_image_np = gan(swapped_rgb)

    # 7) Convertir de vuelta a BGR uint8 para OpenCV
    anime_image_np = (anime_image_np * 255.0).clip(0, 255).astype(np.uint8)
    anime_bgr = cv2.cvtColor(anime_image_np, cv2.COLOR_RGB2BGR)

    # 8) Codificar como JPEG y devolver
    _, buffer = cv2.imencode(".jpg", anime_bgr)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
