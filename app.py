from flask import Flask, request, send_file
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO

import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

# Importamos AnimeGANv2-PyTorch
from animegan2_pytorch import AnimeGANv2Pytorch

app = Flask(__name__)

# --------------------------------------------------
# 1) Inicializar InsightFace (detección + face-swap)
# --------------------------------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)

# Cargamos el modelo ONNX para face-swapping
swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

# --------------------------------------------------
# 2) Inicializar AnimeGANv2 con el preentrenado “paprika”
# --------------------------------------------------
# (paprika es uno de los estilos disponibles: "paprika", "hayao", "shinkai", etc.)
gan = AnimeGANv2Pytorch(
    model_name="paprika",
    device="cpu"  # en Railway o Render sin GPU
)

# ----------------------------------------
# Función para hacer face-swap + anime
# ----------------------------------------
@app.route('/')
def index():
    return "✅ InsightFace + AnimeGANv2 Server Online"

@app.route('/procesar', methods=['POST'])
def procesar():
    # 1) Revisar que llegaron los archivos
    if 'image' not in request.files or 'base' not in request.files:
        return "Faltan archivos 'image' o 'base'", 400

    # 2) Guardar temporalmente las imágenes subidas
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as user_file, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:

        user_file.write(request.files['image'].read())
        base_file.write(request.files['base'].read())
        user_file.flush()
        base_file.flush()

    # 3) Cargar imágenes desde disco
    img_user = cv2.imread(user_file.name)
    img_base = cv2.imread(base_file.name)

    # 4) Detectar caras en ambas imágenes
    faces_user = face_app.get(img_user)
    faces_base = face_app.get(img_base)

    if not faces_user or not faces_base:
        return "No se detectaron suficientes caras", 400

    # 5) Hacer face-swap: incrustar cara del usuario en la imagen base
    swapped = swapper.get(img_base, faces_base[0], faces_user[0], paste_back=True)

    # 6) Aplicar AnimeGANv2 al resultado del face-swap
    #    AnimeGANv2-PyTorch recibe un array BGR (OpenCV), así que convertimos:
    swapped_rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB)
    #    AnimeGANv2-PyTorch espera un tensor H×W×3 en formato [0..1] float, lo normalizamos:
    swapped_rgb = swapped_rgb.astype(np.float32) / 255.0

    #    Pasamos por AnimeGAN
    with np.no_grad():
        anime_image_np = gan(swapped_rgb)  # sale en formato float en [0..1], RGB

    #    Convertimos de nuevo a BGR uint8 para OpenCV
    anime_image_np = (anime_image_np * 255.0).clip(0, 255).astype(np.uint8)
    anime_bgr = cv2.cvtColor(anime_image_np, cv2.COLOR_RGB2BGR)

    # 7) Codificar la imagen final como JPEG y devolverla
    _, buffer = cv2.imencode(".jpg", anime_bgr)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    # En producción usas gunicorn; esto solo es para tests locales.
    app.run(debug=True)
