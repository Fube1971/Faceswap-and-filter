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
# 1) Función “anime/cartoon” (opcional)
# ------------------------------------------------------
def aplicar_estilo_anime(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)

    edges = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    color = cv2.bilateralFilter(img, d=9, sigmaColor=300, sigmaSpace=300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

# ------------------------------------------------------
# 2) Función “color grading terroso/pictórico”
# ------------------------------------------------------
def aplicar_color_terroso(img):
    # Convertir a float [0,1]
    img_f = img.astype(np.float32) / 255.0

    # Multiplicadores BGR para lograr tonos ocre/verde apagado
    balance = np.array([1.1, 1.05, 0.9], dtype=np.float32)
    img_corr = img_f * balance[None,None,:]
    img_corr = np.clip(img_corr, 0.0, 1.0)

    # Pasar a HSV para bajar saturación y subir brillo
    hsv = cv2.cvtColor((img_corr * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] *= 0.6   # reducir saturación
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.2, 0, 255)  # aumentar brillo suave
    img_painted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img_painted

# ------------------------------------------------------
# 3) Función “textura de lienzo/grano”
# ------------------------------------------------------
def aplicar_textura_lienzo(img, intensidad=0.15):
    h, w, _ = img.shape

    # Generar ruido gaussiano en [0,1]
    ruido = np.random.normal(loc=0.5, scale=0.2, size=(h, w)).astype(np.float32)
    ruido = np.clip(ruido, 0.0, 1.0)
    ruido_bgr = np.stack([ruido, ruido, ruido], axis=2)

    # Convertir la imagen original a float [0,1]
    img_f = img.astype(np.float32) / 255.0

    # Mezclar: (1-intensidad)*img + intensidad*(img*ruido)
    mezclado = img_f * ((1.0 - intensidad) + intensidad * ruido_bgr)

    # Suave desenfoque para simular empaste
    mezclado = cv2.GaussianBlur(mezclado, (5,5), sigmaX=1.0, sigmaY=1.0)

    resultado = np.clip(mezclado * 255.0, 0, 255).astype(np.uint8)
    return resultado

# ----------------------------------------
# Inicializa InsightFace (buffalo_l + swapper)
# ----------------------------------------
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)
swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

@app.route('/')
def index():
    return "✅ Memento Quis Style Server Online"

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

    # 5) Hacer face-swap: incrustar cara del jugador en la imagen base
    swapped = swapper.get(img_base, faces_base[0], faces_user[0], paste_back=True)

    # 6) (Opcional) Aplicar filtro anime/cartoon
    anime_img = aplicar_estilo_anime(swapped)

    # 7) Aplicar color grading terroso/pictórico
    terroso = aplicar_color_terroso(anime_img)

    # 8) Aplicar textura de lienzo (grano)
    final_img = aplicar_textura_lienzo(terroso, intensidad=0.15)

    # 9) Codificar como JPEG y devolver la imagen final
    _, buffer = cv2.imencode(".jpg", final_img)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    # En producción usarás gunicorn; debug=True sólo para pruebas locales
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
