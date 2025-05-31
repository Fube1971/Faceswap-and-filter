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
    return "✅ InsightFace + Filtros Server Online"

@app.route('/procesar', methods=['POST'])
def procesar():
    # Verificar que la “base” siempre llegue
    if 'base' not in request.files:
        return "Falta archivo 'base'", 400

    # El parámetro skip_swap puede venir en el form: “true” o no existir.
    skip_swap = request.form.get('skip_swap', 'false').lower() == 'true'

    # Guardar temporalmente base y opcionalmente “image” (foto del jugador)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:
        base_file.write(request.files['base'].read())
        base_file.flush()
        base_path = base_file.name

        # Si no se salta el swap, validamos que ‘image’ exista
        if not skip_swap:
            if 'image' not in request.files:
                return "Falta archivo 'image' para face-swap", 400
            user_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            user_file.write(request.files['image'].read())
            user_file.flush()
            user_path = user_file.name
        else:
            user_path = None

    # Leer la imagen base con OpenCV
    img_base = cv2.imread(base_path)
    if img_base is None:
        return "Error al leer la imagen base", 500

    # Si skip_swap==false, hacemos face-swap (con “image”)
    if not skip_swap:
        img_user = cv2.imread(user_path)
        if img_user is None:
            return "Error al leer la imagen 'image'", 500

        faces_user = face_app.get(img_user)
        faces_base = face_app.get(img_base)
        if not faces_user or not faces_base:
            return "No se detectaron suficientes caras para face-swap", 400

        swapped = swapper.get(img_base, faces_base[0], faces_user[0], paste_back=True)
        working_img = swapped
    else:
        # Modo “solo filtros”: sin face-swap, partimos de la base original
        working_img = img_base

    # Aplicar filtro “anime/cartoon” (opcional)
    working_img = aplicar_estilo_anime(working_img)

    # Aplicar color grading terroso
    working_img = aplicar_color_terroso(working_img)

    # Aplicar textura de lienzo/grano
    final_img = aplicar_textura_lienzo(working_img, intensidad=0.15)

    # Codificar y devolver la imagen final
    _, buffer = cv2.imencode(".jpg", final_img)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
