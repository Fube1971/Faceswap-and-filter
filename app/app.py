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

# Preparar analizador de rostros y modelo de intercambio
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)
swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

@app.route('/')
def index():
    return "✅ InsightFace FaceSwap Server Ready"

@app.route('/procesar', methods=['POST'])
def procesar():
    if 'image' not in request.files or 'base' not in request.files:
        return "Faltan archivos 'image' o 'base'", 400

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as user_file, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:

        user_file.write(request.files['image'].read())
        base_file.write(request.files['base'].read())

        user_file.flush()
        base_file.flush()

    img_user = cv2.imread(user_file.name)
    img_base = cv2.imread(base_file.name)

    faces_user = face_app.get(img_user)
    faces_base = face_app.get(img_base)

    if not faces_user or not faces_base:
        return "No se detectaron suficientes caras", 400

    # Realiza el face swap de la cara del jugador en la imagen base
    swapped = swapper.get(img_base, faces_base[0], faces_user[0], paste_back=True)

    _, buffer = cv2.imencode(".jpg", swapped)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
