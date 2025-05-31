from flask import Flask, request, send_file
import cv2
import numpy as np
import tempfile
from io import BytesIO

import insightface
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

from animegan2_pytorch import AnimeGANv2Pytorch

app = Flask(__name__)

face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0)
swapper = get_model("inswapper_128.onnx", providers=["CPUExecutionProvider"])

gan = AnimeGANv2Pytorch(
    model_name="paprika",
    device="cpu"
)

@app.route('/')
def index():
    return "✅ InsightFace + AnimeGANv2 Server Online"

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

    swapped = swapper.get(img_base, faces_base[0], faces_user[0], paste_back=True)

    swapped_rgb = cv2.cvtColor(swapped, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    anime_image_np = gan(swapped_rgb)

    anime_image_np = (anime_image_np * 255.0).clip(0, 255).astype(np.uint8)
    anime_bgr = cv2.cvtColor(anime_image_np, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode(".jpg", anime_bgr)
    return send_file(BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
