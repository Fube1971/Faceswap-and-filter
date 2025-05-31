from flask import Flask, request, send_file
import cv2
import numpy as np
import mediapipe as mp
import os
from io import BytesIO

app = Flask(__name__)

IMAGE_FOLDER = 'imagenes_banco'

mp_face_detection = mp.solutions.face_detection

def seamless_face_swap(source_face, target_img):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.6) as face_detection:
        source_result = face_detection.process(cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB))
        target_result = face_detection.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

        if not source_result.detections or not target_result.detections:
            return None

        source_box = source_result.detections[0].location_data.relative_bounding_box
        h_src, w_src, _ = source_face.shape
        x_src, y_src = int(source_box.xmin * w_src), int(source_box.ymin * h_src)
        w_src_box, h_src_box = int(source_box.width * w_src), int(source_box.height * h_src)
        jugador_face = source_face[y_src:y_src+h_src_box, x_src:x_src+w_src_box]

        target_box = target_result.detections[0].location_data.relative_bounding_box
        h_tgt, w_tgt, _ = target_img.shape
        x_tgt, y_tgt = int(target_box.xmin * w_tgt), int(target_box.ymin * h_tgt)
        w_tgt_box, h_tgt_box = int(target_box.width * w_tgt), int(target_box.height * h_tgt)

        jugador_face_resized = cv2.resize(jugador_face, (w_tgt_box, h_tgt_box))

        # Crear máscara para seamlessClone
        mask = 255 * np.ones(jugador_face_resized.shape, jugador_face_resized.dtype)

        # Coordenadas centrales para blending
        center = (x_tgt + w_tgt_box // 2, y_tgt + h_tgt_box // 2)

        # Aplicar seamlessClone para blending suave
        blended_img = cv2.seamlessClone(jugador_face_resized, target_img, mask, center, cv2.NORMAL_CLONE)

        return blended_img

@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    if 'image' not in request.files:
        return 'No se envió ninguna imagen', 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    source_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    resultados = []

    for filename in os.listdir(IMAGE_FOLDER):
        target_path = os.path.join(IMAGE_FOLDER, filename)
        target_img = cv2.imread(target_path)

        swapped_img = seamless_face_swap(source_img, target_img)
        if swapped_img is None:
            continue

        _, buffer = cv2.imencode('.jpg', swapped_img)
        resultados.append(buffer.tobytes())

    if resultados:
        io_buf = BytesIO(resultados[0])
        return send_file(io_buf, mimetype='image/jpeg')

    return 'No se pudieron procesar imágenes', 400

if __name__ == "__main__":
    app.run(debug=True)
