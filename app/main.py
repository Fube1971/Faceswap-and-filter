from flask import Flask, request, send_file
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO

app = Flask(__name__)

TARGET_IMAGE_PATH = "target_face.jpg"  # imagen con cara objetivo

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def swap_face(source_img, target_img):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        source_results = face_detection.process(cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB))
        target_results = face_detection.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

        if not source_results.detections or not target_results.detections:
            return None

        # Obtener cajas delimitadoras
        source_box = source_results.detections[0].location_data.relative_bounding_box
        target_box = target_results.detections[0].location_data.relative_bounding_box

        # Obtener dimensiones reales
        h_src, w_src, _ = source_img.shape
        h_tgt, w_tgt, _ = target_img.shape

        # Coordenadas reales
        x_src, y_src = int(source_box.xmin * w_src), int(source_box.ymin * h_src)
        w_src_box, h_src_box = int(source_box.width * w_src), int(source_box.height * h_src)

        x_tgt, y_tgt = int(target_box.xmin * w_tgt), int(target_box.ymin * h_tgt)
        w_tgt_box, h_tgt_box = int(target_box.width * w_tgt), int(target_box.height * h_tgt)

        # Recortar caras
        face_tgt_crop = target_img[y_tgt:y_tgt+h_tgt_box, x_tgt:x_tgt+w_tgt_box]
        face_tgt_resized = cv2.resize(face_tgt_crop, (w_src_box, h_src_box))

        # Intercambiar la cara
        source_img[y_src:y_src+h_src_box, x_src:x_src+w_src_box] = face_tgt_resized

        return source_img

@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    if 'image' not in request.files:
        return 'No se envió ninguna imagen', 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    source_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    target_img = cv2.imread(TARGET_IMAGE_PATH)

    output_img = swap_face(source_img, target_img)
    if output_img is None:
        return 'No se detectaron suficientes caras', 400

    _, buffer = cv2.imencode('.jpg', output_img)
    io_buf = BytesIO(buffer)

    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
