from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
import mediapipe as mp
import os
from io import BytesIO

app = Flask(__name__)

# Directorio donde están las imágenes almacenadas
IMAGE_FOLDER = 'imagenes_banco'

mp_face_detection = mp.solutions.face_detection

def face_swap(source_face, target_img):
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        # Detectar cara en la imagen fuente (jugador)
        source_result = face_detection.process(cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB))
        # Detectar cara en la imagen objetivo
        target_result = face_detection.process(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))

        if not source_result.detections or not target_result.detections:
            return None

        # Coordenadas jugador (source)
        source_box = source_result.detections[0].location_data.relative_bounding_box
        h_src, w_src, _ = source_face.shape
        x_src, y_src = int(source_box.xmin * w_src), int(source_box.ymin * h_src)
        w_src_box, h_src_box = int(source_box.width * w_src), int(source_box.height * h_src)
        jugador_face = source_face[y_src:y_src+h_src_box, x_src:x_src+w_src_box]

        # Coordenadas objetivo (target)
        target_box = target_result.detections[0].location_data.relative_bounding_box
        h_tgt, w_tgt, _ = target_img.shape
        x_tgt, y_tgt = int(target_box.xmin * w_tgt), int(target_box.ymin * h_tgt)
        w_tgt_box, h_tgt_box = int(target_box.width * w_tgt), int(target_box.height * h_tgt)

        # Ajustar cara del jugador al tamaño de la cara objetivo
        jugador_face_resized = cv2.resize(jugador_face, (w_tgt_box, h_tgt_box))

        # Reemplazar la cara objetivo por la del jugador
        target_img[y_tgt:y_tgt+h_tgt_box, x_tgt:x_tgt+w_tgt_box] = jugador_face_resized

        return target_img

@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    if 'image' not in request.files:
        return 'No se envió ninguna imagen', 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    source_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    resultados = []

    # Procesar cada imagen en la carpeta del servidor
    for filename in os.listdir(IMAGE_FOLDER):
        target_path = os.path.join(IMAGE_FOLDER, filename)
        target_img = cv2.imread(target_path)

        swapped_img = face_swap(source_img, target_img)
        if swapped_img is None:
            continue

        # Guarda temporalmente la imagen modificada
        _, buffer = cv2.imencode('.jpg', swapped_img)
        resultados.append(buffer.tobytes())

    # Por simplicidad, devolvemos solo la primera imagen procesada.
    # Puedes adaptar esto a tu lógica (elegir varias, enviarlas en un ZIP, etc.)
    if resultados:
        io_buf = BytesIO(resultados[0])
        return send_file(io_buf, mimetype='image/jpeg')

    return 'No se pudieron procesar imágenes', 400

if __name__ == "__main__":
    app.run(debug=True)
