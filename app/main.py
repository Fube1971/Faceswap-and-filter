from flask import Flask, request, send_file
import cv2
import numpy as np
import face_recognition
from io import BytesIO

app = Flask(__name__)

# Imagen guardada en servidor para intercambio de cara
TARGET_IMAGE_PATH = "target_face.jpg"  # Debes poner aquí la imagen con la cara a poner

@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    if 'image' not in request.files:
        return 'No se envió ninguna imagen', 400

    file = request.files['image']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    target_img = face_recognition.load_image_file(TARGET_IMAGE_PATH)

    # Detección de caras
    source_faces = face_recognition.face_locations(img)
    target_faces = face_recognition.face_locations(target_img)

    if not source_faces or not target_faces:
        return 'No se detectaron suficientes caras', 400

    source_face_encoding = face_recognition.face_encodings(img, source_faces)[0]
    target_face_encoding = face_recognition.face_encodings(target_img, target_faces)[0]

    # Intercambio básico de caras usando landmark points
    source_landmarks = face_recognition.face_landmarks(img)[0]
    target_landmarks = face_recognition.face_landmarks(target_img)[0]

    # Simplemente copiamos la cara del target al source (ejemplo básico)
    x1, y1, x2, y2 = source_faces[0][3], source_faces[0][0], source_faces[0][1], source_faces[0][2]
    face_width, face_height = x2 - x1, y2 - y1

    target_x1, target_y1, target_x2, target_y2 = target_faces[0][3], target_faces[0][0], target_faces[0][1], target_faces[0][2]
    target_face = target_img[target_y1:target_y2, target_x1:target_x2]

    target_face_resized = cv2.resize(target_face, (face_width, face_height))

    img[y1:y2, x1:x2] = target_face_resized

    # Codifica la imagen a JPEG y devuelve
    _, buffer = cv2.imencode('.jpg', img)
    io_buf = BytesIO(buffer)

    return send_file(io_buf, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
