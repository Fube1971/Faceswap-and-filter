from flask import Flask, request, send_file
import replicate
import tempfile
import os
import uuid
from io import BytesIO
import requests

app = Flask(__name__)

# Usa el token de entorno en Render
replicate_token = os.environ.get("REPLICATE_API_TOKEN")
if replicate_token is None:
    raise RuntimeError("Falta el token de Replicate en las variables de entorno")

os.environ["REPLICATE_API_TOKEN"] = replicate_token

@app.route('/procesar', methods=['POST'])
def procesar():
    if 'image' not in request.files or 'base' not in request.files:
        return "Faltan imágenes", 400

    # Guarda las imágenes temporales
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as user_file, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:

        user_file.write(request.files['image'].read())
        base_file.write(request.files['base'].read())

    # Ejecutar modelo
    output = replicate.run(
        "zsxkib/instant-id",
        input={
            "image": open(base_file.name, "rb"),
            "input_image": open(user_file.name, "rb"),
            "enhance_definitions": True,
            "prompt": "fotografía realista"
        }
    )

    if not output:
        return "Error al generar imagen", 500

    # Descarga la imagen generada (URL)
    image_response = requests.get(output)
    return send_file(BytesIO(image_response.content), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run()
