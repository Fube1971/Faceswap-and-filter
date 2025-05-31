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
    "zsxkib/instant-id:2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789",
    input={
        "image": open(base_file.name, "rb"),         # Imagen del banco
        "pose_image": open(user_file.name, "rb"),    # Foto del jugador
        "prompt": "realistic face swap",             # Puedes personalizar esto
        "guidance_scale": 5,
        "sdxl_weights": "protovision-xl-high-fidel",
        "enhance_definitions": True,
        "negative_prompt": "(lowres, glitch, watermark)"
        }
    )


    if not output:
        return "Error al generar imagen", 500

    # Descarga la imagen generada (URL)
    image_response = requests.get(output)
    return send_file(BytesIO(image_response.content), mimetype="image/jpeg")

if __name__ == "__main__":
    app.run()
