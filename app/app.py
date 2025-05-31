from flask import Flask, request, send_file
import replicate
import tempfile
import os
import requests
from io import BytesIO
import logging
import traceback

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Leer el token desde variable de entorno
replicate_token = os.environ.get("REPLICATE_API_TOKEN")
if not replicate_token:
    raise RuntimeError("Falta el token de Replicate (REPLICATE_API_TOKEN)")

os.environ["REPLICATE_API_TOKEN"] = replicate_token

@app.route('/')
def root():
    return "✅ FaceSwap server running", 200

@app.route('/procesar', methods=['POST'])
def procesar():
    try:
        if 'image' not in request.files or 'base' not in request.files:
            logging.error("❌ Archivos 'image' y/o 'base' no fueron enviados.")
            return "Faltan archivos", 400

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as pose_file, \
             tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:

            pose_file.write(request.files['image'].read())
            base_file.write(request.files['base'].read())
            pose_file.flush()
            base_file.flush()

        logging.info("📤 Enviando imágenes a Replicate...")

        output = replicate.run(
            "zsxkib/instant-id:2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789",
            input={
                "image": open(base_file.name, "rb"),
                "pose_image": open(pose_file.name, "rb"),
                "prompt": "realistic photo",
                "guidance_scale": 5,
                "sdxl_weights": "protovision-xl-high-fidel",
                "enhance_definitions": True,
                "negative_prompt": "(lowres, glitch, watermark)"
            }
        )

        if not output:
            logging.error("❌ Replicate no devolvió salida.")
            return "No se generó imagen", 500

        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)

        if response.status_code != 200:
            logging.error(f"❌ Error al descargar imagen generada: {response.status_code}")
            return "Error al descargar imagen generada", 500

        logging.info("✅ Imagen generada con éxito.")
        return send_file(BytesIO(response.content), mimetype="image/jpeg")

    except Exception as e:
        logging.error("💥 Excepción capturada:\n" + traceback.format_exc())
        return f"Error del servidor: {str(e)}", 500

if __name__ == "__main__":
    app.run(debug=True)
