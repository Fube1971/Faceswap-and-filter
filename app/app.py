from flask import Flask, request, send_file
import replicate
import tempfile
import os
import requests
from io import BytesIO

app = Flask(__name__)

# Leer token desde variable de entorno
replicate_token = os.environ.get("REPLICATE_API_TOKEN")
if not replicate_token:
    raise RuntimeError("Falta el token de Replicate. Configura REPLICATE_API_TOKEN en Render.")
os.environ["REPLICATE_API_TOKEN"] = replicate_token

@app.route('/procesar', methods=['POST'])
def procesar():
    if 'image' not in request.files or 'base' not in request.files:
        return "Faltan 'image' o 'base' en la solicitud", 400

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as pose_file, \
         tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as base_file:

        pose_file.write(request.files['image'].read())
        base_file.write(request.files['base'].read())

    try:
        output = replicate.run(
            "zsxkib/instant-id:2e4785a4d80dadf580077b2244c8d7c05d8e3faac04a04c02d8e099dd2876789",
            input={
                "image": open(base_file.name, "rb"),         # Imagen base
                "pose_image": open(pose_file.name, "rb"),    # Foto jugador
                "prompt": "realistic photo",
                "guidance_scale": 5,
                "sdxl_weights": "protovision-xl-high-fidel",
                "enhance_definitions": True,
                "negative_prompt": "(lowres, glitch, watermark)"
            }
        )

        if not output:
            return "No se generó ninguna imagen", 500

        image_url = output[0] if isinstance(output, list) else output
        response = requests.get(image_url)

        if response.status_code != 200:
            return "Error al descargar imagen generada", 500

        return send_file(BytesIO(response.content), mimetype="image/jpeg")

    except replicate.exceptions.ReplicateError as e:
        return f"Error de Replicate: {str(e)}", 500
    except Exception as e:
    import traceback
    print("💥 Error general:", traceback.format_exc())
    return f"Error general: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
