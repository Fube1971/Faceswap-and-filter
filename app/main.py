from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from PIL import Image
import io

app = FastAPI()

@app.post("/procesar")
async def procesar(image: UploadFile = File(...)):
    # 1) leer bytes
    data = await image.read()
    # 2) abrir con PIL desde un buffer en memoria
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # 3) tu lógica (demo: rota 45°)
    img = img.rotate(45)

    # 4) guardar en BytesIO y devolver
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")
