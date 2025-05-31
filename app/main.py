import os, io
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from PIL import Image

app = FastAPI()

@app.post("/procesar")
async def procesar(image: UploadFile):
	img = Image.open(await image.read()).convert("RGB")
	img = img.rotate(45)     # ← demo, cambia por tu face-swap
	buf = io.BytesIO()
	img.save(buf, format="PNG")
	return Response(buf.getvalue(), media_type="image/png")