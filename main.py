from fastapi import FastAPI, Response
import httpx
from PIL import Image
import numpy as np
from io import BytesIO
from functools import lru_cache
from datetime import datetime, timezone, timedelta

app = FastAPI()

GOES_URL = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    "GOES-East_ABI_Band13_Clean_Infrared/default/"
    "{time}/"
    "GoogleMapsCompatible_Level6/{z}/{y}/{x}.png"
)

def get_time():
    now = datetime.now(timezone.utc) - timedelta(hours=1)
    return now.strftime("%Y-%m-%dT%H:00:00Z")


# ----------------------------------------
# 🎨 PROCESSAMENTO SEM OPENCV
# ----------------------------------------
def process_image(content: bytes) -> bytes:
    img = Image.open(BytesIO(content)).convert("RGBA")
    data = np.array(img)

    r, g, b, a = data.T

    # normaliza RGB (0–1)
    r_n = r / 255.0
    g_n = g / 255.0
    b_n = b / 255.0

    # calcula saturação (aproximação HSV)
    max_rgb = np.maximum(np.maximum(r_n, g_n), b_n)
    min_rgb = np.minimum(np.minimum(r_n, g_n), b_n)

    saturation = np.where(max_rgb == 0, 0, (max_rgb - min_rgb) / max_rgb)

    # 🎯 remove tons neutros
    mask = saturation < 0.2  # ajuste aqui

    # aplica transparência
    data[..., -1][mask.T] = 0

    # leve boost de cor
    data[..., :3] = np.clip(data[..., :3] * 1.1, 0, 255)

    img = Image.fromarray(data.astype(np.uint8))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@lru_cache(maxsize=500)
def process_cached(content: bytes) -> bytes:
    return process_image(content)


@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(z: int, x: int, y: int):
    time_str = get_time()
    # 👇 inverter eixo Y (ESSENCIAL)
    tile_matrix = z
    tile_row = (2 ** z - 1) - y
    tile_col = x

    url = GOES_URL.format(
        z=tile_matrix,
        x=tile_col,
        y=tile_row,
        time=time_str
    )

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url)

        if response.status_code != 200:
            raise Exception("Tile não encontrado")

        processed = process_cached(response.content)

        return Response(content=processed, media_type="image/png")

    except:
        # fallback transparente
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")