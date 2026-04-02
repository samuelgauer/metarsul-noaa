from fastapi import FastAPI, Query, Response
import httpx
from PIL import Image
import numpy as np
from io import BytesIO
from functools import lru_cache
from datetime import datetime, timezone, timedelta
import logging

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

GOES_URL = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    "GOES-East_ABI_Band13_Clean_Infrared/default/"
    "{time}/"
    "GoogleMapsCompatible_Level6/{z}/{y}/{x}.png"
)

DEFAULT_SATURATION_THRESHOLD = 0.18
DEFAULT_BRIGHTNESS_THRESHOLD = 0.42
DEFAULT_WHITE_THRESHOLD = 0.78
DEFAULT_COLOR_BOOST = 1.0
MAX_SUPPORTED_ZOOM = 6


def get_time():
    now = datetime.now(timezone.utc) - timedelta(hours=1)
    return now.strftime("%Y-%m-%dT%H:00:00Z")


def transparent_tile() -> bytes:
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def process_image(
    content: bytes,
    saturation_threshold: float,
    brightness_threshold: float,
    white_threshold: float,
    color_boost: float,
) -> bytes:
    img = Image.open(BytesIO(content)).convert("RGBA")
    data = np.array(img)
    rgb = data[..., :3].astype(np.float32) / 255.0

    # Usa saturacao + brilho para remover fundo neutro sem apagar areas coloridas.
    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)
    color_range = max_rgb - min_rgb
    saturation = np.where(max_rgb == 0, 0, color_range / max_rgb)
    luminance = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    neutral_pixels = saturation <= saturation_threshold
    bright_neutral_pixels = neutral_pixels & (luminance >= brightness_threshold)
    almost_white_pixels = np.all(rgb >= white_threshold, axis=2)
    transparent_pixels = bright_neutral_pixels | almost_white_pixels

    data[..., 3][transparent_pixels] = 0

    if color_boost != 1.0:
        boosted = np.clip(data[..., :3].astype(np.float32) * color_boost, 0, 255)
        data[..., :3] = boosted.astype(np.uint8)

    img = Image.fromarray(data.astype(np.uint8))

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@lru_cache(maxsize=500)
def process_cached(
    content: bytes,
    saturation_threshold: float,
    brightness_threshold: float,
    white_threshold: float,
    color_boost: float,
) -> bytes:
    return process_image(
        content,
        saturation_threshold=saturation_threshold,
        brightness_threshold=brightness_threshold,
        white_threshold=white_threshold,
        color_boost=color_boost,
    )


@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(
    z: int,
    x: int,
    y: int,
    saturation_threshold: float = Query(DEFAULT_SATURATION_THRESHOLD, ge=0.0, le=1.0),
    brightness_threshold: float = Query(DEFAULT_BRIGHTNESS_THRESHOLD, ge=0.0, le=1.0),
    white_threshold: float = Query(DEFAULT_WHITE_THRESHOLD, ge=0.0, le=1.0),
    color_boost: float = Query(DEFAULT_COLOR_BOOST, ge=0.5, le=3.0),
):
    time_str = get_time()

    if z > MAX_SUPPORTED_ZOOM:
        logger.info("Zoom %s fora do nivel suportado para o produto GOES", z)
        return Response(content=transparent_tile(), media_type="image/png")

    # Inverte o eixo Y do cliente XYZ para o row esperado pelo GIBS.
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
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as client:
            response = await client.get(url)

        if response.status_code != 200:
            logger.warning("GIBS retornou status %s para %s", response.status_code, url)
            return Response(content=transparent_tile(), media_type="image/png")

        processed = process_cached(
            response.content,
            saturation_threshold=round(saturation_threshold, 3),
            brightness_threshold=round(brightness_threshold, 3),
            white_threshold=round(white_threshold, 3),
            color_boost=round(color_boost, 3),
        )

        return Response(content=processed, media_type="image/png")

    except httpx.HTTPError as exc:
        logger.warning("Erro HTTP ao buscar tile %s: %s", url, exc)
        return Response(content=transparent_tile(), media_type="image/png")
    except Exception:
        logger.exception("Erro inesperado ao processar tile %s", url)
        return Response(content=transparent_tile(), media_type="image/png")