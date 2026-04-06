from fastapi import FastAPI, Query, Response
from contextlib import asynccontextmanager
import httpx
from PIL import Image
import numpy as np
from io import BytesIO
from datetime import datetime, timezone, timedelta
import asyncio
import logging


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global latest_gibs_time
    detected = await _probe_latest_time()
    if detected:
        latest_gibs_time = detected
        logger.info("Imagem GIBS inicial: %s", latest_gibs_time)
    task = asyncio.create_task(_background_probe())
    yield
    task.cancel()
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()


app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("uvicorn.error")

GOES_URL = (
    "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
    "GOES-East_ABI_Band13_Clean_Infrared/default/"
    "{time}/"
    "GoogleMapsCompatible_Level6/{z}/{y}/{x}.png"
)

# Tile usada apenas para probing — tile central da cobertura GOES-East
PROBE_Z, PROBE_X, PROBE_Y = 3, 2, 4
PROBE_ROW = (2 ** PROBE_Z - 1) - PROBE_Y  # converte Y XYZ → row GIBS

DEFAULT_SATURATION_THRESHOLD = 0.15
MAX_SUPPORTED_ZOOM = 6
MAX_CACHE_TILES = 500
PROBE_INTERVAL_SECONDS = 120  # verifica nova imagem a cada 2 min
CACHE_CONTROL_MAX_AGE = 120   # Flutter cacheia tile por 2 min

# Estado global — atualizado pelo probe em background
latest_gibs_time: str = ""
tile_cache: dict[tuple, bytes] = {}

# Client HTTP reutilizado (evita overhead de conexão por request)
_http_client: httpx.AsyncClient | None = None


def _snap_to_10min(dt: datetime) -> datetime:
    """Arredonda para baixo ao múltiplo de 10 minutos."""
    return dt.replace(minute=(dt.minute // 10) * 10, second=0, microsecond=0)


def _format_time(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _transparent_tile() -> bytes:
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _process_image(content: bytes, saturation_threshold: float) -> bytes:
    img = Image.open(BytesIO(content)).convert("RGBA")
    data = np.array(img)
    rgb = data[..., :3].astype(np.float32) / 255.0

    # Calcula saturação HSV: pixels sem cor (fundo escuro ou cinza do IR)
    # têm saturação baixa; dados meteorológicos reais têm cores do colormap.
    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)
    saturation = np.where(max_rgb == 0, 0.0, (max_rgb - min_rgb) / max_rgb)

    # Remove todos os pixels neutros (sem cor), independente de brilho.
    data[..., 3][saturation <= saturation_threshold] = 0

    buf = BytesIO()
    Image.fromarray(data.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


async def _get_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=10, follow_redirects=True)
    return _http_client


async def _probe_latest_time() -> str | None:
    """
    Consulta o GIBS com o slot de 10 min mais recente.
    Retorna o Layer-Time-Actual se a imagem existir, None caso contrário.
    GOES-East tem latência ~20 min, então inicia a busca 20 min atrás.
    """
    client = await _get_client()
    now = datetime.now(timezone.utc)

    # Testa os últimos 6 slots (60 min) do mais recente para o mais antigo
    for offset_10min in range(2, 8):
        candidate = _snap_to_10min(now - timedelta(minutes=offset_10min * 10))
        time_str = _format_time(candidate)
        url = GOES_URL.format(z=PROBE_Z, x=PROBE_X, y=PROBE_ROW, time=time_str)
        try:
            resp = await client.get(url)
            if resp.status_code == 200:
                actual = resp.headers.get("layer-time-actual", "")
                return actual or time_str
        except httpx.HTTPError:
            pass

    return None


async def _background_probe():
    """Task em background: detecta nova imagem no GIBS e invalida cache."""
    global latest_gibs_time, tile_cache

    while True:
        try:
            detected = await _probe_latest_time()
            if detected and detected != latest_gibs_time:
                logger.info("Nova imagem GIBS detectada: %s (anterior: %s)", detected, latest_gibs_time)
                latest_gibs_time = detected
                tile_cache.clear()
        except Exception:
            logger.exception("Erro no probe GIBS")

        await asyncio.sleep(PROBE_INTERVAL_SECONDS)



@app.get("/tiles/{z}/{x}/{y}.png")
async def get_tile(
    z: int,
    x: int,
    y: int,
    saturation_threshold: float = Query(DEFAULT_SATURATION_THRESHOLD, ge=0.0, le=1.0),
):
    if z > MAX_SUPPORTED_ZOOM:
        return Response(content=_transparent_tile(), media_type="image/png")

    time_str = latest_gibs_time
    if not time_str:
        return Response(content=_transparent_tile(), media_type="image/png")

    cache_key = (time_str, z, x, y, round(saturation_threshold, 3))
    cached = tile_cache.get(cache_key)
    if cached:
        return Response(
            content=cached,
            media_type="image/png",
            headers={"Cache-Control": f"max-age={CACHE_CONTROL_MAX_AGE}"},
        )

    # Inverte eixo Y (XYZ → row GIBS TMS)
    tile_row = (2 ** z - 1) - y
    url = GOES_URL.format(z=z, x=x, y=tile_row, time=time_str)

    try:
        client = await _get_client()
        response = await client.get(url)

        if response.status_code != 200:
            logger.warning("GIBS %s para %s", response.status_code, url)
            return Response(content=_transparent_tile(), media_type="image/png")

        processed = _process_image(response.content, saturation_threshold)

        if len(tile_cache) >= MAX_CACHE_TILES:
            # Remove entradas mais antigas quando cache enche
            oldest_key = next(iter(tile_cache))
            del tile_cache[oldest_key]

        tile_cache[cache_key] = processed

        return Response(
            content=processed,
            media_type="image/png",
            headers={"Cache-Control": f"max-age={CACHE_CONTROL_MAX_AGE}"},
        )

    except httpx.HTTPError as exc:
        logger.warning("Erro HTTP ao buscar tile %s: %s", url, exc)
        return Response(content=_transparent_tile(), media_type="image/png")
    except Exception:
        logger.exception("Erro inesperado ao processar tile %s", url)
        return Response(content=_transparent_tile(), media_type="image/png")


@app.get("/status")
async def status():
    """Endpoint de diagnóstico: mostra o estado atual do cache e da imagem."""
    return {
        "latest_gibs_time": latest_gibs_time,
        "cached_tiles": len(tile_cache),
        "max_cache_tiles": MAX_CACHE_TILES,
    }
