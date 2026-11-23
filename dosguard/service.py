"""
dosguard/service.py
====================
API de demostración local, pensada únicamente para generar tráfico HTTP
que el dashboard pueda analizar (pestaña "Servicio API"). No es un clon de
ningún sitio real: `/api/v1/boletos` es un inventario ficticio de solo
lectura, inventado para este prototipo, sin nombres ni marcas de terceros.

Qué SÍ hace: expone dos endpoints, limita la tasa de solicitudes por IP y
registra cada solicitud (admitida o no) en un CSV, con la IP
seudonimizada.

Qué NO hace, a propósito: no procesa compras ni pagos, no tiene base de
datos, no persiste nada más que el log de eventos, no abre puertos salvo
127.0.0.1, y no genera ni reenvía tráfico hacia ningún otro host.
"""

from __future__ import annotations

import csv
import hashlib
import os
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# --- configuración, ajustable por variable de entorno ----------------------
REQUEST_LIMIT = int(os.environ.get("DOSGUARD_REQUEST_LIMIT", "60"))
WINDOW_SECONDS = 60.0
TRUSTED_PROXIES = {
    ip.strip()
    for ip in os.environ.get("DOSGUARD_TRUSTED_PROXIES", "").split(",")
    if ip.strip()
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
EVENTS_PATH = DATA_DIR / "http_events.csv"
EVENTS_HEADER = ["timestamp_utc", "client_id", "method", "path", "status", "latency_ms"]

# Inventario ficticio de solo lectura. Nombres y datos inventados para este
# prototipo académico; no representan ningún producto ni empresa real.
_DEMO_INVENTORY = [
    {"id": 1, "evento": "Demo Jazz Night", "sede": "Auditorio de prueba", "disponibles": 42},
    {"id": 2, "evento": "Demo Tech Talks", "sede": "Sala de conferencias ficticia", "disponibles": 120},
    {"id": 3, "evento": "Demo Teatro Local", "sede": "Foro de pruebas", "disponibles": 8},
]


def _pseudonymize(client_ip: str) -> str:
    """SHA-256 truncado del IP: no queda la IP en claro en el log, pero el
    mismo cliente produce siempre el mismo identificador dentro de esta
    ejecución (útil para contar solicitudes por cliente sin guardar su
    dirección). El informe es explícito en que esto no es anonimato
    irreversible -- el identificador es correlacionable."""
    digest = hashlib.sha256(client_ip.encode("utf-8")).hexdigest()
    return digest[:16]


def _client_ip(request: Request) -> str:
    """La IP directa de la conexión, salvo que coincida exactamente con un
    proxy de confianza declarado en DOSGUARD_TRUSTED_PROXIES -- en ese caso
    se respeta X-Forwarded-For. Sin esa coincidencia exacta, el encabezado
    se ignora por completo: cualquier cliente podría mandarlo con un valor
    arbitrario."""
    direct_ip = request.client.host if request.client else "unknown"
    if direct_ip in TRUSTED_PROXIES:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return direct_ip


def _log_event(client_id: str, method: str, path: str, status: int, latency_ms: float) -> None:
    is_new = not EVENTS_PATH.exists()
    with EVENTS_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(EVENTS_HEADER)
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                client_id,
                method,
                path,
                status,
                f"{latency_ms:.2f}",
            ]
        )


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """Ventana deslizante de REQUEST_LIMIT solicitudes por WINDOW_SECONDS,
    por cliente (identificado por su IP seudonimizada). Vive en memoria de
    este proceso -- reiniciar el servicio borra el historial, y si algún
    día se corriera con varios workers no compartirían el límite entre
    ellos. Eso es exactamente lo que el informe marca como limitación: no
    es un limitador perimetral ni protección contra DDoS real."""

    def __init__(self, app):
        super().__init__(app)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = Lock()

    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        client_ip = _client_ip(request)
        client_id = _pseudonymize(client_ip)
        now = time.time()

        with self._lock:
            window = self._hits[client_id]
            while window and now - window[0] > WINDOW_SECONDS:
                window.popleft()
            allowed = len(window) < REQUEST_LIMIT
            if allowed:
                window.append(now)

        if not allowed:
            latency_ms = (time.monotonic() - start) * 1000
            _log_event(client_id, request.method, request.url.path, 429, latency_ms)
            return JSONResponse(
                {"detail": "Demasiadas solicitudes. Intenta de nuevo más tarde."},
                status_code=429,
                headers={"Retry-After": str(int(WINDOW_SECONDS))},
            )

        response = await call_next(request)
        latency_ms = (time.monotonic() - start) * 1000
        _log_event(client_id, request.method, request.url.path, response.status_code, latency_ms)
        return response


app = FastAPI(title="DOSGuard Demo API", version="1.0")
app.add_middleware(SlidingWindowRateLimiter)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/v1/boletos")
def boletos() -> list[dict]:
    """Inventario ficticio de solo lectura. No acepta compras ni
    modificaciones: es tráfico de demostración para la pestaña
    'Servicio API' del dashboard."""
    return _DEMO_INVENTORY
