# sistema_dos

Prototipo académico de detección y clasificación de ataques DoS mediante
aprendizaje automático. Dos procesos independientes:

- **Dashboard Streamlit** (`app.py`): carga datos (sintéticos, CSV o
  PCAP/PCAPNG), los normaliza a nueve variables y entrena/evalúa dos
  modelos (XGBoost supervisado, Isolation Forest no supervisado).
- **API de demostración** (`dosguard/service.py`): expone un endpoint
  ficticio de solo lectura con límite de solicitudes, para generar tráfico
  HTTP que la pestaña "Servicio API" del dashboard pueda analizar.

No capturan tráfico en vivo, no generan tráfico de ataque y no bloquean
nada automáticamente: son herramientas de análisis de archivos y
resultados, no un IDS/IPS.

## Ejecución local (Windows, PowerShell)

```powershell
cd sistema_dos
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# Terminal 1
streamlit run app.py --server.address 127.0.0.1 --server.port 8501

# Terminal 2 (opcional, para la pestaña "Servicio API")
uvicorn dosguard.service:app --host 127.0.0.1 --port 8000
```

Cada servidor corre en su propia terminal y se detiene con `Ctrl+C`.

## Ejecución local (Linux/macOS)

```bash
cd sistema_dos
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

streamlit run app.py --server.address 127.0.0.1 --server.port 8501
# en otra terminal:
uvicorn dosguard.service:app --host 127.0.0.1 --port 8000
```

## Pruebas

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
```

## Estructura

```
sistema_dos/
  core.py              # esquema, alias de encabezados, datos sintéticos
  ingest.py            # carga de CSV y PCAP/PCAPNG
  models.py            # XGBoost + Isolation Forest
  app.py               # dashboard Streamlit (6 pestañas)
  dosguard/
    service.py          # API de demostración + limitador de tasa
  tests/
    test_pipeline.py    # las 5 pruebas del informe técnico
  data/                 # aquí se escribe http_events.csv (se crea solo)
  requirements.txt
```

## Limitaciones conocidas (ver el informe técnico para el detalle completo)

- Sin captura en vivo ni generación de tráfico de ataque.
- Sin bloqueo automático de IPs ni integración con firewall/IDS reales.
- El limitador de tasa vive en memoria de un solo proceso: no es
  protección perimetral contra DDoS.
- Las métricas del clasificador sobre datos sintéticos no deben citarse
  como evidencia de detección real; los perfiles son deliberadamente
  separables.
