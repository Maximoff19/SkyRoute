# Backend – Guía rápida de ejecución

Este documento explica paso a paso cómo levantar el backend (FastAPI) en tu máquina.

## Requisitos
- Python 3.12 (recomendado) o 3.11
  - macOS (Apple Silicon/Intel): `brew install python@3.12`
- Acceso a internet para instalar dependencias con `pip`

## 1) Clonar o ubicarse en el proyecto
```
cd "/Users/batimoff/Proyectos/agente viajero"
```

## 2) Crear y activar el entorno virtual
```
python3.12 -m venv venv
source venv/bin/activate
```
Si no tienes `python3.12`, usa `python3 -m venv venv` con tu versión 3.11/3.12.

## 3) Instalar dependencias
```
pip install --upgrade pip
pip install -r backend/requirements.txt
```

## 4) Ejecutar el servidor (foreground)
```
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```
- API viva en: `http://127.0.0.1:8000`
- Docs interactivos (Swagger): `http://127.0.0.1:8000/docs`

## 5) (Opcional) Ejecutar en segundo plano
```
nohup python -m uvicorn backend.main:app \
  --host 127.0.0.1 --port 8000 \
  > /tmp/agente_viajero_backend.log 2>&1 & echo $! > /tmp/agente_viajero_backend.pid
```
- Ver logs: `tail -f /tmp/agente_viajero_backend.log`
- Detener: `kill $(cat /tmp/agente_viajero_backend.pid)`

## 6) Probar rápidamente
- Países: `curl http://127.0.0.1:8000/countries`
- Resolver TSP (ejemplo):
```
curl -s -X POST http://127.0.0.1:8000/solve_tsp \
  -H 'Content-Type: application/json' \
  -d '{"country_keys":["spain","france","germany","italy"]}' | jq .
```

## Notas
- El backend usa fuerza bruta con límites adaptativos para evitar tiempos largos en conjuntos grandes de países.
- CORS está habilitado para que el `frontend/index.html` pueda consumir la API directamente.
- Si ves errores de `pydantic-core` compilando en Python 3.14, usa Python 3.12/3.11 como se indica arriba.

