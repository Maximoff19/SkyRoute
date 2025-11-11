"""
Autores (2025):
- Osores Marchese, Pietro — u202310971
- Retuerto Zapata, Renzo Paul — u202320328
- Gómez De La Torre Huertas, Rodrigo — u202311464
- Almeida Mora, Rodrigo Fernando — u20211g708
"""

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Tuple, Iterable, Optional
import math
import itertools
import time
import json

app = FastAPI(title="TSP Solver API", version="1.6.0")

# Configurar CORS para permitir requests desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Países y distancias predefinidas (km)
# =============================

COUNTRIES: Dict[str, Dict[str, str]] = {
    "spain": {"name": "España"},
    "portugal": {"name": "Portugal"},
    "france": {"name": "Francia"},
    "germany": {"name": "Alemania"},
    "italy": {"name": "Italia"},
    "uk": {"name": "Reino Unido"},
    "norway": {"name": "Noruega"},

    "usa": {"name": "Estados Unidos"},
    "canada": {"name": "Canadá"},
    "mexico": {"name": "México"},
    "costarica": {"name": "Costa Rica"},
    "colombia": {"name": "Colombia"},
    "peru": {"name": "Perú"},
    "chile": {"name": "Chile"},
    "brazil": {"name": "Brasil"},
    "argentina": {"name": "Argentina"},
}

COUNTRY_KEYS: List[str] = list(COUNTRIES.keys())
COUNTRY_INDEX: Dict[str, int] = {key: idx for idx, key in enumerate(COUNTRY_KEYS)}

# Distancias predefinidas (grafo completo y simétrico)
DISTANCES_KM: Dict[str, Dict[str, float]] = {
    "spain": {'spain': 0.0, 'portugal': 600.0, 'france': 1050.0, 'germany': 1800.0, 'italy': 1360.0, 'uk': 1300.0, 'norway': 3000.0, 'usa': 7000.0, 'canada': 5600.0, 'mexico': 9000.0, 'costarica': 5500.0, 'colombia': 8000.0, 'peru': 9500.0, 'chile': 5650.0, 'brazil': 8000.0, 'argentina': 10000.0},
    "portugal": {'spain': 600.0, 'portugal': 0.0, 'france': 1200.0, 'germany': 5100.0, 'italy': 5150.0, 'uk': 1600.0, 'norway': 5250.0, 'usa': 5300.0, 'canada': 5350.0, 'mexico': 5400.0, 'costarica': 5450.0, 'colombia': 5500.0, 'peru': 5550.0, 'chile': 5600.0, 'brazil': 5650.0, 'argentina': 5700.0},
    "france": {'spain': 1050.0, 'portugal': 1200.0, 'france': 0.0, 'germany': 1000.0, 'italy': 1100.0, 'uk': 450.0, 'norway': 2000.0, 'usa': 7200.0, 'canada': 5400.0, 'mexico': 5350.0, 'costarica': 5400.0, 'colombia': 5450.0, 'peru': 5500.0, 'chile': 5550.0, 'brazil': 8700.0, 'argentina': 11000.0},
    "germany": {'spain': 1800.0, 'portugal': 5100.0, 'france': 1000.0, 'germany': 0.0, 'italy': 1100.0, 'uk': 900.0, 'norway': 1400.0, 'usa': 7500.0, 'canada': 5250.0, 'mexico': 5300.0, 'costarica': 5350.0, 'colombia': 5400.0, 'peru': 5450.0, 'chile': 5500.0, 'brazil': 5550.0, 'argentina': 5600.0},
    "italy": {'spain': 1360.0, 'portugal': 5150.0, 'france': 1100.0, 'germany': 1100.0, 'italy': 0.0, 'uk': 5050.0, 'norway': 2500.0, 'usa': 5150.0, 'canada': 5200.0, 'mexico': 5250.0, 'costarica': 5300.0, 'colombia': 5350.0, 'peru': 5400.0, 'chile': 5450.0, 'brazil': 5500.0, 'argentina': 5550.0},
    "uk": {'spain': 1300.0, 'portugal': 1600.0, 'france': 450.0, 'germany': 900.0, 'italy': 5050.0, 'uk': 0.0, 'norway': 1200.0, 'usa': 6800.0, 'canada': 5200.0, 'mexico': 5200.0, 'costarica': 5250.0, 'colombia': 5300.0, 'peru': 5350.0, 'chile': 5400.0, 'brazil': 5450.0, 'argentina': 5500.0},
    "norway": {'spain': 3000.0, 'portugal': 5250.0, 'france': 2000.0, 'germany': 1400.0, 'italy': 2500.0, 'uk': 1200.0, 'norway': 0.0, 'usa': 6500.0, 'canada': 5600.0, 'mexico': 8500.0, 'costarica': 8800.0, 'colombia': 8200.0, 'peru': 9800.0, 'chile': 5350.0, 'brazil': 9800.0, 'argentina': 12000.0},
    "usa": {'spain': 7000.0, 'portugal': 5300.0, 'france': 7200.0, 'germany': 7500.0, 'italy': 5150.0, 'uk': 6800.0, 'norway': 6500.0, 'usa': 0.0, 'canada': 800.0, 'mexico': 2500.0, 'costarica': 3500.0, 'colombia': 4000.0, 'peru': 5250.0, 'chile': 5300.0, 'brazil': 7000.0, 'argentina': 9000.0},
    "canada": {'spain': 5600.0, 'portugal': 5350.0, 'france': 5400.0, 'germany': 5250.0, 'italy': 5200.0, 'uk': 5200.0, 'norway': 5600.0, 'usa': 800.0, 'canada': 0.0, 'mexico': 3500.0, 'costarica': 5100.0, 'colombia': 5150.0, 'peru': 5200.0, 'chile': 5250.0, 'brazil': 5300.0, 'argentina': 5350.0},
    "mexico": {'spain': 9000.0, 'portugal': 5400.0, 'france': 5350.0, 'germany': 5300.0, 'italy': 5250.0, 'uk': 5200.0, 'norway': 8500.0, 'usa': 2500.0, 'canada': 3500.0, 'mexico': 0.0, 'costarica': 1600.0, 'colombia': 3000.0, 'peru': 5150.0, 'chile': 5200.0, 'brazil': 6500.0, 'argentina': 7400.0},
    "costarica": {'spain': 5500.0, 'portugal': 5450.0, 'france': 5400.0, 'germany': 5350.0, 'italy': 5300.0, 'uk': 5250.0, 'norway': 8800.0, 'usa': 3500.0, 'canada': 5100.0, 'mexico': 1600.0, 'costarica': 0.0, 'colombia': 1200.0, 'peru': 2600.0, 'chile': 5150.0, 'brazil': 5200.0, 'argentina': 5250.0},
    "colombia": {'spain': 8000.0, 'portugal': 5500.0, 'france': 5450.0, 'germany': 5400.0, 'italy': 5350.0, 'uk': 5300.0, 'norway': 8200.0, 'usa': 4000.0, 'canada': 5150.0, 'mexico': 3000.0, 'costarica': 1200.0, 'colombia': 0.0, 'peru': 1800.0, 'chile': 5100.0, 'brazil': 4300.0, 'argentina': 5800.0},
    "peru": {'spain': 9500.0, 'portugal': 5550.0, 'france': 5500.0, 'germany': 5450.0, 'italy': 5400.0, 'uk': 5350.0, 'norway': 9800.0, 'usa': 5250.0, 'canada': 5200.0, 'mexico': 5150.0, 'costarica': 2600.0, 'colombia': 1800.0, 'peru': 0.0, 'chile': 2450.0, 'brazil': 3300.0, 'argentina': 3100.0},
    "chile": {'spain': 5650.0, 'portugal': 5600.0, 'france': 5550.0, 'germany': 5500.0, 'italy': 5450.0, 'uk': 5400.0, 'norway': 5350.0, 'usa': 5300.0, 'canada': 5250.0, 'mexico': 5200.0, 'costarica': 5150.0, 'colombia': 5100.0, 'peru': 2450.0, 'chile': 0.0, 'brazil': 3000.0, 'argentina': 1400.0},
    "brazil": {'spain': 8000.0, 'portugal': 5650.0, 'france': 8700.0, 'germany': 5550.0, 'italy': 5500.0, 'uk': 5450.0, 'norway': 9800.0, 'usa': 7000.0, 'canada': 5300.0, 'mexico': 6500.0, 'costarica': 5200.0, 'colombia': 4300.0, 'peru': 3300.0, 'chile': 3000.0, 'brazil': 0.0, 'argentina': 2900.0},
    "argentina": {'spain': 10000.0, 'portugal': 5700.0, 'france': 11000.0, 'germany': 5600.0, 'italy': 5550.0, 'uk': 5500.0, 'norway': 12000.0, 'usa': 9000.0, 'canada': 5350.0, 'mexico': 7400.0, 'costarica': 5250.0, 'colombia': 5800.0, 'peru': 3100.0, 'chile': 1400.0, 'brazil': 2900.0, 'argentina': 0.0},
}

# =============================
# Modelos
# =============================

class SolveByKeysRequest(BaseModel):
    """
    Entrada para peticiones que operan sobre un subconjunto de países.

    - country_keys: lista de claves (strings) que deben existir en
      `COUNTRY_KEYS`. El primer elemento se toma como país de partida
      a efectos visuales en el frontend.
    """
    country_keys: List[str]

class Cycle(BaseModel):
    """
    Representa un ciclo hamiltoniano enumerado por fuerza bruta.

    - order: índices (sobre `COUNTRY_KEYS`) que describen el recorrido.
    - distance_km: distancia total del ciclo en kilómetros.
    - route_names: nombres legibles de los países en el orden dado.
    """
    order: List[int]
    distance_km: float
    route_names: List[str]

class SolveResponse(BaseModel):
    """
    Respuesta del endpoint `/solve_tsp` usando fuerza bruta.

    - order: índices (sin repetir el primero al final) del mejor ciclo.
    - distance_km: distancia total del mejor ciclo.
    - edges: detalle de tramos (nombres y distancia por arista).
    - algorithm_used: nombre del algoritmo empleado.
    - execution_time_ms: tiempo de cómputo en ms.
    - evaluated_cycles: cuántos ciclos se evaluaron (con o sin límite).
    - partial: True si se alcanzó un límite (resultado aproximado).
    - max_cycles: límite aplicado, o None si no hubo límite.
    """
    order: List[int]
    distance_km: float
    edges: List[Dict]
    algorithm_used: str
    execution_time_ms: float
    evaluated_cycles: int
    partial: bool
    max_cycles: Optional[int]

# =============================
# Utilidades de distancia por índices (tabla fija)
# =============================

def get_distance_idx(i: int, j: int) -> float:
    """
    Distancia (km) entre los países con índices `i` y `j`,
    consultando la tabla fija `DISTANCES_KM`.
    """
    ai = COUNTRY_KEYS[i]
    aj = COUNTRY_KEYS[j]
    return float(DISTANCES_KM[ai][aj])


def get_distance_path(order: List[int]) -> float:
    """
    Distancia total de un recorrido que visita los índices en `order`
    y cierra el ciclo (del último vuelve al primero).
    """
    n = len(order)
    total = 0.0
    for i in range(n):
        # Suma del tramo actual al siguiente; el último regresa al inicio
        total += get_distance_idx(order[i], order[(i + 1) % n])
    return total

def validate_all_pairs(indices: List[int]):
    """
    Verifica que existan distancias para todos los pares de países
    seleccionados. Actúa como salvaguarda si `DISTANCES_KM` es editado.
    """
    missing = []
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            ai = COUNTRY_KEYS[indices[i]]
            aj = COUNTRY_KEYS[indices[j]]
            # Comprobación unidireccional (ai -> aj); en grafo simétrico
            # también debería existir (aj -> ai)
            if aj not in DISTANCES_KM.get(ai, {}):
                missing.append((ai, aj))
    if missing:
        names = [f"{a}-{b}" for a, b in missing]
        raise HTTPException(status_code=400, detail=f"Faltan distancias para: {', '.join(names)}")

# =============================
# Fuerza bruta
# =============================

def brute_force_best(order_indices: List[int]) -> Tuple[List[int], float]:
    """Evalúa todas las permutaciones posibles y retorna la de menor distancia."""
    
    best_order = None                   # Guarda la mejor ruta encontrada (inicialmente ninguna)
    best_distance = math.inf            # Guarda la menor distancia (inicialmente infinito)
    
    # Recorre todas las permutaciones posibles de los nodos (todas las rutas posibles)
    for perm in itertools.permutations(order_indices):
        d = get_distance_path(list(perm))   # Calcula la distancia total de esta ruta
        if d < best_distance:               # Si la distancia es mejor que la mejor actual...
            best_order = list(perm)         # ...actualiza la mejor ruta
            best_distance = d               # ...y actualiza la mejor distancia
    
    return best_order, best_distance        # Devuelve la mejor ruta y su distancia

def iter_unique_cycles(order_indices: List[int]) -> Iterable[Tuple[List[int], float]]:
    """
    Genera ciclos hamiltonianos únicos (sin duplicados por rotación ni inversión)
    para el conjunto de índices indicado. Cada ciclo regresa al punto inicial.
    """
    if not order_indices:
        return
    if len(order_indices) == 1:
        start = order_indices[0]
        yield [start, start], 0.0
        return

    start = order_indices[0]
    others = order_indices[1:]
    for perm in itertools.permutations(others):
        if perm > perm[::-1]:
            continue
        base_route = [start, *perm]
        total = get_distance_path(base_route)
        yield base_route + [start], total


def brute_force_all_cycles(
    order_indices: List[int],
    *,
    max_cycles: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> List[Tuple[List[int], float]]:
    """
    Retorna una lista de ciclos hamiltonianos ordenados por distancia.
    Permite limitar la enumeración por número de ciclos o por tiempo.
    """
    if not order_indices:
        return []

    start_time = time.time()
    cycles: List[Tuple[List[int], float]] = []
    collected = 0

    for route, dist in iter_unique_cycles(order_indices):
        cycles.append((route, dist))
        collected += 1

        if max_cycles is not None and max_cycles > 0 and collected >= max_cycles:
            break
        if max_seconds is not None and max_seconds > 0:
            if time.time() - start_time >= max_seconds:
                break

    cycles.sort(key=lambda x: x[1])
    return cycles
"""
# Límites adaptativos para fuerza bruta
"""

def default_cycle_limit(n: int) -> Optional[int]:
    """Devuelve un límite de ciclos recomendado en función de n.
    Para n ≤ 8 no se limita; para n > 8 el límite decrece con n para
    mantener tiempos razonables.
    """
    if n <= 8:
        return None
    limits = {
        9: 100_000,
        10: 60_000,
        11: 40_000,
        12: 25_000,
        13: 15_000,
        14: 10_000,
        15: 7_000,
        16: 5_000,
    }
    return limits.get(n, 5_000)







# =============================
# Endpoints
# =============================


@app.get("/")
async def root():
    return {
        "message": "TSP Solver API - 16 países, distancias fijas, algoritmo de Fuerza Bruta con límites adaptativos",
        "algorithms": ["brute_force"],
        "countries": COUNTRY_KEYS,
        "count": len(COUNTRY_KEYS),
    }

@app.get("/countries")
async def get_countries():
    return {
        "countries": [
            {"key": key, "name": COUNTRIES[key]["name"]}
            for key in COUNTRY_KEYS
        ],
        "count": len(COUNTRY_KEYS),
    }

@app.post("/distances")
async def get_distances(req: SolveByKeysRequest) -> Dict:
    try:
        if len(req.country_keys) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 países")
        indices = []
        for key in req.country_keys:
            if key not in COUNTRY_INDEX:
                raise HTTPException(status_code=400, detail=f"País no válido: {key}")
            indices.append(COUNTRY_INDEX[key])
        validate_all_pairs(indices)
        names = [COUNTRIES[k]["name"] for k in req.country_keys]
        matrix = []
        for i in indices:
            row = []
            for j in indices:
                row.append(round(get_distance_idx(i, j), 2))
            matrix.append(row)
        return {"keys": req.country_keys, "names": names, "matrix": matrix}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener matriz: {str(e)}")

@app.post("/cycles")
async def get_cycles(
    req: SolveByKeysRequest,
    stream: bool = True,
    max_cycles: int = 0,
    max_seconds: float = 0.0,
) -> Response:
    try:
        if len(req.country_keys) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 países")
        indices = []
        for key in req.country_keys:
            if key not in COUNTRY_INDEX:
                raise HTTPException(status_code=400, detail=f"País no válido: {key}")
            indices.append(COUNTRY_INDEX[key])

        validate_all_pairs(indices)

        max_cycles = max(0, max_cycles)
        max_seconds = max(0.0, max_seconds)

        # Límite adaptativo por número de nodos si no se especifica
        adaptive_limit = default_cycle_limit(len(indices))
        applied_max_cycles: Optional[int] = None
        if max_cycles > 0:
            applied_max_cycles = max_cycles
        elif adaptive_limit is not None:
            applied_max_cycles = adaptive_limit

        if not stream:
            start_time = time.time()
            cycles = brute_force_all_cycles(
                indices,
                max_cycles=applied_max_cycles if applied_max_cycles and applied_max_cycles > 0 else None,
                max_seconds=max_seconds if max_seconds > 0 else None,
            )
            elapsed = time.time() - start_time
            exec_ms = round(elapsed * 1000.0, 2)
            result: List[Cycle] = []
            for route, dist in cycles:
                names = [COUNTRIES[COUNTRY_KEYS[i]]["name"] for i in route]
                result.append({
                    "order": route,
                    "distance_km": round(dist, 2),
                    "route_names": names,
                })
            partial = False
            if applied_max_cycles and len(result) == applied_max_cycles:
                partial = True
            if max_seconds > 0 and elapsed >= max_seconds:
                partial = True
            return JSONResponse({
                "cycles": result,
                "count": len(result),
                "execution_time_ms": exec_ms,
                "partial": partial,
            })

        def cycle_stream():
            start_time = time.time()
            count = 0
            aborted_by_time = False
            aborted_by_limit = False

            header = {
                "type": "start",
                "requested_keys": req.country_keys,
                "max_cycles": applied_max_cycles if applied_max_cycles and applied_max_cycles > 0 else None,
                "max_seconds": max_seconds if max_seconds > 0 else None,
            }
            yield json.dumps(header) + "\n"

            for route, dist in iter_unique_cycles(indices):
                elapsed = time.time() - start_time
                if max_seconds > 0 and elapsed >= max_seconds:
                    aborted_by_time = True
                    break

                count += 1
                names = [COUNTRIES[COUNTRY_KEYS[i]]["name"] for i in route]
                payload = {
                    "type": "cycle",
                    "index": count,
                    "order": route,
                    "distance_km": round(dist, 2),
                    "route_names": names,
                }
                yield json.dumps(payload) + "\n"

                if applied_max_cycles and count >= applied_max_cycles:
                    aborted_by_limit = True
                    break

            exec_ms = round((time.time() - start_time) * 1000.0, 2)
            summary = {
                "type": "stats",
                "count": count,
                "execution_time_ms": exec_ms,
                "partial": aborted_by_time or aborted_by_limit,
                "limit_reached": aborted_by_limit,
                "time_limit_reached": aborted_by_time,
            }
            yield json.dumps(summary) + "\n"

        return StreamingResponse(cycle_stream(), media_type="application/x-ndjson")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al obtener ciclos: {str(e)}")

@app.post("/solve_tsp", response_model=SolveResponse)
async def solve_tsp(req: SolveByKeysRequest):
    try:
        if len(req.country_keys) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 países")
        indices = []
        for key in req.country_keys:
            if key not in COUNTRY_INDEX:
                raise HTTPException(status_code=400, detail=f"País no válido: {key}")
            indices.append(COUNTRY_INDEX[key])

        validate_all_pairs(indices)

        n = len(indices)
        start_time = time.time()
        algorithm = "brute_force"

        # Ejecutar fuerza bruta con límite adaptativo según n
        applied_max_cycles = default_cycle_limit(n)
        evaluated_cycles = 0
        best_order: Optional[List[int]] = None
        best_distance = math.inf

        for route, dist in iter_unique_cycles(indices):
            evaluated_cycles += 1
            if dist < best_distance:
                best_distance = dist
                best_order = route
            if applied_max_cycles and evaluated_cycles >= applied_max_cycles:
                break

        # Si no se iteró nada (caso extremo), caer a mejor de todas las permutaciones
        if best_order is None:
            best_order, best_distance = brute_force_best(indices)
            evaluated_cycles = 0

        # Normalizar ruta (iter_unique_cycles incluye regreso al inicio)
        if best_order and len(best_order) >= 2 and best_order[0] == best_order[-1]:
            best_order = best_order[:-1]

        route_indices, dist = best_order, best_distance
        exec_ms = round((time.time() - start_time) * 1000.0, 2)
        edges: List[Dict] = []
        for i in range(len(route_indices)):
            a = route_indices[i]
            b = route_indices[(i + 1) % len(route_indices)]
            edges.append({
                "from_idx": i,
                "to_idx": (i + 1) % len(route_indices),
                "from_point": {
                    "name": COUNTRIES[COUNTRY_KEYS[a]]["name"],
                },
                "to_point": {
                    "name": COUNTRIES[COUNTRY_KEYS[b]]["name"],
                },
                "distance_km": round(get_distance_idx(a, b), 2),
            })
        return {
            "order": route_indices,
            "distance_km": round(dist, 2),
            "edges": edges,
            "algorithm_used": algorithm,
            "execution_time_ms": exec_ms,
            "evaluated_cycles": evaluated_cycles,
            "partial": bool(applied_max_cycles) and evaluated_cycles >= (applied_max_cycles or 0),
            "max_cycles": applied_max_cycles if applied_max_cycles and applied_max_cycles > 0 else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al resolver TSP: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
