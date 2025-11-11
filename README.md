# Agente Viajero – TSP Solver

## Autores (2025)
- Osores Marchese, Pietro — u202310971
- Retuerto Zapata, Renzo Paul — u202320328
- Gómez De La Torre Huertas, Rodrigo — u202311464
- Almeida Mora, Rodrigo Fernando — u20211g708

Aplicación que resuelve el Problema del Agente Viajero (TSP) sobre un conjunto de países predefinidos. Backend en Python (FastAPI) y frontend estático.

## Algoritmo

- Fuerza bruta con límites adaptativos: se enumeran ciclos hamiltonianos y se elige el más corto. Para n > 8 se limita la enumeración para mantener tiempos razonables. Elegido por su simplicidad, exactitud en tamaños pequeños y transparencia del resultado.

## Iniciar el backend

En la raíz del proyecto:

```bash
./install.sh
./start.sh
```
