# specpy

Visualizador interactivo de espectros estelares en formato FITS.

## Uso

```bash
spec.py <archivo.fits> [opciones]
```

### Opciones

| Flag | Descripción |
|------|-------------|
| `--window WMIN WMAX` | Recorta el espectro al rango `[WMIN, WMAX]` Å al cargar |
| `--params FILE` | Carga parámetros de ajuste desde un archivo JSON |
| `--normalized` | Entra directamente en modo normalización y sale al terminar |
| `--gaussian` | Entra directamente en modo ajuste de gaussianas y sale al terminar |

---

## Modos interactivos

### Visualizador principal

| Tecla | Acción |
|-------|--------|
| `w` | Activar SpanSelector para definir ventana de recorte |
| `Enter` | Aplicar recorte |
| `z` | Volver al espectro completo |
| `n` | Abrir modo normalización |
| `d` | Abrir modo ajuste de gaussianas |
| `x` | Guardar espectro actual como FITS (`_crop`, `_norm`) |
| `q` | Cerrar |
| — | — |
| `p` | Pan (arrastrar para mover) |
| `o` | Zoom (arrastrar para seleccionar región) |
| scroll | Zoom in/out |
| `home` | Reset vista al estado original |

---

### Modo normalización (`n`)

Ajuste de continuo por rangos con interpolación Akima.

**Flujo de trabajo:**
1. Presionar `a` para activar el SpanSelector y arrastrar para marcar regiones del continuo.
2. Se ajusta un polinomio Chebyshev con σ-clipping a los puntos seleccionados.
3. Presionar `b` para sellar el rango activo e iniciar uno nuevo (cada rango tiene su propio polinomio).
4. Con varios rangos, los polinomios se unen con una interpolación Akima para construir el continuo final.
5. Las zonas fuera de los rangos definidos no se normalizan.

| Tecla | Acción |
|-------|--------|
| `a` | Activar/desactivar SpanSelector de regiones |
| `b` | Sellar rango activo e iniciar uno nuevo |
| `e` | Eliminar última región (elimina el rango si queda vacío) |
| `+` / `-` | Subir/bajar orden del polinomio del rango activo |
| `q` | Confirmar y cerrar (pregunta si guardar) |

Con un solo rango el comportamiento es idéntico al ajuste clásico de un polinomio global.

---

### Modo ajuste de gaussianas (`g`)

Ajuste de líneas espectrales con modelo gaussiano + fondo constante (lmfit).

**Flujo de trabajo:**
1. Presionar `g` para iniciar la definición de una gaussiana.
2. Tres clics: centro (`x` = λ, `y` = profundidad), FWHM izquierdo, FWHM derecho.
3. Repetir para todas las líneas a ajustar.
4. Presionar `a` para ejecutar el ajuste automático.

| Tecla | Acción |
|-------|--------|
| `g` | Iniciar definición de nueva gaussiana |
| `a` | Ejecutar ajuste (lmfit) |
| `b` | Eliminar última gaussiana |
| `c` | Limpiar todas las gaussianas y el ajuste |
| `e` | Activar modo borrado de puntos |
| `r` (en modo borrado) | Restaurar todos los puntos eliminados |
| `q` | Cerrar y guardar resultados en CSV |

Al cerrar, muestra velocidades radiales y anchos equivalentes (EW) para cada gaussiana, e identifica la línea de reposo más cercana con `lines.csv`.

---

## Archivos de salida

| Archivo | Descripción |
|---------|-------------|
| `<nombre>_norm.fits` | Espectro normalizado |
| `<nombre>_crop.fits` | Espectro recortado |
| `<nombre>_crop_norm.fits` | Espectro recortado y normalizado |
| `fitted_<linea>.csv` | Parámetros del ajuste gaussiano (centro, σ, FWHM, EW, vr) |

---

## Dependencias

- `numpy`
- `matplotlib`
- `astropy`
- `lmfit`
- `scipy`
- `pandas`
