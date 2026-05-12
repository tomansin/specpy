# specpy

Visualizador interactivo de espectros estelares en formato FITS.

Incluye dos scripts:
- **`spec.py`** — espectros 1D estándar (REOSC, FEROS, HARPS, SOPHIE, HARPN, UVES, etc.)
- **`multispec.py`** — espectros en formato IRAF MULTISPEC (múltiples órdenes)

---

## spec.py

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

### Visualizador principal

| Tecla | Acción |
|-------|--------|
| `w` | Definir ventana de recorte (arrastrar) |
| `Enter` | Aplicar recorte |
| `z` | Volver al espectro completo |
| `n` | Modo normalización |
| `d` | Modo ajuste de gaussianas |
| `x` | Guardar espectro actual como FITS |
| `h` | Imprimir header FITS en terminal |
| `q` | Cerrar |
| — | — |
| `p` | Pan |
| `o` | Zoom rectangular |
| scroll | Zoom in/out |
| `home` | Reset vista |

---

### Modo normalización (`n`)

Ajuste de continuo por rangos con interpolación Akima entre rangos múltiples.

| Tecla | Acción |
|-------|--------|
| `a` | Activar/desactivar SpanSelector de regiones |
| `b` | Sellar rango activo e iniciar uno nuevo |
| `e` | Eliminar última región (elimina el rango si queda vacío) |
| `+` / `-` | Subir/bajar orden del polinomio Chebyshev del rango activo |
| `q` | Confirmar y cerrar (pregunta si guardar) |

Con un solo rango se ajusta un polinomio global. Con varios rangos, los polinomios se unen con interpolación Akima. Las zonas fuera de todos los rangos no se normalizan.

---

### Modo ajuste de gaussianas (`d`)

Ajuste de líneas espectrales con modelo gaussiano + **fondo lineal** (`bkg_intercept + bkg_slope × λ`), ambos parámetros libres en el ajuste.

**Flujo de trabajo recomendado:**

1. **`w`** — arrastrar para definir la región de continuo **izquierda** de la línea (sombra verde).
2. **`w`** — arrastrar para definir la región de continuo **derecha** de la línea. Al tener las 2 regiones se ajusta una recta al continuo, que se muestra superpuesta al espectro. El rango de ajuste queda definido automáticamente como el intervalo entre ambas regiones.
3. **`d`** — 2 clics para definir cada gaussiana:
   - Clic 1: centro de la línea (x = λ, y = flujo).
   - Clic 2: cualquier punto a un lado del centro para definir el FWHM (simétrico).
4. **`a`** — ejecuta el ajuste lmfit. Las gaussianas y el fondo lineal se ajustan simultáneamente sobre el rango definido.

| Tecla | Acción |
|-------|--------|
| `w` | Definir región de continuo (2 drags: izquierda y derecha de la línea) |
| `W` | Limpiar regiones de continuo |
| `d` | Nueva gaussiana (2 clics: centro + un lado del FWHM) |
| `a` | Ejecutar ajuste lmfit |
| `b` | Eliminar última gaussiana definida |
| `c` | Limpiar todas las gaussianas y el ajuste |
| `e` | Activar modo borrado de puntos individuales |
| `b` (en modo borrado) | Restaurar todos los puntos eliminados |
| `escape` | Cancelar gaussiana en construcción |
| `q` | Cerrar |

Al ajustar, muestra en terminal el reporte lmfit con velocidades radiales (vr) y anchos equivalentes (EW) para cada componente, identificando la línea de reposo más cercana desde `lines.csv`.

---

## multispec.py

```bash
multispec.py <archivo.fits>
```

Lee archivos FITS con `CTYPE1 = 'MULTISPE'` (formato IRAF multispec). Soporta cualquier instrumento que use ese estándar.

### Visualizador MULTISPEC

| Tecla | Acción |
|-------|--------|
| `(` | Orden anterior |
| `)` | Orden siguiente |
| `n` | Normalizar orden activo (hereda regiones del orden anterior) |
| `x` | Guardar FITS con todos los órdenes (normalizados donde corresponda) |
| `w` | Definir ventana de zoom |
| `Enter` | Aplicar ventana |
| `z` | Reset vista |
| `d` | Modo ajuste de gaussianas |
| `h` | Imprimir header |
| `q` | Cerrar |

### Normalización por órdenes

Al presionar `n` se abre el modo normalización para el orden activo. Dentro de ese modo:

| Tecla | Acción |
|-------|--------|
| `(` / `)` | Cambiar al orden anterior/siguiente (guarda el normalizado actual automáticamente y hereda las regiones al nuevo orden) |
| `q` | Confirmar y cerrar |

Al presionar `x` o `q` en el visualizador principal, se ofrece guardar un archivo `<nombre>_norm.fits` con los órdenes normalizados reemplazados y los demás intactos. El título muestra `[O N*/total]` donde `*` indica que el orden está normalizado y el número junto a `N` indica cuántos órdenes se normalizaron.

---

## Archivos de salida

| Archivo | Descripción |
|---------|-------------|
| `<nombre>_norm.fits` | Espectro 1D normalizado |
| `<nombre>_crop.fits` | Espectro 1D recortado |
| `<nombre>_crop_norm.fits` | Espectro 1D recortado y normalizado |
| `<nombre>_norm.fits` | MULTISPEC con órdenes normalizados reemplazados |
| `fitted_<linea>.csv` | Parámetros del ajuste: centro, σ, FWHM, EW, vr, bkg\_intercept, bkg\_slope |

---

## Dependencias

- `numpy`
- `matplotlib`
- `astropy`
- `lmfit`
- `scipy`
- `pandas`
- `specutils`
