# specpy

Herramientas interactivas para visualización, normalización y análisis de espectros estelares en formato FITS.

Scripts incluidos:
- **`spec.py`** — espectros 1D (REOSC, FEROS, HARPS, SOPHIE, HERMES, UVES, etc.)
- **`multispec.py`** — espectros en formato IRAF MULTISPEC (múltiples órdenes)
- **`votable2fits.py`** — conversión de VOTable a FITS BinTable compatible con `spec.py`

---

## spec.py

```bash
spec.py <archivo.fits> [opciones]
```

Lee archivos FITS 1D de cualquier instrumento. Detecta automáticamente si los datos están en la extensión primaria (WCS) o en una extensión BinTable. Los archivos guardados por `spec.py` llevan el keyword `PROCSPEC = 'spec.py'` y son reconocidos directamente al re-abrirlos.

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

Ajuste de continuo por polinomio Chebyshev con soporte de múltiples rangos. Cuando hay más de un rango los polinomios se unen con interpolación Akima. El panel inferior muestra la previsualización normalizada centrada en flujo = 1 (±10 σ calculado sobre los puntos seleccionados). Al cambiar el orden del polinomio (`+`/`-`) el zoom se preserva.

| Tecla | Acción |
|-------|--------|
| `a` | Activar/desactivar SpanSelector de regiones |
| `b` | Sellar rango activo e iniciar uno nuevo |
| `e` | Eliminar última región (elimina el rango si queda vacío) |
| `+` / `-` | Subir/bajar orden del polinomio Chebyshev del rango activo |
| `q` | Confirmar y cerrar (pregunta si guardar) |

---

### Modo ajuste de gaussianas (`d`)

Ajuste de líneas espectrales con modelo suma de gaussianas + fondo constante (`bkg_c`).

**Flujo de trabajo:**

1. **`w`** — arrastrar para definir una región de continuo local. Definir dos regiones (una a cada lado de la línea) ajusta una recta al continuo y la muestra superpuesta.
2. **`d`** — iniciar una gaussiana; 2 clics:
   - Clic 1: centro de la línea (x = λ, y = flujo/profundidad).
   - Clic 2: cualquier punto a un lado del centro para definir el FWHM (simétrico).
3. Repetir paso 2 para cada componente adicional.
4. **`a`** — ejecuta el ajuste lmfit y muestra el reporte con velocidades radiales y anchos equivalentes.

| Tecla | Acción |
|-------|--------|
| `w` | Definir región de continuo local (hasta 2 por línea) |
| `W` | Limpiar regiones de continuo |
| `d` | Nueva gaussiana (2 clics: centro + semiancho) |
| `a` | Ejecutar ajuste lmfit |
| `b` | Eliminar última gaussiana definida |
| `c` | Limpiar todas las gaussianas y el ajuste |
| `e` | Activar modo borrado de puntos individuales |
| `b` (en modo borrado) | Restaurar todos los puntos eliminados |
| `escape` | Cancelar gaussiana en construcción |
| `q` | Cerrar |

El reporte en terminal incluye velocidad radial (vr) y ancho equivalente (EW) para cada componente, identificando la línea de reposo más cercana desde `lines.csv`. Si alguna vr supera 500 km/s se emite una advertencia antes de guardar.

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
| `(` / `)` | Cambiar al orden anterior/siguiente (guarda el normalizado actual y hereda regiones) |
| `q` | Confirmar y cerrar |

Al presionar `x` o `q` en el visualizador principal se ofrece guardar `<nombre>_norm.fits` con los órdenes normalizados reemplazados. El título muestra `[O N*/total]` donde `*` indica orden normalizado.

---

## votable2fits.py

Convierte espectros en formato VOTable (como los descargados del VO de Mercator/HERMES) a FITS BinTable compatible con `spec.py`.

```bash
# Un archivo, salida por defecto (<nombre>_bt.fits)
votable2fits.py espectro.fits

# Un archivo con salida específica
votable2fits.py espectro.fits -o salida.fits

# Múltiples archivos con comodín
votable2fits.py espectros/*.fits
```

El archivo de salida lleva `INSTRUME = 'VOT2FITS'` y los metadatos SSA mapeados a keywords estándar (`MJD-OBS`, `OBJECT`, `EXPTIME`, etc.).

---

## Archivos de salida

| Archivo | Descripción |
|---------|-------------|
| `<nombre>_norm.fits` | Espectro 1D normalizado |
| `<nombre>_crop.fits` | Espectro 1D recortado |
| `<nombre>_crop_norm.fits` | Espectro 1D recortado y normalizado |
| `<nombre>_norm.fits` | MULTISPEC con órdenes normalizados reemplazados |
| `<nombre>_bt.fits` | VOTable convertido a FITS BinTable |
| `fitted_<linea>.csv` | Parámetros del ajuste: centro, σ, FWHM, EW, vr por componente |

---

## Dependencias

- `numpy`
- `matplotlib`
- `astropy`
- `lmfit`
- `scipy`
- `pandas`
- `specutils`
