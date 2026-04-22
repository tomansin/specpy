#!/home/tansin/miniconda3/envs/spec/bin/python
# -*- coding: utf-8 -*-
"""
spec.py - Visualizador interactivo de espectros estelares en formato FITS.

Permite inspeccionar, recortar, normalizar, ajustar gaussianas y guardar
espectros de forma interactiva desde una interfaz grafica matplotlib.

Uso:
    spec.py <archivo.fits> [--window WMIN WMAX]
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from specpy.utils import (read_fits_simple, fit_cont_sigma, mask_generator,
                          gaussian, fit_lines, find_closest_line, vr, vrerr)

# Claves de cabecera FITS aceptadas como tiempo heliocentrizo/baricentrico.
# Se prueban en orden; se usa la primera que se encuentre.
HJD_KEYS = ['HJD', 'JD', 'MJD', 'MJD-OBS', 'OHP DRS BJD', 'I-HJD', 'MJDATE']


def load_spectrum(filename):
    """
    Carga un espectro estelar desde un archivo FITS.

    Utiliza read_fits_simple para obtener cabecera, longitudes de onda y flujo.
    Valida que la cabecera contenga alguna clave de tiempo (HJD_KEYS).

    Parameters
    ----------
    filename : str
        Ruta al archivo FITS.

    Returns
    -------
    header : astropy.io.fits.Header o None
    wavelength : np.ndarray o None
    flux : np.ndarray o None
        Devuelve (None, None, None) si hay error de lectura o falta la clave HJD.
    """
    try:
        header, wavelength, flux = read_fits_simple(filename)
        print(f"LOADED SPECTRUM FROM {filename}")
        print(f"  Wavelength range: {wavelength[0]:.2f} - {wavelength[-1]:.2f}")
        print(f"  Number of points: {len(wavelength)}")

        # Buscar clave de tiempo HJD/JD en la cabecera
        hjd_value = None
        hjd_key_used = None
        for key in HJD_KEYS:
            if key in header:
                hjd_value = header[key]
                hjd_key_used = key
                break

        # if hjd_value is None:
        #     print("\nERROR: No HJD or similar keyword found in header")
        #     print("   Keywords buscadas:", HJD_KEYS)
        #     print("   Keywords disponibles:")
        #     for i, key in enumerate(header.keys()):
        #         if i >= 20:
        #             # break
        #         print(f"     {key}: {header[key]}")
        #     return None, None, None
        if hjd_value is None:
            print("\nWARNING: No HJD or similar keyword found in header")
            return header, wavelength, flux
        
        if hjd_value:
            print(f"  HJD ({hjd_key_used}): {hjd_value}")
            return header, wavelength, flux

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None, None, None
    except Exception as e:
        print(f"Error loading spectrum: {e}")
        return None, None, None


def calculate_smart_ylimits(data, central_fraction=0.8, margin_factor=0.1):
    """
    Calcula limites Y robustos para graficar espectros con outliers.

    En lugar de usar min/max (sensibles a picos cosmicosy artefactos),
    usa los percentiles 1 y 99 de la fraccion central del array,
    y agrega un margen proporcional al rango.

    Parameters
    ----------
    data : np.ndarray
        Array de flujo.
    central_fraction : float
        Fraccion del array a considerar como region central (default 0.8).
    margin_factor : float
        Factor del rango a agregar como margen (default 0.1).

    Returns
    -------
    ymin, ymax : float
    """
    if len(data) == 0:
        return 0, 1

    n = len(data)
    lo = max(0, int((1 - central_fraction) / 2 * n))
    hi = min(n, int((1 + central_fraction) / 2 * n))
    central = data[lo:hi]

    if len(central) == 0:
        return np.min(data), np.max(data)

    ymin = np.percentile(central, 1)
    ymax = np.percentile(central, 99)
    margin = (ymax - ymin) * margin_factor
    ymin = max(ymin - margin, 0) if np.min(data) >= 0 else ymin - margin
    ymax = ymax + margin
    return ymin, ymax


def interactive_normalization(wavelength, flux, filename):
    """
    Interfaz grafica para normalizar un espectro por ajuste de continuo.

    Soporta uno o varios rangos. Cada rango contiene regiones seleccionadas
    y tiene su propio polinomio Chebyshev. Con un solo rango el comportamiento
    es identico al modo clasico (polinomio global). Con varios rangos los
    polinomios se unen mediante interpolacion Akima. Las zonas del espectro
    fuera de todo rango no se normalizan (continuo = 1.0).

    Teclas:
      a         activar/desactivar seleccion de regiones (SpanSelector)
      b         sellar rango activo e iniciar uno nuevo (requiere >= 1 region)
      e         eliminar ultima region del rango activo;
                si queda vacio, elimina el rango
      +/-       subir/bajar orden del polinomio del rango activo
      q         confirmar y cerrar

    Parameters
    ----------
    wavelength : np.ndarray
    flux : np.ndarray
    filename : str

    Returns
    -------
    norm_flux : np.ndarray o None
    """
    from scipy.interpolate import Akima1DInterpolator

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(5, 1)
    ax_spec = fig.add_subplot(gs[:3, 0])
    ax_norm = fig.add_subplot(gs[3:, 0], sharex=ax_spec)

    # Cada rango: dict con 'regions', 'poly_order', 'fit_model', 'reject',
    #             'poly_line', 'reject_scatter', 'region_patches'
    ranges = []
    current_poly_order = [5]   # orden heredado por nuevos rangos
    span_selector = [None]
    selection_active = [False]
    continuum_line = [None]    # linea roja del continuo final en ax_spec

    COLORS = [plt.cm.tab10(i) for i in range(10)]

    # ── Setup ─────────────────────────────────────────────────────────────────
    ax_spec.plot(wavelength, flux, 'k-', linewidth=1.5, label='Espectro')
    ax_spec.set_xlabel('Wavelength (A)', fontsize=12)
    ax_spec.set_ylabel('Flux', fontsize=12)
    ax_spec.set_title(f'Normalizacion: {os.path.basename(filename)}',
                      fontsize=14, fontweight='bold')
    ax_spec.set_ylim(*calculate_smart_ylimits(flux))
    ax_spec.set_xlim()
    ax_spec.grid(True, alpha=0.3, linestyle='--')
    ax_spec.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax_spec.legend()

    ax_norm.set_xlabel('Wavelength (A)', fontsize=12)
    ax_norm.set_ylabel('Flujo normalizado', fontsize=12)
    ax_norm.set_title('Previsualizacion normalizada', fontsize=14, fontweight='bold')
    ax_norm.grid(True, alpha=0.3, linestyle='--')
    ax_norm.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Referencia (1.0)')
    ax_norm.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def range_span(r):
        """Devuelve (wmin, wmax) = envelope de todas las regiones del rango."""
        return (min(reg[0] for reg in r['regions']),
                max(reg[1] for reg in r['regions']))

    def refit_range(r, ridx):
        """Reajusta el polinomio del rango r y redibuja sus artists."""
        if r['poly_line'] is not None:
            r['poly_line'].remove()
            r['poly_line'] = None
        if r['reject_scatter'] is not None:
            r['reject_scatter'].remove()
            r['reject_scatter'] = None

        if not r['regions']:
            r['fit_model'] = None
            return

        mask = mask_generator(wavelength, r['regions'])
        n_pts = int(np.sum(mask))
        if n_pts <= r['poly_order'] + 1:
            print(f"  Aviso: {n_pts} puntos insuficientes para orden "
                  f"{r['poly_order']} en R{ridx + 1}")
            r['fit_model'] = None
            return

        try:
            cont_model, reject, _ = fit_cont_sigma(
                wavelength[mask], flux[mask],
                model='chebyshev', order=r['poly_order'],
                use_sigma_clip=True, sigma_lower=3, sigma_upper=3
            )
            r['fit_model'] = cont_model
            r['reject'] = reject
        except Exception as e:
            print(f"  Error ajuste R{ridx + 1}: {e}")
            r['fit_model'] = None
            return

        # color = COLORS[ridx % len(COLORS)]
        wmin, wmax = range_span(r)
        x_plot = np.linspace(wmin, wmax, 500)
        is_active = (ridx == len(ranges) - 1)
        label = f'R{ridx + 1} ord{r["poly_order"]} (pol activo)' if is_active else '_nolegend_'
        r['poly_line'], = ax_spec.plot(
            x_plot, cont_model(x_plot),
            color='red', linewidth=2, linestyle='--', alpha=0.85,
            label=label
        )
        if reject is not None and len(reject[0]) > 0:
            r['reject_scatter'] = ax_spec.scatter(
                reject[0], reject[1],
                c='green', s=12, alpha=0.5, marker='x', zorder=5
            )

    def build_continuum():
        """
        Construye el array de continuo.
        1 rango valido  -> polinomio evaluado en todo el espectro (modo clasico).
        >1 rangos validos -> Akima dentro del span combinado; 1.0 fuera.
        """
        valid = [r for r in ranges if r.get('fit_model') is not None]
        if not valid:
            return None

        if len(valid) == 1:
            return valid[0]['fit_model'](wavelength)

        all_x, all_y = [], []
        for r in valid:
            wmin, wmax = range_span(r)
            n_pts = max(80, int((wmax - wmin)
                                / (wavelength[-1] - wavelength[0]) * 800))
            x_r = np.linspace(wmin, wmax, n_pts)
            all_x.append(x_r)
            all_y.append(r['fit_model'](x_r))

        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        order = np.argsort(all_x)
        all_x, all_y = all_x[order], all_y[order]
        _, uidx = np.unique(all_x, return_index=True)
        all_x, all_y = all_x[uidx], all_y[uidx]

        if len(all_x) < 4:
            return None

        try:
            interp = Akima1DInterpolator(all_x, all_y)
            continuum = np.ones(len(wavelength))
            in_span = (wavelength >= all_x[0]) & (wavelength <= all_x[-1])
            continuum[in_span] = interp(wavelength[in_span])
            return continuum
        except Exception as e:
            print(f"  Error Akima: {e}")
            return None

    def update_display():
        """Recalcula el continuo y actualiza ambos paneles."""
        if continuum_line[0] is not None:
            continuum_line[0].remove()
            continuum_line[0] = None

        continuum = build_continuum()

        if continuum is not None:
            valid = [r for r in ranges if r.get('fit_model') is not None]
            wmin_all = min(range_span(r)[0] for r in valid)
            wmax_all = max(range_span(r)[1] for r in valid)
            draw_mask = (wavelength >= wmin_all) & (wavelength <= wmax_all)
            # label = ('Continuo' if len(valid) == 1 else 'Continuo (Akima)')
            continuum_line[0], = ax_spec.plot(
                wavelength[draw_mask], continuum[draw_mask],
                'r-', linewidth=2, alpha=0.9, #label=label
            )
        ax_spec.legend(loc='best')

        ax_norm.clear()
        ax_norm.set_xlabel('Wavelength (A)', fontsize=12)
        ax_norm.set_ylabel('Flujo normalizado', fontsize=12)
        ax_norm.set_title('Previsualizacion normalizada', fontsize=14, fontweight='bold')
        ax_norm.grid(True, alpha=0.3, linestyle='--')
        ax_norm.axhline(y=1, color='red', linestyle='--', alpha=0.5,
                        label='Referencia (1.0)')
        ax_norm.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        if continuum is not None:
            norm_flux = flux / continuum
            ax_norm.plot(wavelength, norm_flux, 'k-', linewidth=1.5,
                         label='Normalizado')
            # ylim calculado solo sobre los puntos dentro de rangos
            range_mask = np.zeros(len(wavelength), dtype=bool)
            for r in ranges:
                if r['regions']:
                    wmin, wmax = range_span(r)
                    range_mask |= (wavelength >= wmin) & (wavelength <= wmax)
            if np.any(range_mask):
                ax_norm.set_ylim(*calculate_smart_ylimits(
                    norm_flux[range_mask], margin_factor=0.5))
        ax_norm.legend(loc='best')

        fig.canvas.draw_idle()

    # ── Activacion / sellado de rangos ───────────────────────────────────────

    def seal_range(r):
        """Convierte el rango activo en inactivo: quita patches, pone '+' azules."""
        if r.get('poly_line') is not None:
            r['poly_line'].set_label('_nolegend_')

        for patch in r['region_patches']:
            try:
                patch.remove()
            except Exception:
                pass
        r['region_patches'].clear()

        if r.get('used_points_handle') is not None:
            try:
                r['used_points_handle'].remove()
            except Exception:
                pass

        if r['regions']:
            mask = mask_generator(wavelength, r['regions'])
            r['used_points_handle'], = ax_spec.plot(
                wavelength[mask], flux[mask],
                'b+', markersize=5, alpha=0.6, zorder=3, label='_nolegend_'
            )
        else:
            r['used_points_handle'] = None

    def activate_range(r, ridx):
        """Convierte un rango sellado en activo: quita '+', recrea patches rojos."""
        if r.get('used_points_handle') is not None:
            try:
                r['used_points_handle'].remove()
            except Exception:
                pass
            r['used_points_handle'] = None

        for region in r['regions']:
            patch = ax_spec.axvspan(region[0], region[1],
                                    alpha=0.15, color='red', zorder=0)
            r['region_patches'].append(patch)

    # ── SpanSelector ──────────────────────────────────────────────────────────

    def onselect(xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin

        # Crear primer rango automaticamente si no hay ninguno
        if not ranges:
            ranges.append({
                'regions': [], 'poly_order': current_poly_order[0],
                'fit_model': None, 'reject': None,
                'poly_line': None, 'reject_scatter': None,
                'region_patches': [], 'used_points_handle': None,
            })
            print(f"  Rango 1 iniciado")

        r = ranges[-1]
        ridx = len(ranges) - 1
        r['regions'].append([xmin, xmax])

        # Patch rojo para el rango activo
        patch = ax_spec.axvspan(xmin, xmax, alpha=0.15, color='red', zorder=0)
        r['region_patches'].append(patch)

        print(f"  R{ridx + 1}, region {len(r['regions'])}: "
              f"[{xmin:.2f}, {xmax:.2f}] A")
        refit_range(r, ridx)
        update_display()

    def toggle_selection():
        selection_active[0] = not selection_active[0]
        if selection_active[0]:
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            span_selector[0] = SpanSelector(
                ax_spec, onselect, 'horizontal',
                useblit=True,
                props=dict(alpha=0.2, facecolor='red'),
                interactive=True,
                drag_from_anywhere=True
            )
            print("  Seleccion ACTIVADA")
        else:
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
                span_selector[0] = None
            print("  Seleccion DESACTIVADA")

    # ── Teclado ───────────────────────────────────────────────────────────────

    def on_key(event):
        if event.key == 'a':
            toggle_selection()

        elif event.key == 'b':
            if not ranges or not ranges[-1]['regions']:
                print("  Define al menos una region antes de crear un nuevo rango.")
                return
            seal_range(ranges[-1])
            current_poly_order[0] = 5
            ridx_new = len(ranges)
            ranges.append({
                'regions': [], 'poly_order': current_poly_order[0],
                'fit_model': None, 'reject': None,
                'poly_line': None, 'reject_scatter': None,
                'region_patches': [], 'used_points_handle': None,
            })
            print(f"  Rango {ridx_new} sellado. "
                  f"Rango {ridx_new + 1} activo.")
            fig.canvas.draw_idle()

        elif event.key == 'e':
            if not ranges:
                return
            r = ranges[-1]
            ridx = len(ranges) - 1

            if not r['regions']:
                # Rango vacio (creado con 'b' pero sin regiones aun)
                ranges.pop()
                print(f"  Rango {ridx + 1} (vacio) eliminado.")
                if ranges:
                    activate_range(ranges[-1], len(ranges) - 1)
                update_display()
                return

            patch = r['region_patches'].pop()
            patch.remove()
            removed = r['regions'].pop()
            print(f"  Region [{removed[0]:.2f}, {removed[1]:.2f}] A eliminada. "
                  f"Quedan {len(r['regions'])} en R{ridx + 1}.")

            if r['regions']:
                refit_range(r, ridx)
            else:
                if r['poly_line'] is not None:
                    r['poly_line'].remove()
                    r['poly_line'] = None
                if r['reject_scatter'] is not None:
                    r['reject_scatter'].remove()
                    r['reject_scatter'] = None
                ranges.pop()
                print(f"  Rango {ridx + 1} eliminado. "
                      f"Quedan {len(ranges)} rango(s).")
                if ranges:
                    activate_range(ranges[-1], len(ranges) - 1)
            update_display()

        elif event.key in ('+', '='):
            if ranges:
                r = ranges[-1]
                r['poly_order'] = min(r['poly_order'] + 1, 20)
                current_poly_order[0] = r['poly_order']
                print(f"  Orden R{len(ranges)}: {r['poly_order']}")
                refit_range(r, len(ranges) - 1)
                update_display()

        elif event.key == '-':
            if ranges:
                r = ranges[-1]
                r['poly_order'] = max(r['poly_order'] - 1, 0)
                current_poly_order[0] = r['poly_order']
                print(f"  Orden R{len(ranges)}: {r['poly_order']}")
                refit_range(r, len(ranges) - 1)
                update_display()

        elif event.key == 'q':
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            n_valid = sum(1 for r in ranges if r['regions'])
            print(f"\n  Finalizando normalizacion ({n_valid} rango(s))")
            plt.close(fig)

    def on_scroll_norm(event):
        if event.inaxes not in (ax_spec, ax_norm):
            return
        factor = 0.85 if event.button == 'up' else 1.0 / 0.85
        xmin, xmax = ax_spec.get_xlim()
        xc = event.xdata
        ax_spec.set_xlim(xc + (xmin - xc) * factor, xc + (xmax - xc) * factor)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll_norm)

    print("\n" + "="*60)
    print("MODO NORMALIZACION")
    print("="*60)
    print("  a         activar/desactivar seleccion de regiones")
    print("  b         sellar rango activo e iniciar nuevo rango")
    print("  e         eliminar ultima region (elimina rango si queda vacio)")
    print("  +/-       subir/bajar orden del polinomio del rango activo")
    print("  q         confirmar y cerrar")
    print("="*60)

    plt.tight_layout()
    plt.show()

    # ── Calculo final ─────────────────────────────────────────────────────────
    valid_ranges = [r for r in ranges if r['regions']]
    if not valid_ranges:
        return None

    resp = input("  Guardar normalizacion? [S/n]: ").strip().lower()
    if resp in ('n', 'no'):
        print("  Normalizacion descartada.")
        return None

    # Reajuste limpio de todos los rangos
    for ridx, r in enumerate(valid_ranges):
        mask = mask_generator(wavelength, r['regions'])
        if np.sum(mask) > r['poly_order'] + 1:
            try:
                cont_model, _, _ = fit_cont_sigma(
                    wavelength[mask], flux[mask],
                    model='chebyshev', order=r['poly_order'],
                    use_sigma_clip=True
                )
                r['fit_model'] = cont_model
            except Exception as e:
                print(f"  Error reajuste final R{ridx + 1}: {e}")
                r['fit_model'] = None
        else:
            r['fit_model'] = None

    continuum = build_continuum()
    if continuum is not None:
        return flux / continuum

    return None


def interactive_gaussian_fitting(wavelength, flux, filename, params_dict=None):
    """
    Interfaz grafica para ajustar gaussianas a lineas espectrales.

    Flujo de trabajo:
      1. El usuario presiona 'g' para iniciar la definicion de una gaussiana.
      2. Se hacen 3 clics: centro (x=lambda, y=profundidad), FWHM izquierdo,
         FWHM derecho. La gaussiana se dibuja como linea punteada.
      3. El usuario repite para todas las lineas que quiera ajustar.
      4. Presionando 'a' se ejecuta el ajuste automatico con lmfit y se
         muestra la curva resultante junto al reporte completo en consola.
      5. 'e' activa el modo de borrado de puntos ruidosos antes del ajuste.
      6. 'q' cierra la figura.

    Si se provee params_dict (JSON), los parametros fijos del JSON se respetan
    y los libres se inicializan con los valores definidos manualmente. Si no
    hay gaussianas manuales, se usa el JSON directamente.

    El espectro pasado como argumento se usa solo para ajuste; la funcion
    no modifica los datos del visualizador principal.

    Parameters
    ----------
    wavelength : np.ndarray
        Longitudes de onda del espectro a ajustar.
    flux : np.ndarray
        Flujo del espectro a ajustar.
    filename : str
        Nombre del archivo (solo para el titulo de la figura).
    params_dict : dict o None
        Parametros de ajuste cargados desde un archivo JSON. Formato lmfit:
        claves g1_center, g1_sigma, g1_amplitude, ..., bkg_c; cada valor es
        un dict con 'value', y opcionalmente 'vary', 'min', 'max'.

    Returns
    -------
    result : lmfit.ModelResult o None
        Resultado del ajuste si se ejecuto, None si se cancelo.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # --- Estado mutable del modo de ajuste ---
    # Gaussianas definidas manualmente: lista de (center, amplitude, fwhm)
    gaussians = []
    # Pasos de la gaussiana en construccion: [center, depth] o [center, depth, left_wl]
    current_gaussian = None
    step = None          # 'center', 'fwhm_left', 'fwhm_right'
    result = None        # lmfit.ModelResult del ultimo ajuste exitoso

    # Handles graficos para limpiar sin cla()
    gaussian_lines = []      # lineas y marcadores de gaussianas manuales
    fitted_lines = []        # curva del ajuste final
    gaussian_patches = []    # artists temporales durante la definicion
    json_preview_lines = []  # curvas de prevista cargadas del JSON al inicio

    # Modo de borrado de puntos
    erase_mode = False
    removed_indices = []     # indices en el array original de puntos eliminados

    # Copias de trabajo; se actualizan al borrar puntos
    original_wavelength = wavelength.copy()
    original_flux = flux.copy()
    current_wavelength = wavelength.copy()
    current_flux = flux.copy()

    # Configuracion inicial del eje
    spectrum_line, = ax.plot(current_wavelength, current_flux, 'k-',
                             linewidth=1.5, label='Spectrum')
    ax.set_xlabel('Wavelength (A)', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title(f'Gaussian Fit: {os.path.basename(filename)}',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(*calculate_smart_ylimits(current_flux))
    ax.set_xlim(current_wavelength[0], current_wavelength[-1])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Referencia (1.0)')

    # Texto de estado superpuesto en la figura (actualizado por update_status)
    status_text = ax.text(0.02, 0.98, '',
                          transform=ax.transAxes, fontsize=9,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def update_status():
        """Actualiza el texto de estado superpuesto en la figura."""
        if erase_mode:
            msg = "MODO BORRADO - Click en un punto para eliminarlo (r: restaurar todos)"
        elif step is None:
            msg = "Listo. d: nueva gaussiana"
            if gaussians:
                msg += "  |  a: ajustar"
        elif step == 'center':
            msg = "Paso 1/3: click en el CENTRO de la linea (x=lambda, y=profundidad)"
        elif step == 'fwhm_left':
            msg = "Paso 2/3: click a la IZQUIERDA del centro para el FWHM"
        elif step == 'fwhm_right':
            msg = "Paso 3/3: click a la DERECHA del centro para el FWHM"

        # Resumen de gaussianas definidas
        info = f"\nGaussianas: {len(gaussians)}"
        for i, (c, a, fwhm) in enumerate(gaussians, 1):
            depth = 1 + a
            info += f"\n  G{i}: lambda={c:.2f} A  profundidad={depth:.3f}  FWHM={fwhm:.2f} A"

        if result is not None:
            info += f"\nAjuste: chi2_reducido={result.redchi:.3e}"

        if removed_indices:
            info += f"\nPuntos eliminados: {len(removed_indices)}"

        status_text.set_text(msg + info)
        fig.canvas.draw_idle()

    def draw_gaussian_line(center, amplitude, fwhm, color='blue', alpha=0.8,
                           linewidth=3, linestyle='-', label=None):
        """Dibuja una gaussiana sobre el espectro actual y devuelve el handle."""
        y = gaussian(current_wavelength, center, amplitude, fwhm)
        line, = ax.plot(current_wavelength, y, color=color, linewidth=linewidth,
                        linestyle=linestyle, alpha=alpha, label=label)
        return line

    def clear_current_gaussian():
        """Descarta la gaussiana en construccion y limpia sus artists temporales."""
        nonlocal current_gaussian, step
        current_gaussian = None
        step = None
        for item in gaussian_patches:
            try:
                item.remove()
            except Exception:
                pass
        gaussian_patches.clear()
        update_status()

    def finalize_gaussian():
        """
        Completa la gaussiana en construccion y la agrega a la lista.
        Se llama tras el tercer clic (FWHM derecho).
        """
        nonlocal current_gaussian
        if not current_gaussian or len(current_gaussian) < 4:
            return

        center, depth, left_wl, right_wl = current_gaussian
        fwhm_val = right_wl - left_wl
        amplitude = -(1 - depth)  # negativo para lineas de absorcion

        clear_current_gaussian()

        color = plt.cm.tab10(len(gaussians) % 10)
        line = draw_gaussian_line(center, amplitude, fwhm_val,
                                  color=color, alpha=0.9, linestyle='--',
                                  label=f'G{len(gaussians)+1}: {center:.2f} A')
        gaussian_lines.append(line)

        # Marcador de centro
        idx = np.argmin(np.abs(current_wavelength - center))
        center_marker = ax.plot(center, current_flux[idx], 'r|',
                                markersize=10, alpha=0.7)[0]
        gaussian_lines.append(center_marker)

        # Marcadores de FWHM
        half_max = amplitude / 2
        flux_at_center = current_flux[idx]
        left_marker = ax.plot(center - fwhm_val / 2, flux_at_center + half_max,
                              'b<', markersize=8, alpha=0.7)[0]
        right_marker = ax.plot(center + fwhm_val / 2, flux_at_center + half_max,
                               'b>', markersize=8, alpha=0.7)[0]
        fwhm_line = ax.plot([center - fwhm_val / 2, center + fwhm_val / 2],
                            [flux_at_center + half_max, flux_at_center + half_max],
                            'b:', alpha=0.5, linewidth=1)[0]
        gaussian_lines.extend([left_marker, right_marker, fwhm_line])

        gaussians.append((center, amplitude, fwhm_val))
        ax.legend(loc='best')

        print(f"  G{len(gaussians)}: lambda={center:.2f} A  profundidad={depth:.3f}  FWHM={fwhm_val:.2f} A")
        update_status()

    def build_manual_params():
        """
        Construye el dict de parametros lmfit a partir de las gaussianas
        definidas manualmente, aplicando una tolerancia del 50% como limites.
        """
        tol = 0.5
        fit_params = {}
        for i, (center, amplitude, fwhm_val) in enumerate(gaussians, 1):
            prefix = f'g{i}_'
            sigma = fwhm_val / 2.3548200
            amp = amplitude * sigma * np.sqrt(2 * np.pi)
            fit_params[f'{prefix}center'] = {'value': center,
                                             'min': center * (1 - tol),
                                             'max': center * (1 + tol)}
            fit_params[f'{prefix}sigma'] = {'value': sigma,
                                            'min': sigma * (1 - tol),
                                            'max': sigma * (1 + tol)}
            # amplitude es negativo para absorcion: min/max se invierten
            fit_params[f'{prefix}amplitude'] = {
                'value': amp,
                'min': amp * (1 + tol) if amp < 0 else amp * (1 - tol),
                'max': amp * (1 - tol) if amp < 0 else amp * (1 + tol),
            }
        fit_params['bkg_c'] = {'value': 1.0, 'vary': False}
        return fit_params

    def combine_params():
        """
        Combina los parametros del JSON con las gaussianas manuales.

        Logica de prioridad:
          - Sin gaussianas manuales: usa el JSON tal cual.
          - Con gaussianas manuales y sin JSON: usa las manuales con tolerancia 20%.
          - Ambos con igual cantidad: para cada parametro libre en el JSON usa el
            valor manual; para los fijos conserva el valor del JSON.
          - Cantidades distintas: pregunta al usuario como proceder.

        Returns el dict de parametros listo para fit_lines, o None si se cancela.
        """
        n_manual = len(gaussians)

        # Contar gaussianas en el JSON
        json_gaussians = {}
        if params_dict:
            for key, value in params_dict.items():
                if key.startswith('g') and '_' in key:
                    parts = key.split('_', 1)
                    try:
                        g_num = int(parts[0][1:])
                        param_type = parts[1]
                        if g_num not in json_gaussians:
                            json_gaussians[g_num] = {}
                        json_gaussians[g_num][param_type] = value
                    except ValueError:
                        continue
        n_json = len(json_gaussians)

        # Sin JSON: solo manuales
        if not params_dict:
            if n_manual == 0:
                print("  Primero define al menos una gaussiana con 'd'.")
                return None
            return build_manual_params()

        # Con JSON pero sin manuales: usar JSON directamente
        if n_manual == 0:
            print(f"  Usando parametros del JSON ({n_json} gaussiana(s)).")
            return params_dict.copy()

        # Cantidades distintas: preguntar
        if n_manual != n_json:
            print(f"\n  Aviso: el JSON tiene {n_json} gaussiana(s) y se definieron "
                  f"{n_manual} manualmente.")
            print("  Opciones:")
            print("    1  usar solo el JSON")
            print("    2  usar solo las gaussianas manuales")
            print("    3  cancelar")
            choice = input("  Eleccion (1-3): ").strip()
            if choice == '1':
                print("  Usando parametros del JSON.")
                return params_dict.copy()
            elif choice == '2':
                print("  Usando gaussianas manuales.")
                return build_manual_params()
            else:
                print("  Ajuste cancelado.")
                return None

        # Misma cantidad: combinar (manual inicializa los parametros libres del JSON)
        print(f"  Combinando JSON + {n_manual} gaussiana(s) manuales.")
        combined = {}
        for i in range(1, n_manual + 1):
            if i not in json_gaussians or i - 1 >= len(gaussians):
                continue
            prefix = f'g{i}_'
            jg = json_gaussians[i]
            manual_center, manual_amplitude, manual_fwhm = gaussians[i - 1]
            manual_sigma = manual_fwhm / 2.3548200
            manual_amp = manual_amplitude * manual_sigma * np.sqrt(2 * np.pi)

            for param, manual_val in [('center', manual_center),
                                       ('sigma', manual_sigma),
                                       ('amplitude', manual_amp)]:
                if param in jg and not jg[param].get('vary', True):
                    # Parametro fijo en el JSON: respetar el valor del JSON
                    combined[f'{prefix}{param}'] = jg[param]
                elif param in jg:
                    # Parametro libre: valor manual como estimacion inicial,
                    # pero se respetan min, max y vary del JSON
                    entry = dict(jg[param])
                    entry['value'] = manual_val
                    combined[f'{prefix}{param}'] = entry
                else:
                    combined[f'{prefix}{param}'] = {'value': manual_val}

        combined['bkg_c'] = params_dict.get('bkg_c', {'value': 1.0, 'vary': False})
        return combined

    def do_fit():
        """Determina los parametros a usar y ejecuta el ajuste lmfit."""
        nonlocal result

        fit_params = combine_params()
        if fit_params is None:
            return

        try:
            result = fit_lines(current_wavelength, current_flux, fit_params)

            # Limpiar curva de ajuste anterior y vista previa del JSON
            for line in fitted_lines + json_preview_lines:
                line.remove()
            fitted_lines.clear()
            json_preview_lines.clear()

            fitted_line, = ax.plot(current_wavelength, result.best_fit,
                                   color='red', linewidth=4, alpha=0.6,
                                   label='Ajuste total')
            fitted_lines.append(fitted_line)

            # Graficar componentes individuales: bkg_c + g{i}(x)
            components = result.eval_components(x=current_wavelength)
            bkg = components.get('bkg_', 0.0)
            g_keys = sorted(k for k in components if k.startswith('g'))
            for idx, key in enumerate(g_keys):
                color = plt.cm.tab10(idx % 10)
                g_num = key.rstrip('_')  # 'g1_' -> 'g1'
                center_val = result.params.get(f'{key}center')
                center_str = f'{center_val.value:.2f} A' if center_val else g_num
                comp_line, = ax.plot(current_wavelength, bkg + components[key],
                                     color=color, linewidth=2, linestyle=':',
                                     alpha=0.9, label=f'{g_num}: {center_str}')
                fitted_lines.append(comp_line)

            ax.legend(loc='best')

            print("\n" + "="*60)
            print("REPORTE DEL AJUSTE")
            print("="*60)
            print(result.fit_report())
            print("="*60)

            update_status()

        except Exception as e:
            print(f"  Error en el ajuste: {e}")

    def remove_point_at(x, y):
        """
        Elimina el punto del espectro mas cercano a las coordenadas (x, y).
        Usa tolerancias relativas al rango visible para evitar clics accidentales.
        """
        nonlocal current_wavelength, current_flux, erase_mode

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_tol = (xlim[1] - xlim[0]) * 0.01
        y_tol = (ylim[1] - ylim[0]) * 0.05

        dist_x = np.abs(current_wavelength - x)
        dist_y = np.abs(current_flux - y)
        in_tol = (dist_x < x_tol) & (dist_y < y_tol)

        if not np.any(in_tol):
            print(f"  Ningun punto cerca de lambda={x:.2f} A, flux={y:.3f}")
            return

        # Indice en current_wavelength del punto mas cercano dentro de tolerancia
        score = dist_x / x_tol + dist_y / y_tol
        score[~in_tol] = np.inf
        cur_idx = np.argmin(score)

        # Mapear al indice original (considerando los ya eliminados)
        visible_mask = np.ones(len(original_wavelength), dtype=bool)
        visible_mask[removed_indices] = False
        orig_idx = np.where(visible_mask)[0][cur_idx]

        removed_indices.append(orig_idx)
        removed_indices.sort()

        visible_mask[orig_idx] = False
        current_wavelength = original_wavelength[visible_mask]
        current_flux = original_flux[visible_mask]

        spectrum_line.set_data(current_wavelength, current_flux)

        # Redibujar gaussianas con el nuevo array de longitudes de onda
        for line in gaussian_lines + fitted_lines:
            try:
                line.remove()
            except Exception:
                pass
        gaussian_lines.clear()
        fitted_lines.clear()

        for i, (center, amplitude, fwhm_val) in enumerate(gaussians):
            color = plt.cm.tab10(i % 10)
            line = draw_gaussian_line(center, amplitude, fwhm_val,
                                      color=color, alpha=0.9, linestyle='--',
                                      label=f'G{i+1}: {center:.2f} A')
            gaussian_lines.append(line)

        ax.set_ylim(*calculate_smart_ylimits(current_flux))
        erase_mode = False
        print(f"  Punto eliminado: lambda={original_wavelength[orig_idx]:.2f} A"
              f"  flux={original_flux[orig_idx]:.4f}  (total: {len(removed_indices)})")

        fig.canvas.draw_idle()
        update_status()

    def restore_all_points():
        """Restaura todos los puntos eliminados al array de trabajo."""
        nonlocal current_wavelength, current_flux
        if not removed_indices:
            print("  No hay puntos eliminados.")
            return
        removed_indices.clear()
        current_wavelength = original_wavelength.copy()
        current_flux = original_flux.copy()
        spectrum_line.set_data(current_wavelength, current_flux)
        ax.set_ylim(*calculate_smart_ylimits(current_flux))
        print(f"  Restaurados {len(current_wavelength)} puntos.")
        fig.canvas.draw_idle()
        update_status()

    def on_click(event):
        nonlocal step, current_gaussian, erase_mode
        if event.inaxes != ax:
            return
        x, y = event.xdata, event.ydata

        if erase_mode:
            remove_point_at(x, y)
            return

        if step is None:
            return

        ymin, ymax = ax.get_ylim()

        if step == 'center':
            # Restringir depth al rango [0, 1.2]
            depth = np.clip(y, 0.0, 1.2)
            vline = ax.vlines(x, ymin, ymax, color='blue', linestyle='--', alpha=0.5)
            label = ax.text(x, ymax * 0.95, f'{x:.2f} A',
                            color='blue', ha='center', fontsize=9)
            gaussian_patches.extend([vline, label])
            current_gaussian = [x, depth]
            step = 'fwhm_left'
            print(f"  Centro: {x:.2f} A  profundidad: {depth:.3f}")

        elif step == 'fwhm_left':
            if x >= current_gaussian[0]:
                print("  Click a la IZQUIERDA del centro.")
                return
            vline = ax.vlines(x, ymin, ymax, color='orange', linestyle='--', alpha=0.5)
            gaussian_patches.append(vline)
            current_gaussian.append(x)
            step = 'fwhm_right'
            print(f"  FWHM izquierdo: {x:.2f} A")

        elif step == 'fwhm_right':
            if x <= current_gaussian[0]:
                print("  Click a la DERECHA del centro.")
                return
            vline = ax.vlines(x, ymin, ymax, color='orange', linestyle='--', alpha=0.5)
            gaussian_patches.append(vline)
            current_gaussian.append(x)
            print(f"  FWHM derecho: {x:.2f} A")
            finalize_gaussian()

        update_status()

    def on_key(event):
        nonlocal step, current_gaussian, result, erase_mode

        if event.key == 'q':
            plt.close(fig)
            return

        # Dentro del modo de borrado solo se aceptan 'e'/'escape' para salir y 'r' para restaurar
        if erase_mode:
            if event.key in ('e', 'escape'):
                erase_mode = False
                print("  Modo borrado desactivado.")
                update_status()
            elif event.key == 'b':
                restore_all_points()
            return

        if event.key == 'e':
            erase_mode = True
            print("  Modo borrado activado. Click en un punto para eliminarlo.")
            update_status()

        elif event.key == 'd':
            if step is None:
                step = 'center'
                current_gaussian = []
                update_status()

        elif event.key == 'a':
            do_fit()

        elif event.key == 'b':
            # Eliminar la ultima gaussiana definida
            if gaussians:
                gaussians.pop()
                # Cada gaussiana agrega hasta 5 handles (linea + centro + izq + der + fwhm_line)
                n_remove = min(5, len(gaussian_lines))
                for _ in range(n_remove):
                    if gaussian_lines:
                        gaussian_lines.pop().remove()
                # Invalidar el ajuste anterior
                for line in fitted_lines:
                    line.remove()
                fitted_lines.clear()
                result = None
                ax.legend(loc='best')
                update_status()

        elif event.key == 'c':
            for line in gaussian_lines + fitted_lines:
                try:
                    line.remove()
                except Exception:
                    pass
            gaussians.clear()
            gaussian_lines.clear()
            fitted_lines.clear()
            result = None
            ax.legend(loc='best')
            update_status()

        elif event.key == 'escape':
            if step is not None:
                clear_current_gaussian()

    def on_scroll_gauss(event):
        if event.inaxes != ax:
            return
        factor = 0.85 if event.button == 'up' else 1.0 / 0.85
        xmin, xmax = ax.get_xlim()
        xc = event.xdata
        ax.set_xlim(xc + (xmin - xc) * factor, xc + (xmax - xc) * factor)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll_gauss)

    # Si se cargo un JSON, dibujar las gaussianas como vista previa al inicio
    if params_dict:
        i = 1
        while f'g{i}_center' in params_dict:
            try:
                center = params_dict[f'g{i}_center']['value']
                sigma  = params_dict[f'g{i}_sigma']['value']
                amp    = params_dict[f'g{i}_amplitude']['value']
                height = amp / (sigma * np.sqrt(2 * np.pi))
                fwhm   = sigma * 2.3548200
                color  = plt.cm.tab10((i - 1) % 10)
                y = gaussian(current_wavelength, center, height, fwhm)
                line, = ax.plot(current_wavelength, y, color=color,
                                linewidth=1.5, linestyle='--', alpha=0.7,
                                label=f'JSON G{i}: {center:.2f} A')
                json_preview_lines.append(line)
                print(f"  JSON G{i}: lambda={center:.2f} A  sigma={sigma:.3f}  amp={amp:.4f}")
            except (KeyError, TypeError) as e:
                print(f"  Aviso: no se pudo dibujar JSON G{i}: {e}")
            i += 1
        if json_preview_lines:
            ax.legend(loc='best')
            print(f"  {len(json_preview_lines)} gaussiana(s) del JSON graficadas (---).")

    update_status()

    print("\n" + "="*60)
    print("MODO AJUSTE DE GAUSSIANAS")
    print("="*60)
    print("  d         nueva gaussiana (3 clics: centro, FWHM izq, FWHM der)")
    print("  a         ajustar automaticamente todas las gaussianas")
    print("  b         eliminar ultima gaussiana")
    print("  c         limpiar todas las gaussianas")
    print("  e         activar/desactivar modo borrado de puntos")
    print("  escape    cancelar gaussiana en construccion")
    print("  q         cerrar")
    if params_dict:
        n_json = sum(1 for k in params_dict if k.startswith('g') and '_center' in k)
        print(f"\n  JSON cargado: {n_json} gaussiana(s). Presiona 'a' para ajustar "
              f"con JSON (o define gaussianas manuales primero).")
    print("="*60)

    plt.tight_layout()
    plt.show()

    return result


VR_WARNING_THRESHOLD = 500.0  # km/s


def _print_vr_summary(result, vhelio):
    """
    Imprime un resumen de velocidades radiales y anchos equivalentes
    para cada gaussiana ajustada.

    EW = -amplitude  (analitico: integral de la gaussiana sobre el continuo=1).
    Convencion: EW > 0 absorcion, EW < 0 emision.

    Parameters
    ----------
    result : lmfit.ModelResult
        Resultado del ajuste de gaussianas.
    vhelio : float
        Correccion de velocidad heliocentrica en km/s.

    Returns
    -------
    bool
        True si alguna velocidad radial supera VR_WARNING_THRESHOLD en valor absoluto.
    """
    print("\n" + "="*60)
    print("VELOCIDADES RADIALES  &  ANCHOS EQUIVALENTES")
    print("="*60)

    high_vr = False
    i = 1
    while f'g{i}_center' in result.params:
        p_center = result.params[f'g{i}_center']
        lambda_obs = p_center.value
        lambda_err = p_center.stderr if p_center.stderr is not None else float('nan')

        # Ancho equivalente: EW = -amplitude
        p_amp  = result.params.get(f'g{i}_amplitude')
        ew_val = -p_amp.value if p_amp is not None else float('nan')
        ew_err = p_amp.stderr if (p_amp is not None and p_amp.stderr is not None) else float('nan')
        ew_str = f"{ew_val:.4f}"
        if not np.isnan(ew_err):
            ew_str += f" +/- {ew_err:.4f}"
        ew_str += " A"

        try:
            line_info = find_closest_line(lambda_obs)
            lambda0   = line_info['lambda_rest']
            line_name = line_info.get('name', '')

            vr_val = vr(lambda_obs, lambda0, vhelio)
            vr_err = vrerr(lambda_err, lambda0) if not np.isnan(lambda_err) else float('nan')

            vr_str = f"{vr_val:.2f}"
            if not np.isnan(vr_err):
                vr_str += f" +/- {vr_err:.2f}"
            vr_str += " km/s"

            print(f"  G{i}: {line_name}  lambda0={lambda0:.3f} A"
                  f"  lambda={lambda_obs:.4f} A")
            print(f"       vr={vr_str}  EW={ew_str}")

            if abs(vr_val) > VR_WARNING_THRESHOLD:
                high_vr = True
        except Exception as e:
            print(f"  G{i}: lambda={lambda_obs:.4f} A  EW={ew_str}"
                  f"  (sin linea de referencia: {e})")

        i += 1

    print("="*60)
    return high_vr


def save_fit_to_csv(filename, linename, hjd_value, vhelio, result):
    """
    Guarda los resultados de un ajuste lmfit en un archivo CSV.

    Si el archivo ya existe, agrega la nueva fila. Si las columnas no coinciden
    pregunta al usuario como proceder. El nombre del archivo es
    'fitted_<linename>.csv' en el directorio de trabajo actual.

    Parameters
    ----------
    filename : str
        Ruta al espectro FITS (solo se guarda el basename).
    linename : str
        Nombre de la linea espectral; determina el nombre del CSV.
    hjd_value : float
        Tiempo heliocentrizo/baricentrico del header.
    vhelio : float
        Velocidad heliocentrica (km/s) del header; 0.0 si no esta disponible.
    result : lmfit.ModelResult
        Resultado del ajuste de gaussianas.
    """
    import pandas as pd
    import shutil

    csv_filename = f"fitted_{linename}.csv"

    # Informacion general
    data_dict = {
        'filename': [os.path.basename(filename)],
        'hjd':      [f"{hjd_value:.6f}"],
        'vhelio':   [f"{vhelio:.6f}"],
        'chi2_red': [f"{result.redchi:.4f}"],
        'success':  [result.success],
    }

    # Extraer parametros de cada gaussiana
    gaussian_params = {}
    for param_name in result.params:
        if param_name.startswith('g') and '_' in param_name:
            parts = param_name.split('_', 1)
            try:
                g_num = int(parts[0][1:])
                param_type = parts[1]
                if g_num not in gaussian_params:
                    gaussian_params[g_num] = {}
                p = result.params[param_name]
                gaussian_params[g_num][param_type] = {
                    'value': p.value,
                    'error': p.stderr if p.stderr is not None else np.nan,
                    'vary':  p.vary,
                }
            except ValueError:
                continue

    n_gauss = len(gaussian_params)
    data_dict['n_gauss'] = [n_gauss]

    for i in range(1, n_gauss + 1):
        if i not in gaussian_params:
            continue
        g = gaussian_params[i]

        center_val = g.get('center', {}).get('value', np.nan)
        center_err = g.get('center', {}).get('error', np.nan)
        data_dict[f'center{i}']      = [f"{center_val:.4f}"]
        data_dict[f'center{i}_err']  = [f"{center_err:.4f}" if not np.isnan(center_err) else ""]
        data_dict[f'center{i}_vary'] = [g.get('center', {}).get('vary', True)]

        sigma_val = g.get('sigma', {}).get('value', np.nan)
        sigma_err = g.get('sigma', {}).get('error', np.nan)
        data_dict[f'sigma{i}']      = [f"{sigma_val:.4f}"]
        data_dict[f'sigma{i}_err']  = [f"{sigma_err:.4f}" if not np.isnan(sigma_err) else ""]
        data_dict[f'sigma{i}_vary'] = [g.get('sigma', {}).get('vary', True)]

        fwhm_val = sigma_val * 2.3548200 if not np.isnan(sigma_val) else np.nan
        fwhm_err = sigma_err * 2.3548200 if not np.isnan(sigma_err) else np.nan
        data_dict[f'fwhm{i}']     = [f"{fwhm_val:.4f}"]
        data_dict[f'fwhm{i}_err'] = [f"{fwhm_err:.4f}" if not np.isnan(fwhm_err) else ""]

        amp_val = g.get('amplitude', {}).get('value', np.nan)
        amp_err = g.get('amplitude', {}).get('error', np.nan)
        data_dict[f'amp{i}']      = [f"{amp_val:.4f}"]
        data_dict[f'amp{i}_err']  = [f"{amp_err:.4f}" if not np.isnan(amp_err) else ""]
        data_dict[f'amp{i}_vary'] = [g.get('amplitude', {}).get('vary', True)]

        if not np.isnan(amp_val) and not np.isnan(sigma_val) and sigma_val != 0:
            height_val = amp_val / (sigma_val * np.sqrt(2 * np.pi))
            depth_val  = 1 + height_val
            data_dict[f'height{i}'] = [f"{height_val:.4f}"]
            data_dict[f'depth{i}']  = [f"{depth_val:.4f}"]
        else:
            data_dict[f'height{i}'] = [""]
            data_dict[f'depth{i}']  = [""]

        # Ancho equivalente: EW = -amplitude
        ew_val = -amp_val if not np.isnan(amp_val) else np.nan
        ew_err = amp_err  if not np.isnan(amp_err) else np.nan
        data_dict[f'ew{i}']     = [f"{ew_val:.4f}" if not np.isnan(ew_val) else ""]
        data_dict[f'ew{i}_err'] = [f"{ew_err:.4f}" if not np.isnan(ew_err) else ""]

        # Velocidad radial usando la linea en reposo mas cercana
        center_val = g.get('center', {}).get('value', np.nan)
        center_err = g.get('center', {}).get('error', np.nan)
        try:
            line_info  = find_closest_line(center_val)
            lambda0    = line_info['lambda_rest']
            vr_val     = vr(center_val, lambda0, vhelio)
            vr_err_val = vrerr(center_err, lambda0) if not np.isnan(center_err) else np.nan
            data_dict[f'line{i}_name']    = [line_info.get('name', '')]
            data_dict[f'line{i}_lambda0'] = [f"{lambda0:.4f}"]
            data_dict[f'vr{i}']           = [f"{vr_val:.4f}"]
            data_dict[f'vr{i}_err']       = [f"{vr_err_val:.4f}" if not np.isnan(vr_err_val) else ""]
        except Exception:
            data_dict[f'line{i}_name']    = [""]
            data_dict[f'line{i}_lambda0'] = [""]
            data_dict[f'vr{i}']           = [""]
            data_dict[f'vr{i}_err']       = [""]

    # Fondo (bkg_c)
    bkg_found = False
    for param_name in result.params:
        if param_name.startswith('bkg_'):
            bkg = result.params[param_name]
            data_dict['bkg_c']      = [f"{bkg.value:.4f}"]
            data_dict['bkg_c_err']  = [f"{bkg.stderr:.4f}" if bkg.stderr is not None else ""]
            data_dict['bkg_c_vary'] = [bkg.vary]
            bkg_found = True
            break
    if not bkg_found:
        data_dict['bkg_c'] = data_dict['bkg_c_err'] = data_dict['bkg_c_vary'] = [""]

    # Orden de columnas
    base_cols = ['filename', 'hjd', 'vhelio', 'chi2_red', 'success', 'n_gauss']
    gauss_cols = []
    for i in range(1, n_gauss + 1):
        gauss_cols.extend([
            f'center{i}', f'center{i}_err', f'center{i}_vary',
            f'sigma{i}',  f'sigma{i}_err',  f'sigma{i}_vary',
            f'fwhm{i}',   f'fwhm{i}_err',
            f'amp{i}',    f'amp{i}_err',    f'amp{i}_vary',
            f'height{i}', f'depth{i}',
            f'ew{i}', f'ew{i}_err',
            f'line{i}_name', f'line{i}_lambda0', f'vr{i}', f'vr{i}_err',
        ])
    all_cols = base_cols + gauss_cols + ['bkg_c', 'bkg_c_err', 'bkg_c_vary']
    for col in all_cols:
        if col not in data_dict:
            data_dict[col] = [""]

    df_new = pd.DataFrame({col: data_dict[col] for col in all_cols})

    if os.path.exists(csv_filename):
        try:
            df_existing = pd.read_csv(csv_filename)
            existing_cols = set(df_existing.columns)
            new_cols = set(df_new.columns)

            if existing_cols != new_cols:
                print(f"\n  Aviso: las columnas del CSV existente no coinciden con las nuevas.")
                print(f"  Columnas existentes: {len(existing_cols)}")
                print(f"  Columnas nuevas:     {len(new_cols)}")
                print("  Opciones:")
                print("    1  agregar de todos modos (puede generar columnas vacias)")
                print("    2  crear archivo nuevo (backup del existente)")
                print("    3  cancelar")
                choice = input("  Eleccion (1-3): ").strip()

                if choice == '1':
                    merged_cols = sorted(existing_cols.union(new_cols))
                    for col in merged_cols:
                        if col not in df_existing.columns:
                            df_existing[col] = ""
                        if col not in df_new.columns:
                            df_new[col] = ""
                    df_existing = df_existing[merged_cols]
                    df_new = df_new[merged_cols]
                    pd.concat([df_existing, df_new], ignore_index=True).to_csv(csv_filename, index=False)
                    print(f"  Agregado a {csv_filename} (columnas ajustadas).")

                elif choice == '2':
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    backup = f"{csv_filename}.backup_{timestamp}"
                    shutil.copy2(csv_filename, backup)
                    print(f"  Backup creado: {backup}")
                    df_new.to_csv(csv_filename, index=False)
                    print(f"  Nuevo archivo: {csv_filename}")

                else:
                    print("  Guardado cancelado.")
                    return
            else:
                pd.concat([df_existing, df_new], ignore_index=True).to_csv(csv_filename, index=False)
                print(f"  Fila agregada a {csv_filename}")

        except Exception as e:
            print(f"  Error leyendo CSV existente: {e}. Creando nuevo archivo.")
            df_new.to_csv(csv_filename, index=False)
            print(f"  Creado: {csv_filename}")
    else:
        df_new.to_csv(csv_filename, index=False)
        print(f"  Creado: {csv_filename}")

    print(f"\n  Resumen guardado:")
    print(f"    Archivo:    {os.path.basename(filename)}")
    print(f"    HJD:        {hjd_value:.6f}")
    print(f"    Gaussianas: {n_gauss}")
    print(f"    chi2/nu:    {result.redchi:.4e}")
    print(f"    CSV:        {os.path.abspath(csv_filename)}")


def save_spectrum_fits(out_path, header, wavelength, flux):
    """
    Guarda un espectro 1D en formato FITS preservando el WCS del original.

    Actualiza las claves WCS (CRVAL1, CDELT1, CRPIX1, NAXIS1) para que
    correspondan al array guardado. Maneja el caso de escala log-lineal
    (DC-FLAG = 1), en el que CRVAL1 y CDELT1 se guardan en log10.

    Elimina claves que no aplican a un espectro 1D (NAXIS2, NAXIS3, WCSDIM).

    Parameters
    ----------
    out_path : str
        Ruta de salida del archivo FITS.
    header : astropy.io.fits.Header
        Cabecera original del espectro (se copia y modifica, no se altera).
    wavelength : np.ndarray
        Longitudes de onda del espectro a guardar.
    flux : np.ndarray
        Flujo del espectro a guardar.
    """
    from astropy.io import fits as astropy_fits

    new_header = header.copy()

    # Eliminar claves que no corresponden a un espectro 1D
    for key in ['NAXIS2', 'NAXIS3', 'WCSDIM']:
        if key in new_header:
            del new_header[key]

    new_header['NAXIS'] = 1
    new_header['NAXIS1'] = len(wavelength)
    new_header['CRPIX1'] = 1  # el primer pixel corresponde a CRVAL1

    if 'DC-FLAG' in header and header['DC-FLAG'] == 1:
        # Escala log-lineal: las claves WCS se guardan en log10(lambda)
        new_header['CRVAL1'] = np.log10(wavelength[0])
        new_header['CDELT1'] = np.mean(np.diff(np.log10(wavelength)))
        new_header['DC-FLAG'] = 1
    else:
        # Escala lineal: las claves WCS se guardan en angstrom directamente
        new_header['CRVAL1'] = wavelength[0]
        new_header['CDELT1'] = np.mean(np.diff(wavelength))
        if 'DC-FLAG' in new_header:
            del new_header['DC-FLAG']

    hdu = astropy_fits.PrimaryHDU(flux.astype(np.float32), header=new_header)
    hdu.writeto(out_path, overwrite=True)


def plot_spectrum(wavelength, flux, filename, header, params_dict=None, is_windowed=False,
                  start_mode=None):
    """
    Visualizador interactivo principal del espectro.

    Implementa un loop de sesiones para manejar el ciclo de vida de la figura:
    cuando el usuario presiona 'n', la figura se cierra, se abre la UI de
    normalizacion, y al terminar se reabre el visualizador con el espectro
    normalizado. El estado (recorte y normalizacion) persiste entre sesiones.

    Teclas disponibles:
      q     cerrar
      w     activar modo ventana (click y drag para definir rango)
      Enter aplicar ventana de recorte
      o     volver al espectro completo original
      n     abrir modo de normalizacion
      g     abrir modo de ajuste de gaussianas
      x     guardar espectro actual como FITS (_crop y/o _norm segun estado)

    Parameters
    ----------
    wavelength : np.ndarray
        Longitudes de onda originales (sin recorte).
    flux : np.ndarray
        Flujo original (sin recorte ni normalizacion).
    filename : str
        Ruta al archivo FITS original (usada para el titulo y el nombre de salida).
    header : astropy.io.fits.Header
        Cabecera FITS original (usada al guardar con 'x' y al guardar el ajuste en CSV).
    params_dict : dict o None
        Parametros de ajuste de gaussianas cargados desde JSON (ver --params).
    is_windowed : bool
        True si el espectro ya fue recortado antes de llamar esta funcion (ej. --window).
    start_mode : str o None
        Si es 'normalize' o 'fit_gaussians', entra directamente en ese modo
        sin mostrar primero la visualizacion principal, y al terminar no la reabre.
    """
    if wavelength is None or flux is None or len(wavelength) == 0:
        print("Error: No data to plot")
        return

    # Extraer HJD y velocidad heliocentrica del header para el guardado de ajustes
    hjd_value = None
    for key in HJD_KEYS:
        if key in header:
            hjd_value = header[key]
            break
    vhelio = float(header['VHELIO']) if 'VHELIO' in header else 0.0

    # Estado persistente entre sesiones.
    # Se usan listas de un elemento para permitir mutacion desde closures anidados.
    current = [wavelength, flux]  # espectro actualmente visible (puede ser recortado/normalizado)
    is_normalized = [False]       # True si current[1] es el resultado de una normalizacion
    is_windowed = [is_windowed]   # True si el espectro esta recortado (incluye --window inicial)

    def make_title():
        """Genera el titulo de la figura reflejando el estado actual."""
        norm_tag   = ' (norm)' if is_normalized[0] else ''
        window_tag = ' (crop)' if is_windowed[0] else ''
        if is_windowed[0]:
            return (f'Spectrum{window_tag}{norm_tag}: {os.path.basename(filename)}'
                    f' [{current[0][0]:.1f} - {current[0][-1]:.1f} A]')
        return f'Spectrum{norm_tag}: {os.path.basename(filename)}'

    def save_current(prompt=False):
        """
        Guarda el espectro actual (current) como FITS.

        El nombre de salida se construye agregando sufijos al nombre original:
          _crop  si el espectro esta recortado respecto al original
          _norm  si el espectro esta normalizado

        Parameters
        ----------
        prompt : bool
            Si True, pregunta al usuario antes de guardar. Si no hay nada
            que guardar, retorna silenciosamente sin preguntar.
        """
        base, ext = os.path.splitext(os.path.basename(filename))
        suffix = ('_crop' if is_windowed[0] else '') + ('_norm' if is_normalized[0] else '')
        if not suffix:
            if not prompt:
                print("  Nada que guardar: el espectro no esta cortado ni normalizado")
            return
        out_path = base + suffix + ext
        if prompt:
            respuesta = input(f"\n  Guardar espectro como {os.path.basename(out_path)}? [S/n]: ").strip().lower()
            if respuesta in ('n', 'no'):
                return
        try:
            save_spectrum_fits(out_path, header, current[0], current[1])
            print(f"  Guardado: {out_path}")
        except Exception as e:
            print(f"  Error al guardar: {e}")

    def run_session():
        """
        Abre la figura matplotlib y bloquea hasta que el usuario la cierra.

        Devuelve 'normalize' si el usuario pidio normalizar (tecla 'n'),
        o None en cualquier otro caso de cierre.
        """
        pending = [None]  # almacena la accion solicitada al cerrar
        wl, fl = current[0], current[1]

        fig, ax = plt.subplots(figsize=(12, 6))

        def setup_ax(wl, fl, title):
            """Dibuja el espectro y configura ejes, grilla y limites iniciales."""
            ax.plot(wl, fl, 'k-', linewidth=1.5, label='Spectrum')
            ax.set_xlabel('Wavelength (A)', fontsize=12)
            ax.set_ylabel('Flux', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax.set_ylim(*calculate_smart_ylimits(fl))
            ax.set_xlim(wl[0], wl[-1])
            ax.legend(loc='best')

        def refresh_toolbar():
            """
            Reinicia el stack de navegacion de la barra de herramientas.
            Necesario despues de cambiar los datos de la linea con set_data(),
            para que el boton 'home' refleje el nuevo rango en lugar del original.
            """
            if hasattr(fig.canvas, 'toolbar') and fig.canvas.toolbar is not None:
                fig.canvas.toolbar.update()

        setup_ax(wl, fl, make_title())
        fig.tight_layout()

        window_limits = [None, None]  # [xmin, xmax] de la ventana pendiente de aplicar
        span_selector = [None]        # widget SpanSelector activo para definir ventana

        def onselect_window(xmin, xmax):
            """Callback del SpanSelector: registra los limites de la ventana seleccionada."""
            if xmin > xmax:
                xmin, xmax = xmax, xmin
            window_limits[0] = xmin
            window_limits[1] = xmax
            print(f"  Ventana seleccionada: [{xmin:.2f}, {xmax:.2f}] A")
            print("  Presiona Enter para aplicar, o 'w' para redefinir")

        def activate_window_mode():
            """
            Activa el SpanSelector para que el usuario defina una ventana de recorte.
            Desactiva cualquier selector previo para evitar handlers duplicados.
            """
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            window_limits[0] = None
            window_limits[1] = None
            span_selector[0] = SpanSelector(
                ax, onselect_window, 'horizontal',
                useblit=True,
                props=dict(alpha=0.3, facecolor='blue'),
                interactive=True,
                drag_from_anywhere=True
            )
            print("\n  Modo ventana activado: click y arrastra para definir el rango")
            print("  Presiona Enter para aplicar, 'w' para redefinir")

        def apply_window():
            """
            Aplica el recorte definido por window_limits al espectro original.

            Siempre recorta desde el espectro original (wavelength/flux), no desde
            current, para evitar que recortes sucesivos acumulen errores de mascara.
            Si el espectro estaba normalizado, la normalizacion se descarta porque
            el continuo ajustado ya no es valido para el nuevo rango.
            """
            xmin, xmax = window_limits
            if xmin is None or xmax is None:
                print("  No hay ventana definida. Usa 'w' para definirla.")
                return

            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
                span_selector[0] = None

            # Mascara sobre el espectro original, no sobre current
            mask = (wavelength >= xmin) & (wavelength <= xmax)
            wl_window = wavelength[mask]
            fl_window = flux[mask]

            if len(wl_window) == 0:
                print("  La ventana seleccionada no contiene datos.")
                return

            if is_normalized[0]:
                # El continuo normalizado ya no es valido para el nuevo rango
                print("  Aviso: re-ventanear descarta la normalizacion")
                is_normalized[0] = False

            current[0] = wl_window
            current[1] = fl_window

            # Actualizar la linea en lugar de limpiar el eje, para preservar el resto del plot
            ax.lines[0].set_data(wl_window, fl_window)
            ax.set_title(make_title(), fontsize=14, fontweight='bold')
            ax.set_ylim(*calculate_smart_ylimits(fl_window))
            ax.set_xlim(xmin, xmax)
            is_windowed[0] = True
            refresh_toolbar()
            fig.canvas.draw_idle()
            print(f"\n  Recorte aplicado: [{xmin:.2f}, {xmax:.2f}] A  ({len(wl_window)} puntos)")

        def reset_view():
            """
            Restaura el espectro completo original, descartando recorte y normalizacion.
            """
            is_normalized[0] = False
            is_windowed[0] = False
            current[0] = wavelength
            current[1] = flux
            ax.lines[0].set_data(wavelength, flux)
            ax.set_title(make_title(), fontsize=14, fontweight='bold')
            ax.set_ylim(*calculate_smart_ylimits(flux))
            ax.set_xlim(wavelength[0], wavelength[-1])
            refresh_toolbar()
            fig.canvas.draw_idle()
            print("  Vista reseteada al espectro completo")

        def on_key(event):
            """Despacha los eventos de teclado a las funciones correspondientes."""
            if event.key.lower() == 'q':
                pending[0] = 'quit'
                plt.close(fig)
            elif event.key.lower() == 'w':
                activate_window_mode()
            elif event.key == 'enter':
                apply_window()
            elif event.key.lower() == 'z':
                reset_view()
            elif event.key.lower() == 'n':
                # Senalar que se debe abrir la normalizacion despues de cerrar
                pending[0] = 'normalize'
                plt.close(fig)
            elif event.key.lower() == 'd':
                # Senalar que se debe abrir el ajuste de gaussianas despues de cerrar
                pending[0] = 'fit_gaussians'
                plt.close(fig)
            elif event.key.lower() == 'x':
                save_current()

        def on_scroll(event):
            if event.inaxes != ax:
                return
            factor = 0.85 if event.button == 'up' else 1.0 / 0.85
            xmin, xmax = ax.get_xlim()
            xc = event.xdata
            ax.set_xlim(xc + (xmin - xc) * factor, xc + (xmax - xc) * factor)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        print("\n" + "="*50)
        print("INSTRUCTIONS:")
        print("  q         cerrar")
        print("  w         definir ventana de corte (click y drag)")
        print("  Enter     aplicar ventana")
        print("  z         volver al espectro completo")
        print("  n         modo normalizacion")
        print("  d         modo ajuste de gaussianas")
        print("  x         guardar espectro actual como FITS")
        print("  --- matplotlib ---")
        print("  p         modo pan (arrastrar para mover)")
        print("  o         modo zoom (arrastrar para seleccionar)")
        print("  scroll    zoom in/out")
        print("  home      reset vista")
        print("="*50)

        plt.show()

        # La figura ya cerro: ahora es seguro usar input()
        if pending[0] == 'quit':
            save_current(prompt=True)
            return None

        return pending[0]

    # Loop de sesiones: corre run_session() repetidamente mientras el usuario
    # solicite normalizar. El estado en 'current' e 'is_normalized' persiste
    # entre iteraciones sin necesidad de recursion.
    # Si start_mode esta definido, se salta la sesion principal y se entra
    # directamente al modo indicado, saliendo al terminar sin reabrir la viz.
    if start_mode in ('normalize', 'fit_gaussians'):
        action = start_mode
    else:
        action = run_session()

    while True:
        if action == 'normalize':
            norm_fl = interactive_normalization(current[0], current[1], filename)
            if norm_fl is not None:
                current[1] = norm_fl
                is_normalized[0] = True
                if start_mode is None:
                    print("  Espectro normalizado. Reabriendo visualizacion.")
            else:
                if start_mode is None:
                    print("  Normalizacion cancelada. Reabriendo visualizacion.")
            if start_mode is not None:
                save_current(prompt=True)
                break
        elif action == 'fit_gaussians':
            fit_result = interactive_gaussian_fitting(current[0], current[1], filename, params_dict)
            if fit_result is not None:
                high_vr = _print_vr_summary(fit_result, vhelio)
                if high_vr:
                    print(f"\n  Aviso: una o mas velocidades radiales superan "
                          f"{VR_WARNING_THRESHOLD:.0f} km/s.")
                    continuar = input("  Continuar con el guardado? [s/N]: ").strip().lower()
                    if continuar not in ('s', 'si', 'y', 'yes'):
                        if start_mode is None:
                            print("  Reabriendo visualizacion.")
                            action = run_session()
                            continue
                        else:
                            break
            if fit_result is not None and hjd_value is not None:
                # Usar el nombre de la linea identificada como default del CSV
                linename = 'line'
                if 'g1_center' in fit_result.params:
                    try:
                        line_info = find_closest_line(fit_result.params['g1_center'].value)
                        linename = line_info.get('name', linename)
                        if len(linename) > 4:
                            linename = linename[-4:]
                    except Exception:
                        pass
                save = input("\n  Guardar parametros del ajuste en CSV? [S/n]: ").strip().lower()
                if save not in ('n', 'no'):
                    linename = input(f"  Nombre de la linea [{linename}]: ").strip() or linename
                    save_fit_to_csv(filename, linename, hjd_value, vhelio, fit_result)
            elif fit_result is not None and hjd_value is None:
                print("  Aviso: no se puede guardar sin HJD en el header.")
            if start_mode is not None:
                break
            print("  Reabriendo visualizacion.")
        else:
            break
        action = run_session()


def main():
    parser = argparse.ArgumentParser(description='Interactive spectrum analyzer')
    parser.add_argument('filename', help='FITS spectrum file')
    parser.add_argument('--window', type=float, nargs=2, metavar=('WMIN', 'WMAX'),
                        help='Wavelength window to display: WMIN WMAX (A)')
    parser.add_argument('--params', type=str, metavar='FILE',
                        help='JSON file with Gaussian fit parameters')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--normalized', action='store_true',
                            help='Entrar directamente en modo normalizacion y salir al terminar')
    mode_group.add_argument('--gaussian', action='store_true',
                            help='Entrar directamente en modo ajuste de gaussianas y salir al terminar')

    args = parser.parse_args()

    is_windowed = False

    header, wavelength, flux = load_spectrum(args.filename)
    if wavelength is None or flux is None or header is None:
        sys.exit(1)

    if args.window:
        wmin, wmax = args.window
        mask = (wavelength >= wmin) & (wavelength <= wmax)
        wavelength, flux = wavelength[mask], flux[mask]
        print(f"  Window aplicada: [{wmin:.2f}, {wmax:.2f}] A")
        is_windowed = True

    params_dict = None
    if args.params:
        import json
        try:
            with open(args.params, 'r') as f:
                params_dict = json.load(f)
            n = sum(1 for k in params_dict if k.startswith('g') and '_center' in k)
            print(f"  Parametros cargados desde {args.params} ({n} gaussiana(s)).")
        except Exception as e:
            print(f"  Error al cargar {args.params}: {e}")
            sys.exit(1)

    start_mode = None
    if args.normalized:
        start_mode = 'normalize'
    elif args.gaussian:
        start_mode = 'fit_gaussians'

    plot_spectrum(wavelength, flux, args.filename, header, params_dict,
                 is_windowed=is_windowed, start_mode=start_mode)


if __name__ == "__main__":
    main()
