#!/home/tansin/.conda/envs/spec/bin/python
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
import json
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from scipy.interpolate import Akima1DInterpolator
from specpy.utils import (read_fits_simple, fit_cont_sigma,
                          mask_generator, gaussian, fit_lines,
                          find_closest_line, vr, vrerr,
                          calculate_smart_ylimits, save_spectrum_fits,
                          save_fit_to_csv)

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

    def update_display(preserve_view=False):
        """Recalcula el continuo y actualiza ambos paneles.

        preserve_view : si True, conserva xlim y ylim de ax_spec (para +/-).
                        El ylim de ax_norm siempre se recalcula desde los datos.
        """
        if preserve_view:
            saved_xlim      = ax_spec.get_xlim()
            saved_ylim_spec = ax_spec.get_ylim()

        if continuum_line[0] is not None:
            continuum_line[0].remove()
            continuum_line[0] = None

        continuum = build_continuum()

        if continuum is not None:
            valid = [r for r in ranges if r.get('fit_model') is not None]
            wmin_all = min(range_span(r)[0] for r in valid)
            wmax_all = max(range_span(r)[1] for r in valid)
            draw_mask = (wavelength >= wmin_all) & (wavelength <= wmax_all)
            continuum_line[0], = ax_spec.plot(
                wavelength[draw_mask], continuum[draw_mask],
                'r-', linewidth=2, alpha=0.9,
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

        new_ylim_norm = None
        if continuum is not None:
            norm_flux = flux / continuum
            ax_norm.plot(wavelength, norm_flux, 'k-', linewidth=1.5,
                         label='Normalizado')
            region_mask = np.zeros(len(wavelength), dtype=bool)
            for r in ranges:
                if r['regions']:
                    region_mask |= mask_generator(wavelength, r['regions'])
            if np.any(region_mask):
                s = np.std(norm_flux[region_mask])
                s = s if s > 0 else 0.05
                new_ylim_norm = (1 - 15 * s, 1 + 5 * s)
        ax_norm.legend(loc='best')

        if new_ylim_norm is not None:
            ax_norm.set_ylim(*new_ylim_norm)

        if preserve_view:
            ax_spec.set_xlim(saved_xlim)
            ax_spec.set_ylim(saved_ylim_spec)

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

    def activate_range(r, _ridx=None):
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
        update_display(preserve_view=True)

    def toggle_selection():
        selection_active[0] = not selection_active[0]
        if selection_active[0]:
            if span_selector[0] is not None:
                span_selector[0].set_visible(False)
                span_selector[0].disconnect_events()
                span_selector[0] = None
                fig.canvas.draw_idle()
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
                span_selector[0].set_visible(False)
                span_selector[0].disconnect_events()
                span_selector[0] = None
                fig.canvas.draw_idle()
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
                update_display(preserve_view=True)
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
            update_display(preserve_view=True)

        elif event.key in ('+', '='):
            if ranges:
                r = ranges[-1]
                r['poly_order'] = min(r['poly_order'] + 1, 20)
                current_poly_order[0] = r['poly_order']
                print(f"  Orden R{len(ranges)}: {r['poly_order']}")
                refit_range(r, len(ranges) - 1)
                update_display(preserve_view=True)

        elif event.key == '-':
            if ranges:
                r = ranges[-1]
                r['poly_order'] = max(r['poly_order'] - 1, 0)
                current_poly_order[0] = r['poly_order']
                print(f"  Orden R{len(ranges)}: {r['poly_order']}")
                refit_range(r, len(ranges) - 1)
                update_display(preserve_view=True)

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


def interactive_gaussian_fitting(wavelength, flux, filename, params_dict=None, vhelio=0.0):
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

    # Continuo local: dos regiones (izquierda y derecha de la línea)
    bkg_regions = []       # [[wmin1,wmax1], [wmin2,wmax2]] — max 2
    bkg_coeffs = [None]    # [slope, intercept] de np.polyfit
    bkg_patches = []       # axvspan verdes
    bkg_line_art = [None]  # linea verde del continuo ajustado
    bkg_span = [None]      # SpanSelector activo

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
            msg = "Paso 1/2: click en el CENTRO de la linea"
        elif step == 'fwhm':
            msg = "Paso 2/2: click a cualquier lado del centro para definir el FWHM (simetrico)"

        # Resumen de gaussianas definidas
        info = f"\nGaussianas: {len(gaussians)}"
        for i, (c, a, fwhm) in enumerate(gaussians, 1):
            depth = 1 + a
            info += f"\n  G{i}: lambda={c:.2f} A  profundidad={depth:.3f}  FWHM={fwhm:.2f} A"

        if bkg_regions:
            n = len(bkg_regions)
            info += f"\nContinuo: {n}/2 region(es)"
            if bkg_coeffs[0] is not None:
                info += f"  slope={bkg_coeffs[0][0]:.3g}"
        else:
            info += "\nContinuo: no definido  (w: definir)"

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
        if bkg_coeffs[0] is not None:
            y = y * np.polyval(bkg_coeffs[0], current_wavelength)
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
            # Escalar amplitud al espacio del flujo real usando el continuo en el centro
            if bkg_coeffs[0] is not None:
                cont_at_center = float(np.polyval(bkg_coeffs[0], center))
            else:
                cont_at_center = float(np.percentile(current_flux, 95))
            amp = amplitude * cont_at_center * sigma * np.sqrt(2 * np.pi)
            fit_params[f'{prefix}center'] = {'value': center,
                                             'min': center * (1 - tol),
                                             'max': center * (1 + tol)}
            fit_params[f'{prefix}sigma'] = {'value': sigma,
                                            'min': sigma * (1 - tol),
                                            'max': sigma * (1 + tol)}
            fit_params[f'{prefix}amplitude'] = {
                'value': amp,
                'min': amp * (1 + tol) if amp < 0 else amp * (1 - tol),
                'max': amp * (1 - tol) if amp < 0 else amp * (1 + tol),
            }
        # Fondo lineal: inicializar desde el ajuste del continuo si está disponible
        if bkg_coeffs[0] is not None:
            fit_params['bkg_slope']     = {'value': float(bkg_coeffs[0][0]), 'vary': True}
            fit_params['bkg_intercept'] = {'value': float(bkg_coeffs[0][1]), 'vary': True}
        else:
            flux_scale = float(np.percentile(current_flux, 95))
            fit_params['bkg_intercept'] = {'value': flux_scale, 'vary': True}
            fit_params['bkg_slope']     = {'value': 0.0, 'vary': True}
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

        # Fondo lineal: respetar JSON si tiene bkg_intercept/slope; si tiene bkg_c (legacy) usarlo como intercept
        if 'bkg_intercept' in params_dict or 'bkg_slope' in params_dict:
            combined['bkg_intercept'] = params_dict.get('bkg_intercept', {'value': 1.0, 'vary': True})
            combined['bkg_slope'] = params_dict.get('bkg_slope', {'value': 0.0, 'vary': True})
        elif 'bkg_c' in params_dict:
            combined['bkg_intercept'] = {'value': params_dict['bkg_c'].get('value', 1.0), 'vary': True}
            combined['bkg_slope'] = {'value': 0.0, 'vary': True}
        else:
            if bkg_coeffs[0] is not None:
                combined['bkg_slope']     = {'value': float(bkg_coeffs[0][0]), 'vary': True}
                combined['bkg_intercept'] = {'value': float(bkg_coeffs[0][1]), 'vary': True}
            else:
                flux_scale = float(np.percentile(current_flux, 95))
                combined['bkg_intercept'] = {'value': flux_scale, 'vary': True}
                combined['bkg_slope']     = {'value': 0.0, 'vary': True}
        return combined

    def do_fit():
        """Determina los parametros a usar y ejecuta el ajuste lmfit."""
        nonlocal result

        fit_params = combine_params()
        if fit_params is None:
            return

        try:
            if bkg_regions:
                wmin_fit = min(r[0] for r in bkg_regions)
                wmax_fit = max(r[1] for r in bkg_regions)
                rmask = (current_wavelength >= wmin_fit) & (current_wavelength <= wmax_fit)
                fit_wl = current_wavelength[rmask]
                fit_fl = current_flux[rmask]
            else:
                fit_wl = current_wavelength
                fit_fl = current_flux

            if len(fit_wl) < 5:
                print("  Error: el rango de ajuste tiene muy pocos puntos.")
                return

            result = fit_lines(fit_wl, fit_fl, fit_params)

            # Limpiar curva de ajuste anterior y vista previa del JSON
            for line in fitted_lines + json_preview_lines:
                line.remove()
            fitted_lines.clear()
            json_preview_lines.clear()

            fitted_line, = ax.plot(fit_wl, result.best_fit,
                                   color='red', linewidth=4, alpha=0.6,
                                   label='Ajuste total')
            fitted_lines.append(fitted_line)

            # Graficar componentes individuales: bkg + g{i}(x)
            components = result.eval_components(x=fit_wl)
            bkg = components.get('bkg_', 0.0)
            g_keys = sorted(k for k in components if k.startswith('g'))
            for idx, key in enumerate(g_keys):
                color = plt.cm.tab10(idx % 10)
                g_num = key.rstrip('_')
                center_val = result.params.get(f'{key}center')
                center_str = f'{center_val.value:.2f} A' if center_val else g_num
                comp_line, = ax.plot(fit_wl, bkg + components[key],
                                     color=color, linewidth=2, linestyle=':',
                                     alpha=0.9, label=f'{g_num}: {center_str}')
                fitted_lines.append(comp_line)

            ax.legend(loc='best')

            print("\n" + "="*60)
            print("REPORTE DEL AJUSTE")
            print("="*60)
            print(result.fit_report())
            print("="*60)

            if not result.success:
                print("\n" + "!"*60)
                print("  *** AJUSTE NO CONVERGIO (success=False) ***")
                # Detectar si algun centro esta pegado a sus limites
                i = 1
                center_at_bound = False
                while f'g{i}_center' in result.params:
                    p = result.params[f'g{i}_center']
                    at_min = p.min is not None and abs(p.value - p.min) < 1e-6 * (abs(p.min) + 1)
                    at_max = p.max is not None and abs(p.value - p.max) < 1e-6 * (abs(p.max) + 1)
                    if at_min or at_max:
                        bound = "minimo" if at_min else "maximo"
                        print(f"  *** G{i}: CENTRO EN EL LIMITE {bound.upper()} "
                              f"({p.value:.4f} A) -- RESULTADO NO CONFIABLE ***")
                        center_at_bound = True
                    i += 1
                if center_at_bound:
                    print("  *** El centro es el parametro mas critico: "
                          "los EW y VR calculados son INVALIDOS ***")
                print("!"*60)

            _print_vr_summary(result, vhelio)

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

    def _onselect_bkg(xmin, xmax):
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        if len(bkg_regions) >= 2:
            print("  Ya hay 2 regiones. Presiona W para limpiar y redefinir.")
            return
        bkg_regions.append([xmin, xmax])
        bkg_patches.append(ax.axvspan(xmin, xmax, alpha=0.2, color='green', zorder=0))
        remaining = 2 - len(bkg_regions)
        if remaining > 0:
            print(f"  Region {len(bkg_regions)}/2 [{xmin:.2f}, {xmax:.2f}] A. "
                  f"Arrastra para la {len(bkg_regions)+1}a region.")
        else:
            _fit_bkg_line()
            if bkg_span[0] is not None:
                bkg_span[0].disconnect_events()
                bkg_span[0] = None
        update_status()

    def _fit_bkg_line():
        mask = np.zeros(len(current_wavelength), dtype=bool)
        for wmin, wmax in bkg_regions:
            mask |= (current_wavelength >= wmin) & (current_wavelength <= wmax)
        if np.sum(mask) < 2:
            print("  Continuo: pocos puntos en las regiones.")
            return
        coeffs = np.polyfit(current_wavelength[mask], current_flux[mask], 1)
        bkg_coeffs[0] = coeffs
        if bkg_line_art[0] is not None:
            try:
                bkg_line_art[0].remove()
            except Exception:
                pass
        wmins = [r[0] for r in bkg_regions]
        wmaxs = [r[1] for r in bkg_regions]
        x_draw = np.linspace(min(wmins), max(wmaxs), 300)
        bkg_line_art[0], = ax.plot(x_draw, np.polyval(coeffs, x_draw),
                                    'g-', linewidth=2, alpha=0.85, label='Continuo')
        ax.legend(loc='best')
        fig.canvas.draw_idle()
        print(f"  Continuo ajustado: slope={coeffs[0]:.3g}  intercept={coeffs[1]:.3g}")

    def _clear_bkg():
        bkg_regions.clear()
        bkg_coeffs[0] = None
        if bkg_span[0] is not None:
            bkg_span[0].disconnect_events()
            bkg_span[0] = None
        for patch in bkg_patches:
            try:
                patch.remove()
            except Exception:
                pass
        bkg_patches.clear()
        if bkg_line_art[0] is not None:
            try:
                bkg_line_art[0].remove()
            except Exception:
                pass
            bkg_line_art[0] = None
        ax.legend(loc='best')
        fig.canvas.draw_idle()
        print("  Continuo eliminado")
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
            if bkg_coeffs[0] is not None:
                bkg_at_x = np.polyval(bkg_coeffs[0], x)
                depth = y / bkg_at_x if bkg_at_x != 0 else np.clip(y, 0.0, 1.2)
            else:
                depth = np.clip(y, 0.0, 1.2)
            vline = ax.vlines(x, ymin, ymax, color='blue', linestyle='--', alpha=0.5)
            label = ax.text(x, ymax * 0.95, f'{x:.2f} A',
                            color='blue', ha='center', fontsize=9)
            gaussian_patches.extend([vline, label])
            current_gaussian = [x, depth]
            step = 'fwhm'
            print(f"  Centro: {x:.2f} A  profundidad: {depth:.3f}")

        elif step == 'fwhm':
            center = current_gaussian[0]
            half_width = abs(x - center)
            if half_width == 0:
                print("  Click en un punto distinto al centro.")
                return
            left_x  = center - half_width
            right_x = center + half_width
            for fx in (left_x, right_x):
                vline = ax.vlines(fx, ymin, ymax, color='orange', linestyle='--', alpha=0.5)
                gaussian_patches.append(vline)
            current_gaussian.extend([left_x, right_x])
            print(f"  FWHM: {2*half_width:.2f} A")
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

        elif event.key == 'w':
            if bkg_span[0] is not None:
                bkg_span[0].disconnect_events()
            bkg_span[0] = SpanSelector(
                ax, _onselect_bkg, 'horizontal', useblit=True,
                props=dict(alpha=0.2, facecolor='green'),
                interactive=True, drag_from_anywhere=True)
            remaining = 2 - len(bkg_regions)
            print(f"  Arrastra para definir region de continuo ({remaining} restante(s))")

        elif event.key == 'W':
            _clear_bkg()

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
    print("  w         definir region de continuo (2 drags: izq y der de la linea)")
    print("  W         limpiar regiones de continuo")
    print("  d         nueva gaussiana (2 clics: centro, un lado del FWHM)")
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
            respuesta = input(f"\n  Guardar espectro como {os.path.basename(out_path)}? [S/n/nombre]: ").strip()
            if respuesta.lower() in ('n', 'no'):
                return
            if respuesta and respuesta.lower() not in ('s', 'si', 'y', 'yes'):
                out_path = respuesta
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
                span_selector[0].set_visible(False)
                span_selector[0].disconnect_events()
                span_selector[0] = None
                fig.canvas.draw_idle()
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

        def close_session(action):
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
                span_selector[0] = None
            pending[0] = action
            plt.close(fig)

        def on_key(event):
            """Despacha los eventos de teclado a las funciones correspondientes."""
            if event.key.lower() == 'q':
                close_session('quit')
            elif event.key.lower() == 'w':
                activate_window_mode()
            elif event.key == 'enter':
                apply_window()
            elif event.key.lower() == 'z':
                reset_view()
            elif event.key.lower() == 'n':
                close_session('normalize')
            elif event.key.lower() == 'd':
                close_session('fit_gaussians')
            elif event.key.lower() == 'x':
                save_current()
            elif event.key.lower() == 'h':
                print("\n" + "="*50)
                print("HEADER")
                print("="*50)
                print(repr(header))
                print("="*50)

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
        print("  h         imprimir header FITS en terminal")
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
            fit_result = interactive_gaussian_fitting(current[0], current[1], filename, params_dict, vhelio)
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
                default_csv = f"fitted_{linename}.csv"
                if not fit_result.success:
                    # Comprobar si algun centro esta en el limite
                    j = 1
                    center_bad = False
                    while f'g{j}_center' in fit_result.params:
                        p = fit_result.params[f'g{j}_center']
                        at_min = p.min is not None and abs(p.value - p.min) < 1e-6 * (abs(p.min) + 1)
                        at_max = p.max is not None and abs(p.value - p.max) < 1e-6 * (abs(p.max) + 1)
                        if at_min or at_max:
                            center_bad = True
                            break
                        j += 1
                    if center_bad:
                        print("\n" + "!"*60)
                        print("  *** ADVERTENCIA: el ajuste FALLO y el/los CENTRO/S estan")
                        print("      en el limite del rango -- los parametros son INVALIDOS ***")
                        print("!"*60)
                        save = input(f"  Guardar igualmente en {default_csv}? [s/N/nombre]: ").strip()
                        if not save:
                            save = 'n'
                    else:
                        print("\n  AVISO: el ajuste no convergio (success=False). Los parametros pueden ser incorrectos.")
                        save = input(f"  Guardar de todas formas en {default_csv}? [s/N/nombre]: ").strip()
                        if not save:
                            save = 'n'
                else:
                    save = input(f"\n  Guardar parametros del ajuste en CSV como {default_csv}? [S/n/nombre]: ").strip()
                if save.lower() not in ('n', 'no'):
                    if save and save.lower() not in ('s', 'si', 'y', 'yes'):
                        csv_out = save
                    else:
                        csv_out = default_csv
                    save_fit_to_csv(filename, linename, hjd_value, vhelio, fit_result,
                                    csv_filename=csv_out)
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
