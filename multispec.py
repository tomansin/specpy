#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
multispec.py - Visualizador interactivo de espectros FITS en formato MULTISPEC.

Permite navegar entre órdenes, normalizar cada uno con parámetros heredados del
anterior, acumular las normalizaciones y guardar un FITS multispec con los
órdenes normalizados reemplazados.

Uso:
    multispec.py <archivo.fits>
"""

import sys
import os
import argparse


def _load_heavy_imports():
    """Importa las dependencias pesadas (matplotlib, scipy, astropy, etc.).

    Se difiere hasta despues de parsear los argumentos para que `-h`
    responda al instante sin pagar el costo de cargar estas librerias.
    """
    global plt, np, GridSpec, SpanSelector, fits
    global read_fits_multi, fit_cont_sigma, calculate_smart_ylimits, \
        save_fit_to_csv, find_closest_line
    global interactive_gaussian_fitting, _print_vr_summary, \
        VR_WARNING_THRESHOLD, HJD_KEYS, time_to_hjd, _prompt_vhelio_correction

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.gridspec import GridSpec
    from matplotlib.widgets import SpanSelector
    from astropy.io import fits

    from specpy.utils import (read_fits_multi, fit_cont_sigma,
                              calculate_smart_ylimits, save_fit_to_csv,
                              find_closest_line)

    import spec
    spec._load_heavy_imports()
    interactive_gaussian_fitting = spec.interactive_gaussian_fitting
    _print_vr_summary = spec._print_vr_summary
    VR_WARNING_THRESHOLD = spec.VR_WARNING_THRESHOLD
    HJD_KEYS = spec.HJD_KEYS
    time_to_hjd = spec.time_to_hjd
    _prompt_vhelio_correction = spec._prompt_vhelio_correction


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_multispec(filename):
    """
    Carga un archivo FITS MULTISPEC.

    Returns
    -------
    header, all_wavelengths, all_fluxes
        all_wavelengths : ndarray (norders, nwave)
        all_fluxes      : ndarray (norders, nwave)
    """
    try:
        header, all_wavelengths, all_fluxes = read_fits_multi(filename)
        norders = all_wavelengths.shape[0]
        print(f"LOADED MULTISPEC: {filename}")
        print(f"  Ordenes : {norders}")
        print(f"  Rango   : {all_wavelengths[0,0]:.2f} - {all_wavelengths[-1,-1]:.2f} A")
        print(f"  Puntos  : {all_wavelengths.shape[1]} por orden")

        hjd_value = None
        hjd_nota = None
        for key in HJD_KEYS:
            if key in header:
                hjd_value, hjd_nota = time_to_hjd(key, header[key])
                break
        if hjd_value is None:
            print("  WARNING: No HJD keyword found in header")
        else:
            print(f"  HJD ({hjd_nota}): {hjd_value:.10f}")

        return header, all_wavelengths, all_fluxes
    except FileNotFoundError:
        print(f"Error: archivo '{filename}' no encontrado")
        return None, None, None
    except Exception as e:
        print(f"Error cargando multispec: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# Normalización con soporte multispec
# ---------------------------------------------------------------------------

def interactive_normalization(wavelength, flux, filename,
                               seed_params=None, order_label=None):
    """
    Normalización interactiva para un orden MULTISPEC.

    Ajusta un único polinomio Chebyshev a todo el rango espectral.
    El usuario puede excluir regiones (líneas espectrales) con el SpanSelector.

    Parameters
    ----------
    seed_params : int or None
        Orden del polinomio heredado del orden anterior.  None → usa 5.
    order_label : str, optional
        Texto adicional para el título.

    Returns
    -------
    norm_flux : ndarray or None
    action    : 'done' | 'prev_order' | 'next_order'
    poly_order_out : int   (orden del polinomio, para heredar al siguiente orden)
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(5, 1)
    ax_spec = fig.add_subplot(gs[:3, 0])
    ax_norm = fig.add_subplot(gs[3:, 0], sharex=ax_spec)

    if isinstance(seed_params, dict):
        _init_order = seed_params.get('poly_order', 5)
        _init_excl  = seed_params.get('excluded_regions', [])
    elif isinstance(seed_params, int):
        _init_order = seed_params
        _init_excl  = []
    else:
        _init_order = 5
        _init_excl  = []

    poly_order = [_init_order]
    sigma_lower = [3.0]
    sigma_upper = [5.0]
    excluded_regions = []   # [[wmin, wmax], ...] — intervalos excluidos del fit
    excluded_patches = []   # axvspan grises
    continuum_line = [None]
    fit_model = [None]
    reject_scatter = [None]
    span_selector = [None]
    selection_active = [False]
    order_action = [None]

    title_base = f'Normalizacion: {os.path.basename(filename)}'
    if order_label:
        title_base += f'  {order_label}'

    ax_spec.plot(wavelength, flux, 'k-', linewidth=1.5, label='Espectro')
    ax_spec.set_xlabel('Wavelength (A)', fontsize=12)
    ax_spec.set_ylabel('Flux', fontsize=12)
    ax_spec.set_title(title_base, fontsize=14, fontweight='bold')
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

    # ── Fit y display ─────────────────────────────────────────────────────────

    def _refit():
        """Ajusta Chebyshev al rango completo menos las regiones excluidas."""
        if reject_scatter[0] is not None:
            try:
                reject_scatter[0].remove()
            except Exception:
                pass
            reject_scatter[0] = None

        mask = np.ones(len(wavelength), dtype=bool)
        for wmin, wmax in excluded_regions:
            mask &= ~((wavelength >= wmin) & (wavelength <= wmax))
        if int(np.sum(mask)) <= poly_order[0] + 1:
            fit_model[0] = None
            return
        try:
            cont_model, reject, _ = fit_cont_sigma(
                wavelength[mask], flux[mask],
                model='chebyshev', order=poly_order[0],
                use_sigma_clip=True, sigma_lower=sigma_lower[0], sigma_upper=sigma_upper[0])
            fit_model[0] = cont_model
            if reject is not None and len(reject[0]) > 0:
                reject_scatter[0] = ax_spec.scatter(
                    reject[0], reject[1],
                    c='green', s=14, alpha=0.6, marker='x', zorder=5)
        except Exception as e:
            print(f'  Error ajuste: {e}')
            fit_model[0] = None

    def _update(preserve_view=False):
        if preserve_view:
            sx, sy = ax_spec.get_xlim(), ax_spec.get_ylim()
        if continuum_line[0] is not None:
            continuum_line[0].remove(); continuum_line[0] = None
        if fit_model[0] is not None:
            continuum_line[0], = ax_spec.plot(
                wavelength, fit_model[0](wavelength), 'r-', linewidth=2, alpha=0.9)
        ax_spec.legend(loc='best')

        ax_norm.clear()
        ax_norm.set_xlabel('Wavelength (A)', fontsize=12)
        ax_norm.set_ylabel('Flujo normalizado', fontsize=12)
        ax_norm.set_title('Previsualizacion normalizada', fontsize=14, fontweight='bold')
        ax_norm.grid(True, alpha=0.3, linestyle='--')
        ax_norm.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax_norm.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        if fit_model[0] is not None:
            nfl = flux / fit_model[0](wavelength)
            ax_norm.plot(wavelength, nfl, 'k-', linewidth=1.5)
            mask = np.ones(len(wavelength), dtype=bool)
            for wmin, wmax in excluded_regions:
                mask &= ~((wavelength >= wmin) & (wavelength <= wmax))
            if np.any(mask):
                s = max(np.std(nfl[mask]), 0.01)
                ax_norm.set_ylim(1 - 10*s, 1 + 5*s)

        if preserve_view:
            ax_spec.set_xlim(sx); ax_spec.set_ylim(sy)
        fig.canvas.draw_idle()

    def _refit_and_update():
        _refit(); _update(preserve_view=True)

    # ── SpanSelector para exclusiones ─────────────────────────────────────────

    def _onselect_excl(xmin, xmax):
        if xmin > xmax: xmin, xmax = xmax, xmin
        excluded_regions.append([xmin, xmax])
        excluded_patches.append(
            ax_spec.axvspan(xmin, xmax, alpha=0.3, color='gray', zorder=1))
        print(f'  Excluido [{xmin:.2f}, {xmax:.2f}] A  '
              f'({len(excluded_regions)} exclusion(es))')
        _refit_and_update()

    def _toggle_selection():
        selection_active[0] = not selection_active[0]
        if selection_active[0]:
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            span_selector[0] = SpanSelector(
                ax_spec, _onselect_excl, 'horizontal', useblit=True,
                props=dict(alpha=0.25, facecolor='gray'),
                interactive=True, drag_from_anywhere=True)
            print('  Seleccion de EXCLUSION activada (arrastra sobre lineas a ignorar)')
        else:
            if span_selector[0] is not None:
                span_selector[0].disconnect_events(); span_selector[0] = None
            print('  Seleccion desactivada')

    # ── Teclado ───────────────────────────────────────────────────────────────

    def on_key(event):
        if event.key == 'a':
            _toggle_selection()

        elif event.key == 'e':
            if excluded_regions:
                excluded_regions.pop()
                patch = excluded_patches.pop()
                try: patch.remove()
                except Exception: pass
                print(f'  Ultima exclusion eliminada. Quedan {len(excluded_regions)}.')
                _refit_and_update()
            else:
                print('  No hay exclusiones.')

        elif event.key in ('+', '='):
            poly_order[0] = min(poly_order[0] + 1, 20)
            print(f'  Orden polinomio: {poly_order[0]}')
            _refit_and_update()

        elif event.key == '-':
            poly_order[0] = max(poly_order[0] - 1, 0)
            print(f'  Orden polinomio: {poly_order[0]}')
            _refit_and_update()

        elif event.key == 'up':
            sigma_lower[0] = round(sigma_lower[0] + 0.5, 1)
            print(f'  sigma_lower: {sigma_lower[0]}  sigma_upper: {sigma_upper[0]}')
            _refit_and_update()

        elif event.key == 'down':
            sigma_lower[0] = max(0.5, round(sigma_lower[0] - 0.5, 1))
            print(f'  sigma_lower: {sigma_lower[0]}  sigma_upper: {sigma_upper[0]}')
            _refit_and_update()

        elif event.key == 'right':
            sigma_upper[0] = round(sigma_upper[0] + 0.5, 1)
            print(f'  sigma_lower: {sigma_lower[0]}  sigma_upper: {sigma_upper[0]}')
            _refit_and_update()

        elif event.key == 'left':
            sigma_upper[0] = max(0.5, round(sigma_upper[0] - 0.5, 1))
            print(f'  sigma_lower: {sigma_lower[0]}  sigma_upper: {sigma_upper[0]}')
            _refit_and_update()

        elif event.key in ('(', ')'):
            order_action[0] = 'prev_order' if event.key == '(' else 'next_order'
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            plt.close(fig)

        elif event.key == 'q':
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            plt.close(fig)

    def on_scroll(event):
        if event.inaxes not in (ax_spec, ax_norm): return
        factor = 0.85 if event.button == 'up' else 1.0 / 0.85
        xmin, xmax = ax_spec.get_xlim(); xc = event.xdata
        ax_spec.set_xlim(xc + (xmin - xc) * factor, xc + (xmax - xc) * factor)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    fig.canvas.mpl_connect('scroll_event', on_scroll)

    print("\n" + "="*60)
    print("MODO NORMALIZACION MULTISPEC")
    print("="*60)
    print(f"  Orden polinomio inicial: {poly_order[0]}")
    print(f"  sigma_lower: {sigma_lower[0]}  sigma_upper: {sigma_upper[0]}")
    print("  a         activar/desactivar exclusion de regiones (arrastra en gris)")
    print("  e         eliminar ultima exclusion")
    print("  +/-       subir/bajar orden del polinomio Chebyshev")
    print("  arr.arriba/abajo   subir/bajar sigma_lower (paso 0.5)")
    print("  arr.der/izq        subir/bajar sigma_upper (paso 0.5)")
    print("  (/)       orden anterior/siguiente (guarda automaticamente)")
    print("  q         confirmar y cerrar")
    print("="*60)

    # Pre-poblar exclusiones del orden si las había
    for wmin, wmax in _init_excl:
        wmin_c = max(float(wmin), float(wavelength[0]))
        wmax_c = min(float(wmax), float(wavelength[-1]))
        if wmin_c < wmax_c:
            excluded_regions.append([wmin_c, wmax_c])
            excluded_patches.append(
                ax_spec.axvspan(wmin_c, wmax_c, alpha=0.3, color='gray', zorder=1))

    _refit()
    _update()
    plt.tight_layout()
    plt.show()

    # ── Post-show ─────────────────────────────────────────────────────────────
    def _compute_norm():
        _refit()
        if fit_model[0] is None:
            return None
        return flux / fit_model[0](wavelength)

    params_out = {'poly_order': poly_order[0],
                  'excluded_regions': list(excluded_regions)}

    if order_action[0] in ('prev_order', 'next_order'):
        return _compute_norm(), order_action[0], params_out

    if fit_model[0] is None:
        return None, 'done', params_out

    resp = input("  Guardar normalizacion? [S/n]: ").strip().lower()
    if resp in ('n', 'no'):
        print("  Normalizacion descartada.")
        return None, 'done', params_out

    return _compute_norm(), 'done', params_out



# ---------------------------------------------------------------------------
# Guardado multispec
# ---------------------------------------------------------------------------

def save_multispec(out_path, header, all_fluxes):
    """Guarda un array de flujos (norders, nwave) como FITS multispec."""
    new_header = header.copy()
    hdu = fits.PrimaryHDU(all_fluxes.astype(np.float32), header=new_header)
    hdu.writeto(out_path, overwrite=True)
    print(f"  Guardado: {out_path}")


# ---------------------------------------------------------------------------
# Visualizador principal
# ---------------------------------------------------------------------------

def plot_multispec(all_wavelengths, all_fluxes, filename, header, start_order=None,
                   start_mode=None):
    """
    Visualizador interactivo para espectros MULTISPEC.

    Teclas:
      (/)   orden anterior/siguiente
      n     normalizar orden activo (hereda regiones del anterior)
      x     guardar FITS con todos los órdenes (normalizados donde corresponda)
      q     cerrar
      w     ventana de zoom / Enter aplicar / z resetear
      d     ajuste de gaussianas
      h     imprimir header

    Parameters
    ----------
    start_order : int or None
        Numero de orden (1-indexado) con el que abrir el visualizador.
    start_mode : str or None
        Si es 'normalize' o 'fit_gaussians', entra directamente en ese modo
        para el orden activo (sin mostrar primero el visualizador principal)
        y sale al terminar sin reabrirlo.
    """
    norders = all_wavelengths.shape[0]
    norm_fluxes = {}          # {order_idx: norm_flux}
    order_norm_params = {}    # {order_idx: {'poly_order': int, 'excluded_regions': [...]}}

    idx0 = 0
    if start_order is not None:
        if 1 <= start_order <= norders:
            idx0 = start_order - 1
        else:
            print(f"  WARNING: orden {start_order} fuera de rango "
                  f"[1, {norders}]. Abriendo en el orden 1.")

    current_order = [idx0]
    current = [all_wavelengths[idx0].copy(), all_fluxes[idx0].copy()]
    is_normalized = [False]
    is_windowed = [False]

    hjd_value = None
    for key in HJD_KEYS:
        if key in header:
            hjd_value, _ = time_to_hjd(key, header[key])
            break
    # La correccion heliocentrica solo se pregunta al momento de guardar el
    # ajuste (ver _prompt_vhelio_correction); durante el ajuste interactivo
    # las velocidades radiales se muestran sin corregir.
    vhelio = 0.0

    def order_label():
        n = len(norm_fluxes)
        norm_info = f' {n}norm' if n > 0 else ''
        marker = '*' if is_normalized[0] else ''
        return f'[O {current_order[0]+1}{marker}/{norders}{norm_info}]'

    def make_title():
        norm_tag   = ' (norm)' if is_normalized[0] else ''
        window_tag = ' (crop)' if is_windowed[0] else ''
        ol = order_label()
        if is_windowed[0]:
            return (f'Spectrum{window_tag}{norm_tag} {ol}: {os.path.basename(filename)}'
                    f' [{current[0][0]:.1f} - {current[0][-1]:.1f} A]')
        return f'Spectrum{norm_tag} {ol}: {os.path.basename(filename)}'

    def _orig_wl():
        return all_wavelengths[current_order[0]]

    def _orig_fl():
        idx = current_order[0]
        return norm_fluxes[idx] if idx in norm_fluxes else all_fluxes[idx]

    def save_multispec_prompt(prompt=False):
        base, ext = os.path.splitext(os.path.basename(filename))
        n_norm = len(norm_fluxes)
        if n_norm == 0:
            if not prompt:
                print("  Nada que guardar: ningun orden normalizado")
            return
        out_path = base + '_norm' + ext
        if prompt:
            resp = input(f"\n  Guardar multispec ({n_norm}/{norders} ordenes norm.) "
                         f"como {out_path}? [S/n/nombre]: ").strip()
            if resp.lower() in ('n', 'no'):
                return
            if resp and resp.lower() not in ('s', 'si', 'y', 'yes'):
                out_path = resp
        flux_out = all_fluxes.copy()
        for idx, nfl in norm_fluxes.items():
            flux_out[idx] = nfl
        save_multispec(out_path, header, flux_out)

    def run_session():
        pending = [None]
        fig, ax = plt.subplots(figsize=(12, 6))

        wl = current[0]; fl = current[1]
        ax.plot(wl, fl, 'k-', linewidth=1.5, label='Spectrum')
        ax.set_xlabel('Wavelength (A)', fontsize=12)
        ax.set_ylabel('Flux', fontsize=12)
        ax.set_title(make_title(), fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(*calculate_smart_ylimits(fl))
        ax.set_xlim(wl[0], wl[-1])
        ax.legend(loc='best')
        fig.tight_layout()

        def refresh_toolbar():
            if hasattr(fig.canvas, 'toolbar') and fig.canvas.toolbar is not None:
                fig.canvas.toolbar.update()

        window_limits = [None, None]
        span_selector = [None]

        def onselect_window(xmin, xmax):
            if xmin > xmax: xmin, xmax = xmax, xmin
            window_limits[0] = xmin; window_limits[1] = xmax

        def activate_window_mode():
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            window_limits[0] = window_limits[1] = None
            span_selector[0] = SpanSelector(
                ax, onselect_window, 'horizontal', useblit=True,
                props=dict(alpha=0.3, facecolor='blue'),
                interactive=True, drag_from_anywhere=True)
            print("  Modo ventana: click y arrastra, luego Enter para aplicar")

        def apply_window():
            xmin, xmax = window_limits
            if xmin is None:
                print("  No hay ventana definida.")
                return
            if span_selector[0] is not None:
                span_selector[0].disconnect_events(); span_selector[0] = None
            wl_orig = _orig_wl()
            fl_orig = _orig_fl()
            mask = (wl_orig >= xmin) & (wl_orig <= xmax)
            wl_w = wl_orig[mask]; fl_w = fl_orig[mask]
            if len(wl_w) == 0:
                print("  Ventana sin datos."); return
            if is_normalized[0]:
                print("  Aviso: re-ventanear descarta la normalizacion")
                is_normalized[0] = False
            current[0] = wl_w; current[1] = fl_w
            ax.lines[0].set_data(wl_w, fl_w)
            ax.set_title(make_title(), fontsize=14, fontweight='bold')
            ax.set_ylim(*calculate_smart_ylimits(fl_w))
            ax.set_xlim(xmin, xmax)
            is_windowed[0] = True
            refresh_toolbar(); fig.canvas.draw_idle()

        def reset_view():
            is_normalized[0] = False; is_windowed[0] = False
            wl_orig = _orig_wl(); fl_orig = _orig_fl()
            current[0] = wl_orig; current[1] = fl_orig
            ax.lines[0].set_data(wl_orig, fl_orig)
            ax.set_title(make_title(), fontsize=14, fontweight='bold')
            ax.set_ylim(*calculate_smart_ylimits(fl_orig))
            ax.set_xlim(wl_orig[0], wl_orig[-1])
            refresh_toolbar(); fig.canvas.draw_idle()
            print("  Vista reseteada")

        def switch_order(new_idx):
            current_order[0] = new_idx
            is_windowed[0] = False
            idx = new_idx
            wl_new = all_wavelengths[idx]
            fl_new = norm_fluxes[idx] if idx in norm_fluxes else all_fluxes[idx]
            is_normalized[0] = (idx in norm_fluxes)
            current[0] = wl_new; current[1] = fl_new
            ax.lines[0].set_data(wl_new, fl_new)
            ax.set_title(make_title(), fontsize=14, fontweight='bold')
            ax.set_ylim(*calculate_smart_ylimits(fl_new))
            ax.set_xlim(wl_new[0], wl_new[-1])
            refresh_toolbar(); fig.canvas.draw_idle()

        def on_key(event):
            if event.key.lower() == 'q':
                pending[0] = 'quit'; plt.close(fig)
            elif event.key == '(':
                switch_order((current_order[0] - 1) % norders)
            elif event.key == ')':
                switch_order((current_order[0] + 1) % norders)
            elif event.key.lower() == 'w':
                activate_window_mode()
            elif event.key == 'enter':
                apply_window()
            elif event.key.lower() == 'z':
                reset_view()
            elif event.key.lower() == 'n':
                pending[0] = 'normalize'; plt.close(fig)
            elif event.key.lower() == 'd':
                pending[0] = 'fit_gaussians'; plt.close(fig)
            elif event.key.lower() == 'x':
                save_multispec_prompt()
            elif event.key.lower() == 'h':
                print("\n" + "="*50)
                print(repr(header))
                print("="*50)

        def on_scroll(event):
            if event.inaxes != ax: return
            factor = 0.85 if event.button == 'up' else 1.0 / 0.85
            xmin, xmax = ax.get_xlim(); xc = event.xdata
            ax.set_xlim(xc + (xmin - xc) * factor, xc + (xmax - xc) * factor)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        print("\n" + "="*50)
        print("MULTISPEC INSTRUCTIONS:")
        print(f"  (/)       orden anterior/siguiente  ({norders} ordenes)")
        print("  n         normalizar orden activo")
        print("  x         guardar FITS (ordenes normalizados reemplazados)")
        print("  w/Enter   ventana de zoom")
        print("  z         resetear vista")
        print("  d         ajuste de gaussianas")
        print("  q         cerrar")
        print("="*50)

        plt.show()

        if pending[0] == 'quit':
            save_multispec_prompt(prompt=True)
            return None
        return pending[0]

    if start_mode in ('normalize', 'fit_gaussians'):
        action = start_mode
    else:
        action = run_session()

    while True:
        if action == 'normalize':
            idx = current_order[0]
            wl_orig = all_wavelengths[idx]
            fl_orig = all_fluxes[idx]
            ol = order_label()
            while True:
                norm_fl, norm_action, saved_params = interactive_normalization(
                    wl_orig, fl_orig, filename,
                    seed_params=order_norm_params.get(idx),
                    order_label=ol)
                order_norm_params[idx] = saved_params
                if norm_fl is not None:
                    norm_fluxes[idx] = norm_fl
                    current[0] = all_wavelengths[idx]
                    current[1] = norm_fl
                    is_normalized[0] = True
                    is_windowed[0] = False
                if norm_action in ('prev_order', 'next_order'):
                    delta = -1 if norm_action == 'prev_order' else 1
                    idx = (idx + delta) % norders
                    current_order[0] = idx
                    wl_orig = all_wavelengths[idx]
                    fl_orig = all_fluxes[idx]
                    is_normalized[0] = (idx in norm_fluxes)
                    current[0] = wl_orig
                    current[1] = norm_fluxes[idx] if idx in norm_fluxes else fl_orig
                    is_windowed[0] = False
                    ol = order_label()
                else:
                    break
            n = len(norm_fluxes)
            print(f"  {n}/{norders} ordenes normalizados.")
            if start_mode is not None:
                save_multispec_prompt(prompt=True)
                break

        elif action == 'fit_gaussians':
            fit_result = interactive_gaussian_fitting(
                current[0], current[1], filename, None, vhelio)
            if fit_result is not None:
                high_vr = _print_vr_summary(fit_result, vhelio)
                if high_vr:
                    print(f"\n  Aviso: velocidades radiales superan {VR_WARNING_THRESHOLD:.0f} km/s.")
                    if input("  Continuar? [s/N]: ").strip().lower() not in ('s', 'si', 'y', 'yes'):
                        if start_mode is None:
                            action = run_session()
                            continue
                        else:
                            break
            if fit_result is not None and hjd_value is not None:
                linename = 'line'
                if 'g1_center' in fit_result.params:
                    try:
                        li = find_closest_line(fit_result.params['g1_center'].value)
                        linename = li.get('name', linename)
                        if len(linename) > 4: linename = linename[-4:]
                    except Exception:
                        pass
                default_csv = f"fitted_{linename}.csv"
                save = input(f"\n  Guardar ajuste en {default_csv}? [S/n/nombre]: ").strip()
                if save.lower() not in ('n', 'no'):
                    csv_out = save if save and save.lower() not in ('s', 'si', 'y', 'yes') else default_csv
                    save_vhelio = _prompt_vhelio_correction(header)
                    save_fit_to_csv(filename, linename, hjd_value, save_vhelio,
                                    fit_result, csv_filename=csv_out)
            if start_mode is not None:
                break
        else:
            break

        action = run_session()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualizador interactivo MULTISPEC')
    parser.add_argument('filename', help='Archivo FITS MULTISPEC')
    parser.add_argument('-o', '--order', type=int, default=None,
                        help='Numero de orden (1-indexado) con el que abrir el visualizador')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--normalized', action='store_true',
                            help='Entrar directamente en modo normalizacion (orden activo) y salir al terminar')
    mode_group.add_argument('--gaussian', action='store_true',
                            help='Entrar directamente en modo ajuste de gaussianas (orden activo) y salir al terminar')
    args = parser.parse_args()

    _load_heavy_imports()

    header, all_wavelengths, all_fluxes = load_multispec(args.filename)
    if all_wavelengths is None:
        sys.exit(1)

    start_mode = None
    if args.normalized:
        start_mode = 'normalize'
    elif args.gaussian:
        start_mode = 'fit_gaussians'

    plot_multispec(all_wavelengths, all_fluxes, args.filename, header,
                   start_order=args.order, start_mode=start_mode)


if __name__ == "__main__":
    main()
