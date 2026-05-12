#!/home/tansin/.conda/envs/spec/bin/python
# -*- coding: utf-8 -*-
"""
multispec.py - Visualizador interactivo de espectros FITS en formato MULTISPEC.

Permite navegar entre órdenes, normalizar cada uno con parámetros heredados del
anterior, acumular las normalizaciones y guardar un FITS multispec con los
órdenes normalizados reemplazados.

Uso:
    multispec.py <archivo.fits>
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from scipy.interpolate import Akima1DInterpolator
from astropy.io import fits

from specpy.utils import (read_fits_multi, fit_cont_sigma, mask_generator,
                          calculate_smart_ylimits, save_fit_to_csv,
                          find_closest_line, vr, vrerr,
                          gaussian, fit_lines)
from spec import (interactive_gaussian_fitting, _print_vr_summary,
                  VR_WARNING_THRESHOLD, HJD_KEYS)


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
        hjd_key_used = None
        for key in HJD_KEYS:
            if key in header:
                hjd_value = header[key]
                hjd_key_used = key
                break
        if hjd_value is None:
            print("  WARNING: No HJD keyword found in header")
        else:
            print(f"  HJD ({hjd_key_used}): {hjd_value}")

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
    Interfaz gráfica para normalizar un espectro por ajuste de continuo.

    Igual a la de spec.py pero con soporte para navegar entre órdenes:
      (  → orden anterior   )  → orden siguiente

    Parameters
    ----------
    seed_params : list of dict, optional
        Lista de {'regions': [[w1,w2],...], 'poly_order': n} con la que
        pre-poblar los rangos (herencia del orden anterior).
    order_label : str, optional
        Texto adicional para el título (p. ej. '[O 5/26]').

    Returns
    -------
    norm_flux : ndarray or None
    action    : 'done' | 'prev_order' | 'next_order'
    seed_out  : list of dict  (parámetros para pasar al siguiente orden)
    """
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(5, 1)
    ax_spec = fig.add_subplot(gs[:3, 0])
    ax_norm = fig.add_subplot(gs[3:, 0], sharex=ax_spec)

    ranges = []
    current_poly_order = [5]
    span_selector = [None]
    selection_active = [False]
    continuum_line = [None]
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

    # ── Helpers internos ──────────────────────────────────────────────────────

    def range_span(r):
        return (min(reg[0] for reg in r['regions']),
                max(reg[1] for reg in r['regions']))

    def refit_range(r, ridx):
        if r['poly_line'] is not None:
            r['poly_line'].remove(); r['poly_line'] = None
        if r['reject_scatter'] is not None:
            r['reject_scatter'].remove(); r['reject_scatter'] = None
        if not r['regions']:
            r['fit_model'] = None; return
        mask = mask_generator(wavelength, r['regions'])
        n_pts = int(np.sum(mask))
        if n_pts <= r['poly_order'] + 1:
            r['fit_model'] = None; return
        try:
            cont_model, reject, _ = fit_cont_sigma(
                wavelength[mask], flux[mask],
                model='chebyshev', order=r['poly_order'],
                use_sigma_clip=True, sigma_lower=3, sigma_upper=3)
            r['fit_model'] = cont_model
            r['reject'] = reject
        except Exception:
            r['fit_model'] = None; return
        wmin, wmax = range_span(r)
        x_plot = np.linspace(wmin, wmax, 500)
        is_active = (ridx == len(ranges) - 1)
        label = f'R{ridx+1} ord{r["poly_order"]} (pol activo)' if is_active else '_nolegend_'
        r['poly_line'], = ax_spec.plot(x_plot, cont_model(x_plot),
                                        color='red', linewidth=2, linestyle='--',
                                        alpha=0.85, label=label)
        if reject is not None and len(reject[0]) > 0:
            r['reject_scatter'] = ax_spec.scatter(
                reject[0], reject[1], c='green', s=12, alpha=0.5, marker='x', zorder=5)

    def build_continuum():
        valid = [r for r in ranges if r.get('fit_model') is not None]
        if not valid:
            return None
        if len(valid) == 1:
            return valid[0]['fit_model'](wavelength)
        all_x, all_y = [], []
        for r in valid:
            wmin, wmax = range_span(r)
            n_pts = max(80, int((wmax - wmin) / (wavelength[-1] - wavelength[0]) * 800))
            x_r = np.linspace(wmin, wmax, n_pts)
            all_x.append(x_r); all_y.append(r['fit_model'](x_r))
        all_x = np.concatenate(all_x); all_y = np.concatenate(all_y)
        order = np.argsort(all_x); all_x, all_y = all_x[order], all_y[order]
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
        except Exception:
            return None

    def update_display(preserve_view=False):
        if preserve_view:
            saved_xlim      = ax_spec.get_xlim()
            saved_ylim_spec = ax_spec.get_ylim()
        if continuum_line[0] is not None:
            continuum_line[0].remove(); continuum_line[0] = None
        continuum = build_continuum()
        if continuum is not None:
            valid = [r for r in ranges if r.get('fit_model') is not None]
            wmin_all = min(range_span(r)[0] for r in valid)
            wmax_all = max(range_span(r)[1] for r in valid)
            draw_mask = (wavelength >= wmin_all) & (wavelength <= wmax_all)
            continuum_line[0], = ax_spec.plot(
                wavelength[draw_mask], continuum[draw_mask], 'r-', linewidth=2, alpha=0.9)
        ax_spec.legend(loc='best')
        ax_norm.clear()
        ax_norm.set_xlabel('Wavelength (A)', fontsize=12)
        ax_norm.set_ylabel('Flujo normalizado', fontsize=12)
        ax_norm.set_title('Previsualizacion normalizada', fontsize=14, fontweight='bold')
        ax_norm.grid(True, alpha=0.3, linestyle='--')
        ax_norm.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Referencia (1.0)')
        ax_norm.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        if continuum is not None:
            norm_flux = flux / continuum
            ax_norm.plot(wavelength, norm_flux, 'k-', linewidth=1.5, label='Normalizado')
            region_mask = np.zeros(len(wavelength), dtype=bool)
            for r in ranges:
                if r['regions']:
                    region_mask |= mask_generator(wavelength, r['regions'])
            if np.any(region_mask):
                s = np.std(norm_flux[region_mask]); s = s if s > 0 else 0.05
                ax_norm.set_ylim(1 - 15 * s, 1 + 5 * s)
        ax_norm.legend(loc='best')
        if preserve_view:
            ax_spec.set_xlim(saved_xlim); ax_spec.set_ylim(saved_ylim_spec)
        fig.canvas.draw_idle()

    def seal_range(r):
        if r.get('poly_line') is not None:
            r['poly_line'].set_label('_nolegend_')
        for patch in r['region_patches']:
            try: patch.remove()
            except Exception: pass
        r['region_patches'].clear()
        if r.get('used_points_handle') is not None:
            try: r['used_points_handle'].remove()
            except Exception: pass
        if r['regions']:
            mask = mask_generator(wavelength, r['regions'])
            r['used_points_handle'], = ax_spec.plot(
                wavelength[mask], flux[mask],
                'b+', markersize=5, alpha=0.6, zorder=3, label='_nolegend_')
        else:
            r['used_points_handle'] = None

    def activate_range(r, _ridx=None):
        if r.get('used_points_handle') is not None:
            try: r['used_points_handle'].remove()
            except Exception: pass
            r['used_points_handle'] = None
        for region in r['regions']:
            patch = ax_spec.axvspan(region[0], region[1], alpha=0.15, color='red', zorder=0)
            r['region_patches'].append(patch)

    def _prepopulate(seed_params):
        wmin_data, wmax_data = wavelength[0], wavelength[-1]
        valid_seeds = []
        for sp in seed_params:
            valid_regions = [
                [max(r[0], wmin_data), min(r[1], wmax_data)]
                for r in sp['regions']
                if r[0] < wmax_data and r[1] > wmin_data
            ]
            if valid_regions:
                valid_seeds.append({'regions': valid_regions, 'poly_order': sp['poly_order']})
        if not valid_seeds:
            return
        for i, sp in enumerate(valid_seeds):
            r = {'regions': list(sp['regions']), 'poly_order': sp['poly_order'],
                 'fit_model': None, 'reject': None, 'poly_line': None,
                 'reject_scatter': None, 'region_patches': [], 'used_points_handle': None}
            for region in sp['regions']:
                patch = ax_spec.axvspan(region[0], region[1], alpha=0.15, color='red', zorder=0)
                r['region_patches'].append(patch)
            ranges.append(r)
            refit_range(r, len(ranges) - 1)
            if i < len(valid_seeds) - 1:
                seal_range(r)
        current_poly_order[0] = valid_seeds[-1]['poly_order']

    def onselect(xmin, xmax):
        if xmin > xmax: xmin, xmax = xmax, xmin
        if not ranges:
            ranges.append({'regions': [], 'poly_order': current_poly_order[0],
                           'fit_model': None, 'reject': None, 'poly_line': None,
                           'reject_scatter': None, 'region_patches': [],
                           'used_points_handle': None})
        r = ranges[-1]; ridx = len(ranges) - 1
        r['regions'].append([xmin, xmax])
        patch = ax_spec.axvspan(xmin, xmax, alpha=0.15, color='red', zorder=0)
        r['region_patches'].append(patch)
        refit_range(r, ridx)
        update_display(preserve_view=True)

    def toggle_selection():
        selection_active[0] = not selection_active[0]
        if selection_active[0]:
            if span_selector[0] is not None:
                span_selector[0].disconnect_events()
            span_selector[0] = SpanSelector(
                ax_spec, onselect, 'horizontal', useblit=True,
                props=dict(alpha=0.2, facecolor='red'),
                interactive=True, drag_from_anywhere=True)
            print("  Seleccion ACTIVADA")
        else:
            if span_selector[0] is not None:
                span_selector[0].disconnect_events(); span_selector[0] = None
            print("  Seleccion DESACTIVADA")

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
            ranges.append({'regions': [], 'poly_order': current_poly_order[0],
                           'fit_model': None, 'reject': None, 'poly_line': None,
                           'reject_scatter': None, 'region_patches': [],
                           'used_points_handle': None})
            print(f"  Rango {ridx_new} sellado. Rango {ridx_new + 1} activo.")
            fig.canvas.draw_idle()

        elif event.key == 'e':
            if not ranges: return
            r = ranges[-1]; ridx = len(ranges) - 1
            if not r['regions']:
                ranges.pop()
                print(f"  Rango {ridx + 1} (vacio) eliminado.")
                if ranges: activate_range(ranges[-1], len(ranges) - 1)
                update_display(preserve_view=True)
                return
            patch = r['region_patches'].pop(); patch.remove()
            removed = r['regions'].pop()
            print(f"  Region [{removed[0]:.2f}, {removed[1]:.2f}] A eliminada.")
            if r['regions']:
                refit_range(r, ridx)
            else:
                if r['poly_line'] is not None:
                    r['poly_line'].remove(); r['poly_line'] = None
                if r['reject_scatter'] is not None:
                    r['reject_scatter'].remove(); r['reject_scatter'] = None
                ranges.pop()
                if ranges: activate_range(ranges[-1], len(ranges) - 1)
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
    print("MODO NORMALIZACION")
    print("="*60)
    print("  a         activar/desactivar seleccion de regiones")
    print("  b         sellar rango activo e iniciar nuevo rango")
    print("  e         eliminar ultima region")
    print("  +/-       subir/bajar orden del polinomio")
    print("  (/)       orden anterior/siguiente (guarda automaticamente)")
    print("  q         confirmar y cerrar")
    print("="*60)

    if seed_params:
        _prepopulate(seed_params)
        update_display()

    plt.tight_layout()
    plt.show()

    # ── Helpers de calculo ────────────────────────────────────────────────────
    def _extract_seed():
        return [{'regions': list(r['regions']), 'poly_order': r['poly_order']}
                for r in ranges if r['regions']]

    def _compute_norm():
        valid = [r for r in ranges if r['regions']]
        if not valid:
            return None
        for r in valid:
            mask = mask_generator(wavelength, r['regions'])
            if np.sum(mask) > r['poly_order'] + 1:
                try:
                    cont_model, _, _ = fit_cont_sigma(
                        wavelength[mask], flux[mask],
                        model='chebyshev', order=r['poly_order'],
                        use_sigma_clip=True)
                    r['fit_model'] = cont_model
                except Exception:
                    r['fit_model'] = None
            else:
                r['fit_model'] = None
        continuum = build_continuum()
        return flux / continuum if continuum is not None else None

    # ── Calculo final ─────────────────────────────────────────────────────────
    seed_out = _extract_seed()

    if order_action[0] in ('prev_order', 'next_order'):
        return _compute_norm(), order_action[0], seed_out

    valid_ranges = [r for r in ranges if r['regions']]
    if not valid_ranges:
        return None, 'done', seed_out

    resp = input("  Guardar normalizacion? [S/n]: ").strip().lower()
    if resp in ('n', 'no'):
        print("  Normalizacion descartada.")
        return None, 'done', seed_out

    return _compute_norm(), 'done', seed_out


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

def plot_multispec(all_wavelengths, all_fluxes, filename, header):
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
    """
    norders = all_wavelengths.shape[0]
    norm_fluxes = {}          # {order_idx: norm_flux}
    current_order = [0]
    current = [all_wavelengths[0].copy(), all_fluxes[0].copy()]
    is_normalized = [False]
    is_windowed = [False]
    _norm_seed = [None]

    hjd_value = None
    for key in HJD_KEYS:
        if key in header:
            hjd_value = header[key]
            break
    vhelio = float(header['VHELIO']) if 'VHELIO' in header else 0.0

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

    action = run_session()

    while True:
        if action == 'normalize':
            idx = current_order[0]
            wl_orig = all_wavelengths[idx]
            fl_orig = all_fluxes[idx]
            ol = order_label()
            while True:
                norm_fl, norm_action, _norm_seed[0] = interactive_normalization(
                    wl_orig, fl_orig, filename,
                    seed_params=_norm_seed[0],
                    order_label=ol)
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

        elif action == 'fit_gaussians':
            fit_result = interactive_gaussian_fitting(
                current[0], current[1], filename, None, vhelio)
            if fit_result is not None:
                high_vr = _print_vr_summary(fit_result, vhelio)
                if high_vr:
                    print(f"\n  Aviso: velocidades radiales superan {VR_WARNING_THRESHOLD:.0f} km/s.")
                    if input("  Continuar? [s/N]: ").strip().lower() not in ('s', 'si', 'y', 'yes'):
                        action = run_session()
                        continue
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
                    save_fit_to_csv(filename, linename, hjd_value, vhelio,
                                    fit_result, csv_filename=csv_out)
        else:
            break

        action = run_session()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualizador interactivo MULTISPEC')
    parser.add_argument('filename', help='Archivo FITS MULTISPEC')
    args = parser.parse_args()

    header, all_wavelengths, all_fluxes = load_multispec(args.filename)
    if all_wavelengths is None:
        sys.exit(1)

    plot_multispec(all_wavelengths, all_fluxes, args.filename, header)


if __name__ == "__main__":
    main()
