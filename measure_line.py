#!/home/tansin/miniconda3/envs/spec/bin/python
# -*- coding: utf-8 -*-
# plot_spec.py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import argparse
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import SpanSelector
from specpy.utils import read_fits_simple, fit_cont_sigma, mask_generator, gaussian, fit_lines

def load_spectrum(filename):
    """Load multispec data from FITS file"""
    try:
        header, wavelength, flux = read_fits_simple(filename)
        print(f"LOADED SPECTRUM FROM {filename}")
        print(f"  Wavelength range: {wavelength[0]:.2f} - {wavelength[-1]:.2f}")
        print(f"  Number of points: {len(wavelength)}")
        
        # Verificar que existe HJD en el header
        hjd_keys = ['HJD', 'JD', 'MJD','MJD-OBS', 'OHP DRS BJD']  # Variantes posibles
        hjd_value = None
        hjd_key_used = None
        
        for key in hjd_keys:
            if key in header:
                hjd_value = header[key]
                hjd_key_used = key
                break
        
        if hjd_value is None:
            print(f"\n❌ ERROR: No HJD or similar keyword found in header")
            print("   Las keywords buscadas fueron:", hjd_keys)
            print("   Las keywords disponibles son:")
            for i, key in enumerate(header.keys()):
                if i < 20:  # Mostrar solo las primeras 20
                    print(f"     {key}: {header[key]}")
            return None, None, None
        
        print(f"  HJD ({hjd_key_used}): {hjd_value}")
        return header, wavelength, flux
        
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return None, None, None
    except Exception as e:
        print(f"Error loading spectrum: {e}")
        return None, None, None

def calculate_smart_ylimits(data, central_fraction=0.8, margin_factor=0.1):
    """Calculate smart Y-axis limits"""
    if len(data) == 0:
        return 0, 1
    
    n_points = len(data)
    start_idx = int((1 - central_fraction) / 2 * n_points)
    end_idx = int((1 + central_fraction) / 2 * n_points)
    
    start_idx = max(0, start_idx)
    end_idx = min(n_points, end_idx)
    
    central_data = data[start_idx:end_idx]
    
    if len(central_data) == 0:
        return np.min(data), np.max(data)
    
    ymin = np.percentile(central_data, 1)
    ymax = np.percentile(central_data, 99)
    
    margin = (ymax - ymin) * margin_factor
    ymin = max(ymin - margin, 0) if np.min(data) >= 0 else ymin - margin
    ymax = ymax + margin
    
    return ymin, ymax

def plot_spectrum(wavelength, flux, filename):
    """Plot spectrum that can be closed with 'q' key"""
    if wavelength is None or flux is None or len(wavelength) == 0:
        print("Error: No data to plot")
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(wavelength, flux, 'k-', linewidth=1.5, label='Spectrum')
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    
    ax.set_title(f'Spectrum: {os.path.basename(filename)}', fontsize=14, fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ymin, ymax = calculate_smart_ylimits(flux)
    ax.set_ylim(ymin, ymax)
    
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend(loc='best')
    
    fig.tight_layout()
    
    is_closed = False
    
    def on_key(event):
        nonlocal is_closed
        if event.key == 'q' or event.key == 'Q':
            print("\n" + "="*50)
            is_closed = True
            plt.close(fig)
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("• Press 'q' in the figure window to close and continue")
    print("="*50)
    
    plt.show()
    
    return is_closed

def interactive_continuum_fitting(wavelength, flux, filename, initial_regions=None):
    """
    Interactive function to select continuum regions and adjust polynomial order.
    
    Returns:
        tuple: (normalized_flux, selected_regions, polynomial_order)
    """
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(5,1)
    ax_spectrum = fig.add_subplot(gs[:3,0])
    ax_normalized = fig.add_subplot(gs[3:,0])
    
    # Variables globales para el ajuste
    current_poly_order = 1
    selected_regions = initial_regions if initial_regions else []
    continuum_line = None
    masked_points = None
    reject_points = None
    span_selector = None  # Variable para el selector
    selection_active = False  # Estado de selección
    
    # Plot original spectrum
    ax_spectrum.plot(wavelength, flux, 'k-', linewidth=1.5, label='Original Spectrum')
    ax_spectrum.set_xlabel('Wavelength (Å)', fontsize=12)
    ax_spectrum.set_ylabel('Flux', fontsize=12)
    ax_spectrum.set_title(f'Select Continuum Regions: {os.path.basename(filename)}', 
                         fontsize=14, fontweight='bold')
    
    # Calculate limits for spectrum plot
    ymin, ymax = calculate_smart_ylimits(flux)
    ax_spectrum.set_ylim(ymin, ymax)
    ax_spectrum.set_xlim() # No sé porque pero es necesario que este esta línea para que no se rompa visualización en span selector mode
    ax_spectrum.legend()
    
    ax_spectrum.grid(True, alpha=0.3, linestyle='--')
    ax_spectrum.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Plot for normalized spectrum (initially empty)
    ax_normalized.set_xlabel('Wavelength (Å)', fontsize=12)
    ax_normalized.set_ylabel('Normalized Flux', fontsize=12)
    ax_normalized.set_title('Normalized Spectrum Preview', fontsize=14, fontweight='bold')
    ax_normalized.grid(True, alpha=0.3, linestyle='--')
    ax_normalized.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Reference (1.0)')
    ax_normalized.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
    def update_continuum_fit():
        """Update the continuum fit based on current regions and polynomial order"""
        nonlocal continuum_line, masked_points, reject_points
        
        # Remove old continuum line if exists
        if continuum_line is not None:
            continuum_line.remove()
            continuum_line = None
        
        # Remove old masked points if exist
        if masked_points is not None:
            masked_points.remove()
            masked_points = None
        
        # Remove old reject points if exist
        if reject_points is not None:
            reject_points.remove()
            reject_points = None
        
        if selected_regions and len(selected_regions) > 0:
            try:
                # Create mask for selected regions
                mask = mask_generator(wavelength, selected_regions)
                
                if np.sum(mask) > current_poly_order + 1:  # Need enough points for fitting
                    # Fit continuum
                    cont_model, reject, _ = fit_cont_sigma(wavelength[mask], flux[mask], 
                                           model='chebyshev', order=current_poly_order,
                                           use_sigma_clip=True, sigma_lower=3, sigma_upper=3)
                    
                    # Calculate continuum over entire wavelength range
                    continuum = cont_model(wavelength)
                    
                    # Plot continuum line
                    continuum_line, = ax_spectrum.plot(wavelength, continuum, 
                                                     'r-', linewidth=2, 
                                                     label=f'Continuum (order {current_poly_order})')
                    
                    # Plot masked points used for fitting
                    masked_points = ax_spectrum.plot(wavelength[mask], flux[mask], 
                                                    'bo', markersize=2, alpha=0.2,
                                                    label='Points used for fitting')[0]
                    # Plot sigmaclipped points 
                    reject_points = ax_spectrum.plot(reject[0], reject[1], 
                                                    'gx', markersize=4, alpha=0.6,
                                                    label='Points reject for sigmaclip')[0]
                    
                    ax_spectrum.legend()
                    
                    # Calculate normalized flux
                    normalized_flux = flux / continuum
                    
                    # Update normalized spectrum plot
                    ax_normalized.clear()
                    ax_normalized.plot(wavelength, normalized_flux, 'k-', linewidth=1.5, 
                                     label='Normalized Spectrum')
                    ax_normalized.set_xlabel('Wavelength (Å)', fontsize=12)
                    ax_normalized.set_ylabel('Normalized Flux', fontsize=12)
                    ax_normalized.set_title('Normalized Spectrum Preview', 
                                          fontsize=14, fontweight='bold')
                    ax_normalized.grid(True, alpha=0.3, linestyle='--')
                    ax_normalized.axhline(y=1, color='red', linestyle='--', 
                                        alpha=0.5, label='Reference (1.0)')
                    ax_normalized.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
                    
                    # Set reasonable limits for normalized plot
                    norm_ymin, norm_ymax = calculate_smart_ylimits(normalized_flux[mask], margin_factor=0.5)
                    ax_normalized.set_ylim(norm_ymin, norm_ymax)
                    ax_normalized.legend(loc='best')
                    
                    fig.canvas.draw_idle()
                    return normalized_flux
                
                else:
                    print(f"Warning: Not enough points ({np.sum(mask)}) for polynomial order {current_poly_order}")
                    
            except Exception as e:
                print(f"Error in continuum fitting: {e}")
        
        return None
    
    def onselect(xmin, xmax):
        """Callback for region selection"""
        if not selection_active:
            print("⚠ Selection is not active. Press 'a' to activate.")
            return
        
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        
        selected_regions.append([xmin, xmax])
                        
        print(f"✓ Selected region {len(selected_regions)}: [{xmin:.2f}, {xmax:.2f}] Å")
        
        # Update continuum fit
        update_continuum_fit()
        # update_instructions()
    
    def toggle_selection():
        """Toggle region selection mode"""
        nonlocal selection_active, span_selector
        
        selection_active = not selection_active
        
        if selection_active:
            # Create SpanSelector
            span_selector = SpanSelector(ax_spectrum, onselect, 'horizontal',
                                    useblit=True,
                                    props=dict(alpha=0.2, facecolor='red'),
                                    interactive=True,
                                    drag_from_anywhere=True)
            print("✓ Region selection ACTIVATED")
        else:
            # Destroy SpanSelector
            if span_selector is not None:
                span_selector.set_active(False)
                # Para desactivarlo completamente, necesitamos remover la conexión
                span_selector = None
            print("✓ Region selection DEACTIVATED")
        
        # update_instructions()
            
    def on_key_press(event):
        """Handle key press events"""
        nonlocal current_poly_order, selection_active, span_selector
        
        if event.key == 'a':
            # Toggle selection mode
            toggle_selection()
        
        elif event.key == 'r':
            # Remove last region
            if selected_regions:
                region = selected_regions.pop()
                print(f"✗ Removed region: [{region[0]:.2f}, {region[1]:.2f}] Å")
                update_continuum_fit()
                # update_instructions()
        
        elif event.key == 'c':
            # Clear all regions
            if selected_regions:
                selected_regions.clear()
                print("✗ Cleared all regions")
                update_continuum_fit()
                # update_instructions()
        
        elif event.key == '+':
            # Increase polynomial order
            if current_poly_order < 20:  # Maximum order
                current_poly_order += 1
                print(f"↑ Increased polynomial order to {current_poly_order}")
                update_continuum_fit()
                # update_instructions()
            else:
                print(f"Maximum polynomial order reached ({current_poly_order})")
        
        elif event.key == '-':
            # Decrease polynomial order
            if current_poly_order > 0:  # Minimum order
                current_poly_order -= 1
                print(f"↓ Decreased polynomial order to {current_poly_order}")
                update_continuum_fit()
                # update_instructions()
            else:
                print(f"Minimum polynomial order reached ({current_poly_order})")
        
        elif event.key == 'q':
            # Finish selection
            print("\n" + "="*50)
            print(f"FINISHED CONTINUUM FITTING")
            print(f"Polynomial order: {current_poly_order}")
            print(f"Total regions selected: {len(selected_regions)}")
            for i, region in enumerate(selected_regions, 1):
                print(f"  Region {i}: [{region[0]:.2f}, {region[1]:.2f}] Å")
            print("="*50)
            plt.close(fig)
    
    # Show instructions in console
    print("\n" + "="*60)
    print("INTERACTIVE CONTINUUM FITTING")
    print("="*60)
    print("\nInstructions:")
    print("• Press 'a' to ACTIVATE/DEACTIVATE region selection")
    print("• When active: click and drag to select continuum regions")
    print("• Press 'r' to remove last region")
    print("• Press 'c' to clear all regions")
    print("• Press '+' to increase polynomial order")
    print("• Press '-' to decrease polynomial order")
    print("• Press 'q' to finish selection")
    print("\nTip: Select regions that represent the continuum (no lines)")
    print("="*60)
    
    # Initial update
    # update_instructions()
    
    # Connect keyboard events
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Update layout and show
    plt.tight_layout()
    plt.show()
    
    # After closing, calculate final normalized flux
    if selected_regions:
        try:
            mask = mask_generator(wavelength, selected_regions)
            if np.sum(mask) > current_poly_order + 1:
                cont_model, _,_ = fit_cont_sigma(wavelength[mask], flux[mask], 
                                       model='chebyshev', order=current_poly_order,
                                       use_sigma_clip=True)
                normalized_flux = flux / cont_model(wavelength)
                return normalized_flux, selected_regions, current_poly_order
        except Exception as e:
            print(f"Error in final normalization: {e}")
        
    
    return flux, selected_regions, current_poly_order  # Return original if normalization failed

def interactive_lines_fitting(wavelength, flux, filename, params_dict=None):
    """
    Interactive function to fit gaussians to absorption lines.
    
    Returns:
    --------
    result : lmfit.ModelResult or None
        Resultado del ajuste, o None si no se realizó ajuste
    """
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot()
    
    # Variables globales para las gaussianas
    gaussians = []  # Lista de tuplas (center, amplitude, fwhm) - MANUALES
    fitted_gaussians = []  # Lista de tuplas (center, amplitude, fwhm) - AJUSTADAS
    current_gaussian = None  # Lista: [center, depth, fwhm_left, fwhm_right]
    step = None  # 'center', 'fwhm_left', 'fwhm_right'
    gaussian_lines = []  # Líneas de gaussianas manuales (incluye marcadores)
    fitted_lines = []  # Líneas de gaussianas ajustadas
    gaussian_patches = []  # Elementos temporales durante creación
    result = None  # Para guardar el resultado del ajuste
    
    # Variables para el modo de eliminación de puntos
    erase_mode = False  # Estado del modo de eliminación
    removed_indices = []  # Índices de puntos eliminados (basados en arrays ORIGINALES)
    
    # Hacer copias inmutables de los datos originales
    original_wavelength = wavelength.copy()
    original_flux = flux.copy()
    
    # Datos de trabajo (se actualizan cuando se eliminan puntos)
    current_wavelength = wavelength.copy()
    current_flux = flux.copy()
    
    # Plot spectrum
    spectrum_line, = ax.plot(current_wavelength, current_flux, 'k-', linewidth=1.5, label='Spectrum')
    ax.set_xlabel('Wavelength (Å)', fontsize=12)
    ax.set_ylabel('Flux', fontsize=12)
    ax.set_title(f'Fit Lines: {os.path.basename(filename)}', 
                 fontsize=14, fontweight='bold')
    
    # Calcular límites iniciales
    ymin, ymax = calculate_smart_ylimits(current_flux)
    ax.set_ylim(ymin, ymax)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Reference (1.0)')
    
    # Texto para instrucciones
    status_text = ax.text(0.02, 0.98, '',
                         transform=ax.transAxes,
                         fontsize=10,
                         verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update_status():
        """Actualizar el texto de estado"""
        if erase_mode:
            status = "ERASE MODE - Click points to remove them (press 'r' to restore all)"
        elif step is None:
            status = "Ready - Press 'g' to add a new gaussian"
            if gaussians:
                status += " or 'a' to auto-fit"
        elif step == 'center':
            status = "Step 1/3: Click to set CENTER and DEPTH"
        elif step == 'fwhm_left':
            status = "Step 2/3: Click LEFT of center for FWHM"
        elif step == 'fwhm_right':
            status = "Step 3/3: Click RIGHT of center for FWHM"
        
        gaussian_info = f"\nGaussians: {len(gaussians)} (dashed lines)"
        if gaussians:
            gaussian_info += "\n"
            for i, (c, a, fwhm) in enumerate(gaussians, 1):
                depth = 1 + a
                gaussian_info += f"  G{i}: λ={c:.2f}Å, Depth={depth:.3f}, FWHM={fwhm:.2f}Å\n"
        
        if result is not None:
            gaussian_info += f"\nBest fit (red solid line)"
            gaussian_info += f"\n  Reduced χ²: {result.redchi:.3e}"
        
        if removed_indices:
            gaussian_info += f"\n\nRemoved points: {len(removed_indices)}"
            gaussian_info += f"\nRemaining points: {len(current_wavelength)}"
        
        status_text.set_text(status + gaussian_info)
        fig.canvas.draw_idle()
    
    def draw_gaussian(center, amplitude, fwhm, color='blue', alpha=0.7, 
                      linewidth=2, linestyle='-', label=None):
        """Dibujar una gaussiana en el plot"""
        height = amplitude
        y_gauss = gaussian(current_wavelength, center, height, fwhm)
        line, = ax.plot(current_wavelength, y_gauss, color=color, linewidth=linewidth,
                       linestyle=linestyle, alpha=alpha, label=label)
        return line
    
    def clear_current_gaussian():
        """Limpiar la gaussiana actual y sus indicadores"""
        nonlocal current_gaussian, step
        
        current_gaussian = None
        step = None
        
        for item in gaussian_patches:
            try:
                if hasattr(item, 'remove'):
                    item.remove()
            except:
                pass
        
        gaussian_patches.clear()
        update_status()
    
    def finalize_gaussian():
        """Finalizar la gaussiana actual y añadirla a la lista"""
        nonlocal current_gaussian, gaussians
        
        if current_gaussian and len(current_gaussian) == 4:
            center, depth, left_wl, right_wl = current_gaussian
            fwhm_val = right_wl - left_wl
            amplitude = -(1 - depth)
            
            clear_current_gaussian()
            
            color = plt.cm.tab10(len(gaussians) % 10)
            line = draw_gaussian(center, amplitude, fwhm_val, 
                            color=color, alpha=0.7, linestyle='--',
                            label=f'G{len(gaussians)+1}: λ={center:.2f}Å')
            gaussian_lines.append(line)
            
            idx = np.argmin(np.abs(current_wavelength - center))
            flux_at_center = current_flux[idx]
            
            center_marker = ax.plot(center, flux_at_center, 'r|', 
                                markersize=10, alpha=0.7)[0]
            gaussian_lines.append(center_marker)
            
            half_max = amplitude / 2
            left_marker = ax.plot(center - fwhm_val/2, flux_at_center + half_max,
                                'b<', markersize=8, alpha=0.7)[0]
            right_marker = ax.plot(center + fwhm_val/2, flux_at_center + half_max,
                                'b>', markersize=8, alpha=0.7)[0]
            gaussian_lines.extend([left_marker, right_marker])
            
            fwhm_line = ax.plot([center - fwhm_val/2, center + fwhm_val/2],
                            [flux_at_center + half_max, flux_at_center + half_max],
                            'b:', alpha=0.5, linewidth=1)[0]
            gaussian_lines.append(fwhm_line)
            
            gaussians.append((center, amplitude, fwhm_val))
            
            print(f"\nAdded Gaussian {len(gaussians)}:")
            print(f"  Center: {center:.2f} Å")
            print(f"  Depth: {depth:.3f}")
            print(f"  FWHM: {fwhm_val:.2f} Å")
            
            ax.legend(loc='best')
            update_status()
    
    def toggle_erase_mode():
        """Activar/desactivar modo de eliminación de puntos"""
        nonlocal erase_mode, step
        
        if step is not None:
            print("Cannot enter erase mode while defining a gaussian")
            print("Press 'escape' to cancel current gaussian first")
            return
        
        erase_mode = not erase_mode
        
        if erase_mode:
            print("\n" + "="*40)
            print("ERASE MODE ACTIVATED")
            print("Click on data points to remove them")
            print("Press 'e' again to deactivate")
            print("Press 'r' to restore all removed points")
            print("="*40)
        else:
            print("\nErase mode DEACTIVATED")
        
        update_status()
    
    def remove_point_at(x, y):
        """Eliminar el punto más cercano a la posición (x, y) si está dentro de la tolerancia"""
        nonlocal removed_indices, current_wavelength, current_flux, spectrum_line, erase_mode
        
        if len(current_wavelength) == 0:
            return
        
        # TOLERANCIAS - AJUSTABLES
        x_tolerance = 0.5  # Å - tolerancia en wavelength
        y_tolerance = 0.01  # tolerancia en flux (relativa al rango)
        
        # Calcular rango de flux para normalizar la tolerancia en Y
        flux_range = np.max(current_flux) - np.min(current_flux)
        if flux_range > 0:
            y_tolerance_abs = y_tolerance * flux_range
        else:
            y_tolerance_abs = y_tolerance
        print(y_tolerance_abs)
        
        # Encontrar TODOS los puntos dentro de la tolerancia en los datos ACTUALES
        distances_x = np.abs(current_wavelength - x)
        distances_y = np.abs(current_flux - y)
        
        # Crear máscara para puntos dentro de la tolerancia
        within_tolerance = (distances_x <= x_tolerance) & (distances_y <= y_tolerance_abs)
        
        # Verificar si hay algún punto dentro de la tolerancia
        if not np.any(within_tolerance):
            print(f"\nNo point found within tolerance ({x_tolerance:.2f}Å, {y_tolerance:.1%} of flux range)")
            print(f"Click was at: λ={x:.2f}Å, flux={y:.3f}")
            
            # Mostrar el punto más cercano para debugging
            nearest_idx = np.argmin(distances_x + distances_y)
            print(f"Nearest point: λ={current_wavelength[nearest_idx]:.2f}Å, flux={current_flux[nearest_idx]:.3f}")
            print(f"Distance: Δλ={distances_x[nearest_idx]:.3f}Å, Δflux={distances_y[nearest_idx]:.4f}")
            return
        
        # Encontrar el índice del punto más cercano dentro de la tolerancia
        distances_combined = distances_x / x_tolerance + distances_y / y_tolerance_abs
        distances_combined[~within_tolerance] = np.inf
        
        nearest_idx = np.argmin(distances_combined)
        
        # Obtener el índice ORIGINAL correspondiente
        # Necesitamos mapear del índice actual al índice original
        visible_mask = np.ones(len(original_wavelength), dtype=bool)
        visible_mask[removed_indices] = False
        visible_indices = np.where(visible_mask)[0]
        
        if nearest_idx >= len(visible_indices):
            print("Error: Índice fuera de rango")
            return
            
        original_idx = visible_indices[nearest_idx]
        
        # Verificar si el punto ya fue eliminado (no debería pasar)
        if original_idx in removed_indices:
            print(f"\nWarning: Point at {original_wavelength[original_idx]:.2f}Å already removed")
            return
        
        # Agregar a la lista de índices eliminados (usando índice ORIGINAL)
        removed_indices.append(original_idx)
        removed_indices.sort()  # Mantener ordenados
        
        # Recalcular los datos visibles
        visible_mask[original_idx] = False
        current_wavelength = original_wavelength[visible_mask]
        current_flux = original_flux[visible_mask]
        
        # Actualizar el plot del espectro
        spectrum_line.set_data(current_wavelength, current_flux)
        
        # Actualizar todas las gaussianas y líneas
        for line in gaussian_lines + fitted_lines:
            try:
                line.remove()
            except:
                pass
        
        gaussian_lines.clear()
        fitted_lines.clear()
        fitted_gaussians.clear()
        
        # Redibujar las gaussianas manuales si existen
        for i, (center, amplitude, fwhm_val) in enumerate(gaussians):
            color = plt.cm.tab10(i % 10)
            line = draw_gaussian(center, amplitude, fwhm_val, 
                              color=color, alpha=0.7, linestyle='--',
                              label=f'G{i+1}: λ={center:.2f}Å')
            gaussian_lines.append(line)
        
        # Recalcular límites
        ymin_new, ymax_new = calculate_smart_ylimits(current_flux)
        ax.set_ylim(ymin_new, ymax_new)
        
        point_wl = original_wavelength[original_idx]
        point_flux = original_flux[original_idx]
        
        print(f"\n✓ Removed point at {point_wl:.2f}Å (flux={point_flux:.3f})")
        print(f"  Distance from click: Δλ={abs(point_wl - x):.3f}Å, Δflux={abs(point_flux - y):.4f}")
        print(f"  Total removed points: {len(removed_indices)}")
        print(f"  Remaining points: {len(current_wavelength)}")
        
        # SALIR DEL MODO DE BORRADO DESPUÉS DE ELIMINAR UN PUNTO
        erase_mode = False
        print("\nAuto-exited erase mode")
        
        fig.canvas.draw_idle()
        update_status()
    
    def restore_all_points():
        """Restaurar todos los puntos eliminados"""
        nonlocal removed_indices, current_wavelength, current_flux, spectrum_line
        
        if removed_indices:
            # Limpiar lista de índices eliminados
            removed_indices.clear()
            
            # Restaurar todos los datos originales
            current_wavelength = original_wavelength.copy()
            current_flux = original_flux.copy()
            
            # Actualizar el plot del espectro
            spectrum_line.set_data(current_wavelength, current_flux)
            
            # Recalcular límites
            ymin_new, ymax_new = calculate_smart_ylimits(current_flux)
            ax.set_ylim(ymin_new, ymax_new)
            
            print(f"\n✓ Restored all {len(current_wavelength)} points")
            
            fig.canvas.draw_idle()
            update_status()
        else:
            print("\nNo points to restore")
    
    def on_click(event):
        """Manejar clicks del mouse"""
        nonlocal step, current_gaussian, erase_mode
        
        if event.inaxes != ax:
            return
        
        x, y = event.xdata, event.ydata
        
        if erase_mode:
            remove_point_at(x, y)
            return
        
        if step is None:
            return
        
        if step == 'center':
            center = x
            depth = y
            
            if depth > 1.2:
                depth = 1.0
            elif depth < 0:
                depth = 0.1
            
            vline_center = ax.vlines(center, ymin, ymax, color='blue', 
                                linestyle='--', alpha=0.5)
            gaussian_patches.append(vline_center)
            
            text = ax.text(center, ymax*0.95, f'λ={center:.2f}Å', 
                        color='blue', ha='center', fontsize=9)
            gaussian_patches.append(text)
            
            current_gaussian = [center, depth]
            step = 'fwhm_left'
            
            print(f"\nCenter set at {center:.2f} Å")
            print(f"Depth set to {depth:.3f}")
            print(f"Now click LEFT of center for FWHM measurement")
            
        elif step == 'fwhm_left':
            if x < current_gaussian[0]:
                current_gaussian.append(x)
                step = 'fwhm_right'
                
                vline_left = ax.vlines(x, ymin, ymax, color='orange', 
                                    linestyle='--', alpha=0.5)
                gaussian_patches.append(vline_left)
                
                print(f"\nLeft FWHM point at {x:.2f} Å")
                print(f"Now click RIGHT of center for FWHM measurement")
            else:
                print("\nPlease click to the LEFT of the center")
                return
        
        elif step == 'fwhm_right':
            if x > current_gaussian[0]:
                current_gaussian.append(x)
                
                vline_right = ax.vlines(x, ymin, ymax, color='orange', 
                                    linestyle='--', alpha=0.5)
                gaussian_patches.append(vline_right)
                
                print(f"\nRight FWHM point at {x:.2f} Å")
                finalize_gaussian()
            else:
                print("\nPlease click to the RIGHT of the center")
                return
        
        update_status()
    
    def combine_manual_and_json_params():
        """Combinar parámetros de JSON con gaussianas manuales"""
        combined_params = {}
        
        if not params_dict:
            print("\nNo JSON parameters provided")
            return None
            
        print("\n" + "="*60)
        print("COMBINING JSON PARAMETERS WITH MANUAL GAUSSIANS")
        print("="*60)
        
        # Contar gaussianas en el JSON
        json_gaussians = {}
        for key, value in params_dict.items():
            if key.startswith('g') and '_' in key:
                parts = key.split('_')
                if len(parts) == 2:
                    try:
                        g_num = int(parts[0][1:])  # Número después de 'g'
                        param_type = parts[1]  # 'center', 'sigma', 'amplitude'
                        
                        if g_num not in json_gaussians:
                            json_gaussians[g_num] = {}
                        
                        # Guardar el valor y si varía
                        json_gaussians[g_num][param_type] = value
                    except ValueError:
                        continue
        
        n_json_gauss = len(json_gaussians)
        n_manual_gauss = len(gaussians)
        
        print(f"JSON has {n_json_gauss} gaussian(s)")
        print(f"You defined {n_manual_gauss} gaussian(s) manually")
        
        # Si no hay gaussianas manuales, usar solo el JSON
        if n_manual_gauss == 0:
            print("\nUsing JSON parameters only (no manual gaussians)")
            return params_dict.copy()
        
        # Verificar que coincida el número
        if n_manual_gauss != n_json_gauss:
            print(f"\n⚠ WARNING: Number of gaussians doesn't match!")
            print(f"  JSON has {n_json_gauss}, you defined {n_manual_gauss}")
            
            # Preguntar qué hacer
            print("\nOptions:")
            print("1. Use only JSON parameters (ignore manual)")
            print("2. Use only manual gaussians (ignore JSON)")
            print("3. Cancel fitting")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == "1":
                print("\nUsing JSON parameters only")
                return params_dict.copy()
            elif choice == "2":
                print("\nUsing manual gaussians only")
                # Crear parámetros desde gaussianas manuales
                fit_params = {}
                for i, (center, amplitude, fwhm_val) in enumerate(gaussians, 1):
                    prefix = f'g{i}_'
                    sigma = fwhm_val / 2.3548200
                    amp = amplitude * sigma * np.sqrt(2 * np.pi)
                    
                    fit_params[f'{prefix}center'] = {'value': center}
                    fit_params[f'{prefix}sigma'] = {'value': sigma}
                    fit_params[f'{prefix}amplitude'] = {'value': amp}
                
                # Agregar fondo del JSON si existe
                if 'bkg_c' in params_dict:
                    fit_params['bkg_c'] = params_dict['bkg_c']
                else:
                    fit_params['bkg_c'] = {'value': 1.0, 'vary': False}
                
                return fit_params
            else:
                print("\nFitting cancelled")
                return None
        
        # Combinar parámetros: usar valores manuales para parámetros libres en JSON
        print(f"\nCombining parameters (using manual values for free JSON parameters):")
        
        for i in range(1, n_manual_gauss + 1):
            if i in json_gaussians and i-1 < len(gaussians):
                json_gauss = json_gaussians[i]
                manual_center, manual_amplitude, manual_fwhm = gaussians[i-1]
                
                prefix = f'g{i}_'
                
                # Convertir manual a formato lmfit
                manual_sigma = manual_fwhm / 2.3548200
                manual_amp = manual_amplitude * manual_sigma * np.sqrt(2 * np.pi)
                
                # CENTER
                if 'center' in json_gauss:
                    if json_gauss['center'].get('vary', True):  # Si es libre en JSON
                        combined_params[f'{prefix}center'] = {'value': manual_center}
                        print(f"  G{i} center: Using MANUAL value ({manual_center:.2f}Å) - JSON was FREE")
                    else:  # Si es fijo en JSON
                        combined_params[f'{prefix}center'] = json_gauss['center']
                        print(f"  G{i} center: Using JSON FIXED value ({json_gauss['center'].get('value', '?'):.2f}Å)")
                else:
                    combined_params[f'{prefix}center'] = {'value': manual_center}
                    print(f"  G{i} center: Using MANUAL value ({manual_center:.2f}Å)")
                
                # SIGMA (convertir de FWHM)
                if 'sigma' in json_gauss:
                    if json_gauss['sigma'].get('vary', True):  # Si es libre en JSON
                        combined_params[f'{prefix}sigma'] = {'value': manual_sigma}
                        print(f"  G{i} sigma: Using MANUAL value (from FWHM={manual_fwhm:.2f}Å)")
                    else:  # Si es fijo en JSON
                        combined_params[f'{prefix}sigma'] = json_gauss['sigma']
                        print(f"  G{i} sigma: Using JSON FIXED value")
                else:
                    combined_params[f'{prefix}sigma'] = {'value': manual_sigma}
                    print(f"  G{i} sigma: Using MANUAL value (from FWHM={manual_fwhm:.2f}Å)")
                
                # AMPLITUDE
                if 'amplitude' in json_gauss:
                    if json_gauss['amplitude'].get('vary', True):  # Si es libre en JSON
                        combined_params[f'{prefix}amplitude'] = {'value': manual_amp}
                        print(f"  G{i} amplitude: Using MANUAL value (from depth)")
                    else:  # Si es fijo en JSON
                        combined_params[f'{prefix}amplitude'] = json_gauss['amplitude']
                        print(f"  G{i} amplitude: Using JSON FIXED value")
                else:
                    combined_params[f'{prefix}amplitude'] = {'value': manual_amp}
                    print(f"  G{i} amplitude: Using MANUAL value (from depth)")
        
        # Copiar el fondo del JSON
        if 'bkg_c' in params_dict:
            combined_params['bkg_c'] = params_dict['bkg_c']
        else:
            combined_params['bkg_c'] = {'value': 1.0, 'vary': False}
        
        print("\n✓ Combined parameters ready")
        return combined_params
    
    def on_key_press(event):
        """Manejar eventos de teclado"""
        nonlocal step, current_gaussian, result, erase_mode, fitted_lines
        
        # 'q' siempre funciona para salir
        if event.key == 'q':
            plt.close(fig)
            return
        
        # Modo de borrado
        if erase_mode:
            if event.key == 'e':
                toggle_erase_mode()
            elif event.key == 'r':
                restore_all_points()
            elif event.key == 'escape':
                toggle_erase_mode()
            return
        
        # Modo normal (no borrado)
        if event.key == 'e':
            toggle_erase_mode()
            return
        
        if event.key == 'g':
            if step is None:
                step = 'center'
                current_gaussian = []
                update_status()
        
        elif event.key == 'a':
            # Decidir qué parámetros usar para el ajuste
            fit_params = {}
            
            # PRIORIDAD: Si hay JSON Y gaussianas manuales, combinarlos
            if params_dict and gaussians:
                fit_params = combine_manual_and_json_params()
                if fit_params is None:
                    return  # Usuario canceló
            # Si solo hay JSON y no hay gaussianas manuales
            elif params_dict and not gaussians:
                print("\n" + "="*60)
                print("FITTING WITH JSON PARAMETERS...")
                print("="*60)
                fit_params = params_dict.copy()
                gaussian_count = sum(1 for key in params_dict.keys() 
                                   if key.startswith('g') and '_center' in key)
                print(f"Using {gaussian_count} gaussian(s) from JSON file")
            # Si solo hay gaussianas manuales y no hay JSON
            elif gaussians and len(gaussians) > 0 and not params_dict:
                print("\n" + "="*60)
                print("FITTING WITH MANUAL GAUSSIANS...")
                print("="*60)
                
                for i, (center, amplitude, fwhm) in enumerate(gaussians, 1):
                    prefix = f'g{i}_'
                    
                    # Convertir FWHM a sigma para el ajuste
                    sigma = fwhm / 2.3548200
                    
                    # Convertir height a amplitude de gaussiana
                    amp = amplitude * sigma * np.sqrt(2 * np.pi)
                    
                    # Agregar parámetros en formato lmfit
                    fit_params[f'{prefix}center'] = {'value': center, 'min': center-sigma, 'max': center+sigma}
                    fit_params[f'{prefix}sigma'] = {'value': sigma}
                    fit_params[f'{prefix}amplitude'] = {'value': amp}
                
                # Agregar fondo (fijo en 1 si ya normalizamos)
                fit_params['bkg_c'] = {'value': 1.0, 'vary': False}
                
                print(f"Using {len(gaussians)} manual gaussian(s)")
            else:
                print("\nFirst define at least one gaussian with 'g' or provide JSON with --params")
                return
            
            try:
                # Llamar a fit_lines con los parámetros
                result = fit_lines(current_wavelength, current_flux, fit_params)
                
                # Mostrar el reporte completo de lmfit
                print("\n" + "="*60)
                print("FIT REPORT")
                print("="*60)
                print(result.fit_report())
                
                # Limpiar líneas anteriores de ajuste
                for line in fitted_lines:
                    line.remove()
                fitted_lines.clear()
                
                # Dibujar ajuste
                fitted_line = ax.plot(current_wavelength, result.best_fit,
                                     color='red', linewidth=2,
                                     linestyle='-', alpha=0.9,
                                     label='Best fit')[0]
                fitted_lines.append(fitted_line)
                
                ax.legend(loc='best')
                update_status()
                
                print(f"\n✓ Fitting completed successfully")
                print("="*60)
                
            except Exception as e:
                print(f"\n❌ Error in fit: {e}")
                import traceback
                traceback.print_exc()
        
        elif event.key == 'r':
            if gaussians:
                removed = gaussians.pop()
                
                if gaussian_lines:
                    elements_to_remove = min(5, len(gaussian_lines))
                    for _ in range(elements_to_remove):
                        if gaussian_lines:
                            line = gaussian_lines.pop()
                            line.remove()
                
                # Limpiar también líneas de ajuste si existen
                for line in fitted_lines:
                    line.remove()
                fitted_lines.clear()
                result = None
                
                ax.legend(loc='best')
                update_status()
        
        elif event.key == 'c':
            if gaussians or fitted_lines:
                for line in gaussian_lines:
                    line.remove()
                gaussians.clear()
                gaussian_lines.clear()
                
                for line in fitted_lines:
                    line.remove()
                fitted_lines.clear()
                result = None
                
                ax.legend(loc='best')
                update_status()
        
        elif event.key == 'escape':
            if step is not None:
                clear_current_gaussian()
    
    # Conectar eventos
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Mostrar instrucciones iniciales
    update_status()
    
    print("\n" + "="*60)
    print("INTERACTIVE GAUSSIAN FITTING")
    print("="*60)
    print("\nInstructions:")
    print("Press 'g' to start adding a gaussian")
    print("Then click 3 times:")
    print("  1. Center position (x=λ, y=depth)")
    print("  2. Left FWHM point (half height)")
    print("  3. Right FWHM point (half height)")
    print("Press 'a' to automatically fit gaussians")
    
    if params_dict:
        gaussian_count = sum(1 for key in params_dict.keys() 
                           if key.startswith('g') and '_center' in key)
        print(f"\nJSON parameters loaded: {gaussian_count} gaussian(s) found")
        if gaussians:
            print("Manual gaussians will be used for FREE parameters in JSON")
        else:
            print("Press 'a' to fit using JSON parameters")
    
    print("\nOther commands:")
    print("Press 'e' to toggle ERASE MODE (remove bad data points)")
    print("In erase mode: click ONE point to remove it (auto-exits after)")
    print("In erase mode: press 'r' to restore all removed points")
    print("Press 'r' in normal mode to remove last gaussian")
    print("Press 'c' to clear all gaussians")
    print("Press 'escape' to cancel current gaussian or exit erase mode")
    print("Press 'q' to finish and continue")
    print("="*60)
    
    plt.tight_layout()
    plt.show()
    
    # Devolver solo el resultado del ajuste
    return result

def save_fit_to_csv(filename, linename, hjd_value, vhelio, result):
    """
    Save Gaussian fit results from lmfit ModelResult to CSV file.
    Only saves specific columns as requested.
    
    Parameters:
    -----------
    filename : str
        File name
    linename : str
        Line name
    hjd_value : float
        HJD value from header
    result : lmfit.ModelResult
        Result of the fit
    """
    
    csv_filename = "fitted_"+linename+".csv"
    
    # Crear diccionario base con información general obligatoria
    data_dict = {
        'filename': [os.path.basename(filename)],
        'hjd': [f"{hjd_value:.6f}"],
        'vhelio': [f"{vhelio:10f}"],
        'chi2_red': [f"{result.redchi:.4f}"],
        'success': [result.success],
        }
    
    # Contar gaussianas en el ajuste
    gaussian_params = {}
    
    for param_name in result.params:
        if param_name.startswith('g') and '_' in param_name:
            # Extraer número de gaussiana y tipo de parámetro
            parts = param_name.split('_')
            if len(parts) == 2:
                try:
                    g_num = int(parts[0][1:])  # Número después de 'g'
                    param_type = parts[1]  # 'center', 'sigma', 'amplitude'
                    
                    if g_num not in gaussian_params:
                        gaussian_params[g_num] = {}
                    
                    param = result.params[param_name]
                    gaussian_params[g_num][param_type] = {
                        'value': param.value,
                        'error': param.stderr if param.stderr is not None else np.nan,
                        # 'error': param.stderr,
                        'vary': param.vary,
                        'min': param.min,
                        'max': param.max
                    }
                except ValueError:
                    continue
    
    n_gauss = len(gaussian_params)
    data_dict['n_gauss'] = [n_gauss]
    
    # Agregar parámetros de cada gaussiana (solo las columnas especificadas)
    for i in range(1, n_gauss + 1):
        if i in gaussian_params:
            gauss = gaussian_params[i]
            
            # Center (siempre presente)
            center_val = gauss.get('center', {}).get('value', np.nan)
            center_err = gauss.get('center', {}).get('error', np.nan)
            center_vary = gauss.get('center', {}).get('vary', True)
            
            data_dict[f'center{i}'] = [f"{center_val:.4f}"]
            
            if not np.isnan(center_err):
                data_dict[f'center{i}_err'] = [f"{center_err:.4f}"]
            else:
                data_dict[f'center{i}_err'] = [""]
            
            data_dict[f'center{i}_vary'] = [center_vary]
            
            # Sigma
            sigma_val = gauss.get('sigma', {}).get('value', np.nan)
            sigma_err = gauss.get('sigma', {}).get('error', np.nan)
            sigma_vary = gauss.get('sigma', {}).get('vary', True)
            
            data_dict[f'sigma{i}'] = [f"{sigma_val:.4f}"]
            
            if not np.isnan(sigma_err):
                data_dict[f'sigma{i}_err'] = [f"{sigma_err:.4f}"]
            else:
                data_dict[f'sigma{i}_err'] = [""]
            
            data_dict[f'sigma{i}_vary'] = [sigma_vary]
            
            # Convertir sigma a FWHM
            fwhm_val = sigma_val * 2.3548200 if not np.isnan(sigma_val) else np.nan
            fwhm_err = sigma_err * 2.3548200 if sigma_err and not np.isnan(sigma_err) else np.nan
            
            data_dict[f'fwhm{i}'] = [f"{fwhm_val:.3f}"]
            
            if not np.isnan(fwhm_err):
                data_dict[f'fwhm{i}_err'] = [f"{fwhm_err:.3f}"]
            else:
                data_dict[f'fwhm{i}_err'] = [""]
            
            # Amplitude
            amp_val = gauss.get('amplitude', {}).get('value', np.nan)
            amp_err = gauss.get('amplitude', {}).get('error', np.nan)
            amp_vary = gauss.get('amplitude', {}).get('vary', True)
            
            data_dict[f'amp{i}'] = [f"{amp_val:.4f}"]
            
            if not np.isnan(amp_err):
                data_dict[f'amp{i}_err'] = [f"{amp_err:.4f}"]
            else:
                data_dict[f'amp{i}_err'] = [""]
            
            data_dict[f'amp{i}_vary'] = [amp_vary]
            
            # Convertir amplitud a height y depth (solo si tenemos los valores necesarios)
            if not np.isnan(amp_val) and not np.isnan(sigma_val) and sigma_val != 0:
                height_val = amp_val / (sigma_val * np.sqrt(2 * np.pi))
                depth_val = 1 + height_val  # Para líneas de absorción
                data_dict[f'height{i}'] = [f"{height_val:.4f}"]
                data_dict[f'depth{i}'] = [f"{depth_val:.4f}"]
            else:
                data_dict[f'height{i}'] = [""]
                data_dict[f'depth{i}'] = [""]
    
    # Agregar parámetros de fondo
    if 'bkg_c' in result.params:
        bkg = result.params['bkg_c']
        data_dict['bkg_c'] = [f"{bkg.value:.4f}"]
        
        if bkg.stderr is not None:
            data_dict['bkg_c_err'] = [f"{bkg.stderr:.4f}"]
        else:
            data_dict['bkg_c_err'] = [""]
        
        data_dict['bkg_c_vary'] = [bkg.vary]
    else:
        # Buscar otros nombres de fondo
        bkg_found = False
        for param_name in result.params:
            if param_name.startswith('bkg_'):
                bkg = result.params[param_name]
                data_dict['bkg_c'] = [f"{bkg.value:.4f}"]
                
                if bkg.stderr is not None:
                    data_dict['bkg_c_err'] = [f"{bkg.stderr:.4f}"]
                else:
                    data_dict['bkg_c_err'] = [""]
                
                data_dict['bkg_c_vary'] = [bkg.vary]
                bkg_found = True
                break
        
        if not bkg_found:
            data_dict['bkg_c'] = [""]
            data_dict['bkg_c_err'] = [""]
            data_dict['bkg_c_vary'] = [""]
    
    # Definir el orden específico de columnas que queremos
    base_columns = ['filename', 'hjd', 'vhelio', 'chi2_red', 'success', 'n_gauss']
    
    # Generar columnas para cada gaussiana
    gaussian_columns = []
    for i in range(1, n_gauss + 1):
        gaussian_columns.extend([
            f'center{i}', f'center{i}_err', f'center{i}_vary',
            f'sigma{i}', f'sigma{i}_err', f'sigma{i}_vary',
            f'fwhm{i}', f'fwhm{i}_err',
            f'amp{i}', f'amp{i}_err', f'amp{i}_vary',
            f'height{i}', f'depth{i}',
        ])
    
    # Columnas de fondo
    background_columns = ['bkg_c', 'bkg_c_err', 'bkg_c_vary']
    
    # Orden final de columnas
    all_columns = base_columns + gaussian_columns + background_columns
    
    # Asegurarse de que todas las columnas existan en el diccionario
    for col in all_columns:
        if col not in data_dict:
            data_dict[col] = [""]
    
    # Crear DataFrame con el orden específico
    df_new = pd.DataFrame({col: data_dict[col] for col in all_columns})
    
    # Verificar si el archivo existe
    if os.path.exists(csv_filename):
        try:
            df_existing = pd.read_csv(csv_filename)
            
            # Verificar si el archivo tiene las mismas columnas
            existing_cols = set(df_existing.columns)
            new_cols = set(df_new.columns)
            
            if existing_cols != new_cols:
                print(f"Warning: Columns don't match between existing file and new data.")
                print(f"Existing columns ({len(existing_cols)}): {sorted(existing_cols)}")
                print(f"New columns ({len(new_cols)}): {sorted(new_cols)}")
                
                # Preguntar qué hacer
                print("\nOptions:")
                print("1. Append anyway (may create NaN columns)")
                print("2. Create new file with current columns")
                print("3. Cancel save")
                
                choice = input("Enter choice (1-3): ").strip()
                
                if choice == "1":
                    # Añadir columnas faltantes a ambos DataFrames
                    all_cols = sorted(existing_cols.union(new_cols))
                    
                    # Asegurar que ambos DataFrames tengan todas las columnas
                    for col in all_cols:
                        if col not in df_existing.columns:
                            df_existing[col] = ""
                        if col not in df_new.columns:
                            df_new[col] = ""
                    
                    # Reordenar columnas
                    df_existing = df_existing[all_cols]
                    df_new = df_new[all_cols]
                    
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(csv_filename, index=False)
                    print(f"✓ Appended to existing {csv_filename} (with column adjustment)")
                    
                elif choice == "2":
                    # Crear backup y nuevo archivo
                    import shutil
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    backup_name = f"{csv_filename}.backup_{timestamp}"
                    shutil.copy2(csv_filename, backup_name)
                    print(f"Created backup: {backup_name}")
                    
                    df_new.to_csv(csv_filename, index=False)
                    print(f"✓ Created new {csv_filename} with current columns")
                    
                else:
                    print("✗ Save cancelled")
                    return
            else:
                # Mismas columnas, simplemente concatenar
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(csv_filename, index=False)
                print(f"✓ Added fit to existing {csv_filename}")
                
        except Exception as e:
            print(f"Error reading existing CSV: {e}")
            print(f"Creating new file...")
            df_new.to_csv(csv_filename, index=False)
            print(f"✓ Created new {csv_filename}")
    else:
        # Crear nuevo archivo
        df_new.to_csv(csv_filename, index=False)
        print(f"✓ Created new {csv_filename}")
    
    # Mostrar resumen
    print(f"\nSummary saved:")
    print(f"  File: {filename}")
    print(f"  HJD: {hjd_value:.6f}")
    print(f"  Gaussians: {n_gauss}")
    print(f"  χ²/ν: {result.redchi:.4e}")
    print(f"  Success: {result.success}")
    print(f"  CSV file: {os.path.abspath(csv_filename)}")
    
    return csv_filename

def main():
    """Main function"""
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Interactive spectrum analyzer')
    parser.add_argument('filename', help='FITS spectrum file')
    parser.add_argument('center', nargs='?', type=float, 
                       help='Central wavelength for zoom (optional)')
    parser.add_argument('width', nargs='?', type=float, default=40.0,
                       help='Width around center (default: 40 Å)')
    parser.add_argument('--params', type=str, 
                       help='JSON file with fit parameters')
    
    args = parser.parse_args()
    
    filename = args.filename
    
    # Load spectrum
    header, wavelength, flux = load_spectrum(filename)
    
    if wavelength is None or flux is None or header is None:
        sys.exit(1)
    
    # Cargar parámetros si se especificó
    params_dict = {}
    if args.params:
        try:
            import json
            with open(args.params, 'r') as f:
                params_dict = json.load(f)
            print(f"\n✓ Loaded parameters from {args.params}")
        except Exception as e:
            print(f"\n❌ Error loading parameters: {e}")
        
    # Recortar si se especificó centro
    if args.center:
        central_wavelength = args.center
        width = args.width
        region = [[central_wavelength-width, central_wavelength+width]]
        mask = mask_generator(wavelength, region)
        wavelength, flux = wavelength[mask], flux[mask]
        print(f"\n✓ Zooming to {central_wavelength}±{width} Å")
    
    print("\n" + "="*60)
    print("CONTINUUM FITTING - Press 'q' in plot window to close")
    print("="*60)
    flux, regions, poly_order = interactive_continuum_fitting(
        wavelength, flux, filename
    )
    print(f"\n✓ Normalized with polynomial order {poly_order}")
    print(f"✓ Used {len(regions)} continuum regions")
    
    # Ajuste de líneas
    print("\n" + "="*60)
    print("LINE FITTING - Interactive mode")
    print("="*60)
    result = interactive_lines_fitting(wavelength, flux, filename, params_dict)
    
    # Mostrar resumen final
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    if result is not None:
        # Extraer información del resultado
        n_gauss = sum(1 for key in result.params.keys() 
                     if key.startswith('g') and '_center' in key)
        
        print(f"\nFitted {n_gauss} gaussian(s):")
        
        for i in range(1, n_gauss + 1):
            prefix = f'g{i}_'
            if f'{prefix}center' in result.params and f'{prefix}sigma' in result.params:
                center = result.params[f'{prefix}center'].value
                sigma = result.params[f'{prefix}sigma'].value
                amplitude = result.params[f'{prefix}amplitude'].value
                
                # Convertir a valores más útiles
                fwhm = sigma * 2.3548200
                height = amplitude / (sigma * np.sqrt(2 * np.pi))
                depth = 1 + height
                ew = np.abs(amplitude * fwhm / (sigma * np.sqrt(2 * np.pi)))
                
                print(f"\nGaussian {i}:")
                print(f"  Center: {center:.4f} Å")
                print(f"  Depth: {depth:.4f}")
                print(f"  FWHM: {fwhm:.3f} Å")
                print(f"  Equivalent width: {ew:.3f} Å")
        
        print(f"\nFit statistics:")
        print(f"  Reduced χ²: {result.redchi:.4e}")
        print(f"  AIC: {result.aic:.2f}")
        print(f"  BIC: {result.bic:.2f}")
        print(f"  Success: {result.success}")
    
        # Para armar tabla de medidas
        vhelio = 0.0
        if 'VHELIO' in header:
            vhelio = header['VHELIO']
        
        # Guardar resultados
        linename = str(int(args.center))
        hjd_keys = ['HJD', 'HDJ', 'JD', 'MJD','MJD-OBS', 'OHP DRS BJD']
        hjd_value = None
        for key in hjd_keys:
            if key in header:
                hjd_value = header[key]
                break
                        
        if hjd_value:
            save = input("\nSave fit to CSV? [Y/n]: ").strip().lower()
            if save in ['', 'yes', 'y']:
                save_fit_to_csv(filename, linename, hjd_value, vhelio, result)
        else:
            print("\nNo HJD or similar not found in header.")
            
    else:
        print("\nNo fit was performed.")
    
    print("\n✓ Program finished.")
    print("="*60)

if __name__ == "__main__":
    main()