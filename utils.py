#!/home/tansin/miniconda3/envs/spec/bin/python
# -*- coding: utf-8 -*-
"""
utils.py — Utilidades para lectura, procesamiento y ajuste de espectros estelares.

Funciones principales
---------------------
read_fits_simple : Lee archivos FITS 1D de instrumentos soportados (REOSC, FEROS, HARPS, SOPHIE, etc.).
read_fits_multi  : Lee archivos FITS en formato MULTISPEC (múltiples órdenes).
mask_generator   : Genera máscaras booleanas a partir de rangos de longitud de onda.
fit_cont         : Ajusta el continuo con polinomios de Legendre o Chebyshev.
fit_cont_sigma   : Igual que fit_cont con opción de sigma clipping.
gaussian         : Evalúa una función gaussiana parametrizada por centro, altura y FWHM.
fit_lines        : Ajusta una o varias gaussianas a líneas espectrales usando lmfit.
resampler        : Remuestrea un espectro a un nuevo eje espectral conservando el flujo.
concat_flux      : Combina espectros superpuestos promediando e interpolando gaps.
save_concat_fits : Guarda un espectro concatenado como FITS 1D compatible con IRAF.
find_closest_line: Busca la línea espectral del catálogo más cercana a una longitud de onda.

Clases
------
spectrum : Contenedor ligero de espectro (wavelength, flux, header) con método de guardado.
"""
import os

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.io.votable import parse as parse_votable
import logging

from astropy.stats import sigma_clip
import astropy.units as u
from astropy.modeling.polynomial import Legendre1D, Chebyshev1D
from astropy.modeling.fitting import LinearLSQFitter, FittingWithOutlierRemoval
from specutils import Spectrum
from specutils.manipulation import FluxConservingResampler
import warnings
from astropy.io.fits.verify import VerifyWarning

warnings.simplefilter('ignore', category=VerifyWarning)
logger = logging.getLogger(__name__)

def read_fits_simple(file_name):
    """
    Lee un archivo FITS 1D y extrae cabecera, longitudes de onda y flujo.

    Soporta los instrumentos: REOSC, FEROS, Echelle/SITe2K-1, HARPS y SOPHIE.
    Para espectros MULTISPEC usar :func:`read_fits_multi`.

    Parameters
    ----------
    file_name : str
        Ruta al archivo FITS a leer.

    Returns
    -------
    header : astropy.io.fits.Header
        Cabecera FITS de la extensión primaria.
    wavelength : numpy.ndarray
        Array 1D de longitudes de onda en Ångströms.
    flux : numpy.ndarray
        Array de flujo asociado al eje espectral.

    Raises
    ------
    ValueError
        Si el espectro es de tipo MULTISPEC (usar :func:`read_fits_multi` en su lugar).

    Notes
    -----
    Si el keyword ``DC-FLAG=1`` está presente, la escala de longitudes de onda
    se trata como logarítmica y se aplica la conversión ``10**wavelength``.

    Para agregar un nuevo instrumento, añadir una entrada en ``_INSTRUMENTS``:
    - ``'wcs'``            : longitud de onda desde keywords WCS del header.
    - ``'bintable'``       : longitud de onda y flujo desde hdul[1].data[0][0/1].
    - ``'wcs_or_bintable'``: intenta WCS primero; si no hay CTYPE1, usa bintable.
    El valor puede ser el nombre exacto o un fragmento (se usa ``in`` para comparar).
    """
    # Registro de instrumentos: fragmento_nombre → estrategia
    # Para agregar uno nuevo, añadir aquí una línea.
    _INSTRUMENTS = {
        'REOSC'             : 'wcs',
        'Reosc'             : 'wcs',
        'Echelle/SITe2K-1'  : 'wcs',
        'SOPHIE'            : 'wcs',
        'FEROS'             : 'wcs_or_bintable',
        'HARPS'             : 'bintable',
        'HERMES'            : 'bintable',
        'SES V4.0'          : 'bintable',
        'CAFE 2.2'          : 'bintable',
        'HARPN'             : 'bintable',
        'UVES'              : 'bintable',
        'GRACES'            : 'bintable',
        'FIES'              : 'bintable',
    }
    _WCS_CTYPES = {'LINEAR', 'wavelength', 'WAVELENGTH', 'pixel', 'AWAV'}

    def _wave_from_header(hdr):
        """Calcula el eje de longitudes de onda a partir de keywords WCS lineales."""
        start, step, num, ref = (hdr['CRVAL1'], hdr['CDELT1'],
                                 hdr['NAXIS1'], hdr['CRPIX1'])
        wave = (np.arange(num, dtype=float) + 1 - ref) * step + start
        if hdr.get('DC-FLAG') == 1:
            wave = 10.0 ** wave
        return wave

    def _read_bintable(hdul):
        data = hdul[0].data
        if data is not None:
            return data[0], data[1]
        else:
            data = hdul[1].data
            return data[0][0], data[0][1]
     
        
    with fits.open(file_name) as hdul:
        hdul[0].verify('fix')
        header = hdul[0].header
        
        if header.get('CTYPE1') == 'MULTISPE':
            raise ValueError("This spectrum is MULTISPEC")

        if header.get('PROCSPEC') == 'spec.py':
            return header, _wave_from_header(header), hdul[0].data

        # Buscar INSTRUME en todas las extensiones si no está en la primaria
        if 'INSTRUME' not in header:
            for ext in hdul[1:]:
                if 'INSTRUME' in ext.header:
                    header = ext.header
                    break
                
                # PARA ESPECTROS DE UNWIND
                if 'COMP' in header:
                    wave, flux = _read_bintable(hdul)
                    return header, wave, flux
                
            else:
                print('No se encontró keyword INSTRUME en header')
                return            

        instrument = header['INSTRUME']

        strategy = next(
            (s for key, s in _INSTRUMENTS.items() if key in instrument),
            None
        )

        if strategy is None:
            print(f'{instrument} no está implementado en esta función')
            return

        ctype = header.get('CTYPE1')

        if strategy == 'bintable':
            wave, flux = _read_bintable(hdul)
            return header, wave, flux

        if strategy in ('wcs', 'wcs_or_bintable'):
            if ctype in _WCS_CTYPES:
                return header, _wave_from_header(header), hdul[0].data
            elif strategy == 'wcs_or_bintable':
                wave, flux = _read_bintable(hdul)
                return header, wave, flux
            else:
                raise ValueError('CTYPE1 no coincide con las opciones soportadas')


def read_votable(file_name, spectral_col='spectral', flux_col='flux'):
    """
    Lee un espectro desde un archivo VOTable con columnas de eje espectral y flujo.

    Parameters
    ----------
    file_name : str
        Ruta al archivo VOTable.
    spectral_col : str
        Nombre de la columna de longitudes de onda (default 'spectral').
    flux_col : str
        Nombre de la columna de flujo (default 'flux').

    Returns
    -------
    header : astropy.io.fits.Header
        Cabecera FITS minima (sin claves WCS ni de tiempo).
    wavelength : numpy.ndarray
        Array 1D de longitudes de onda en Angstroms.
    flux : numpy.ndarray
        Array 1D de flujo.

    Raises
    ------
    ValueError
        Si no se encuentran las columnas esperadas en la tabla.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        vot = parse_votable(file_name)

    table = vot.get_first_table()
    field_names = [f.name for f in table.fields]

    if spectral_col not in field_names:
        raise ValueError(
            f"Columna '{spectral_col}' no encontrada. Columnas disponibles: {field_names}"
        )
    if flux_col not in field_names:
        raise ValueError(
            f"Columna '{flux_col}' no encontrada. Columnas disponibles: {field_names}"
        )

    arr = table.array
    wavelength = np.asarray(arr[spectral_col], dtype=float)
    flux = np.asarray(arr[flux_col], dtype=float)

    # Eliminar NaN/masked
    mask = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength = wavelength[mask]
    flux = flux[mask]

    # Ordenar por longitud de onda (las VOTable no siempre vienen ordenadas)
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    flux = flux[order]

    # Cabecera FITS minima para compatibilidad con el resto del pipeline
    header = fits.Header()
    header['SIMPLE'] = True
    header['NAXIS'] = 1
    header['NAXIS1'] = len(wavelength)
    header['ORIGIN'] = 'VOTable'
    header['VOTFILE'] = os.path.basename(file_name)

    # Extraer metadatos del VOTable; mapear ssa_dateObs -> MJD-OBS explicitamente
    for param in list(vot.params) + list(table.params):
        name_lower = param.name.lower()
        if name_lower == 'ssa_dateobs' and param.value not in ('', None):
            try:
                header['MJD-OBS'] = float(param.value)
            except (TypeError, ValueError):
                pass
            continue
        key = param.name.upper()[:8]
        try:
            header[key] = param.value
        except Exception:
            pass

    return header, wavelength, flux


def read_fits_multi(file_name, extension=None):
    """
    Lee archivos FITS con datos MULTISPEC de instrumentos específicos.
    
    Parámetros:
    -----------
    file_name : str
        Ruta del archivo FITS a leer
    extension : int, optional
        Número de extensión a leer. Si es None y hay múltiples extensiones,
        se lanzará una excepción.
    
    Retorna:
    --------
    tuple (header, wavelen, data)
        header : Header de la extensión
        wavelen : Ejes de dispersión de los flujos (norders × nwave)
        data : Flujos de la imagen
    
    Excepciones:
    ------------
    FileNotFoundError: Si el archivo no existe
    ValueError: Si los parámetros son inválidos
    KeyError: Si faltan keywords esenciales
    Exception: Para errores específicos del formato MULTISPEC
    """
    
    # Validación básica del nombre de archivo
    if not isinstance(file_name, str):
        raise TypeError(f"file_name debe ser string, no {type(file_name).__name__}")
    
    if not file_name.lower().endswith('.fits'):
        logger.warning(f"El archivo {file_name} no tiene extensión .fits")
    
    try:
        with fits.open(file_name) as hdul:
            # Manejo de extensión
            if extension is None:
                if len(hdul) > 1:
                    error_msg = (
                        f"El archivo tiene {len(hdul)} extensiones. "
                        "Debe especificar cuál leer usando el parámetro 'extension'."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                else:
                    extension = 0
            
            # Validación de extensión
            if not (0 <= extension < len(hdul)):
                raise IndexError(
                    f"Extensión {extension} no válida. "
                    f"El archivo tiene {len(hdul)} extensiones (0-{len(hdul)-1})"
                )
            
            # Obtener datos y header
            hdu = hdul[extension]
            hdu.verify('fix')  # Corregir problemas de FITS si es necesario
            header = hdu.header
            data = hdu.data
            
            # Verificar que hay datos
            if data is None:
                raise ValueError(f"Extensión {extension} no contiene datos")
            
            # Determinar dimensiones
            data_shape = data.shape
            nbands = data_shape[0] if len(data_shape) == 3 else 1
            norders = data_shape[0] if nbands == 1 else data_shape[1]
            nwave = data_shape[-1]
            
            # Log información básica
            logger.info(f"Leyendo archivo: {file_name}")
            logger.info(f"Extensión: {extension}")
            logger.info(f"Forma de datos: {data_shape}")
            logger.info(f"Dimensiones: {nbands} bandas, {norders} órdenes, {nwave} puntos de longitud de onda")
            
            try:
                # Verificar instrumento
                if 'INSTRUME' not in header:
                    raise KeyError("Keyword 'INSTRUME' no encontrada en el header")
                
                instrument = header['INSTRUME'].strip()
                logger.info(f"Instrumento: {instrument}")
                
                # Lista de instrumentos soportados
                REOSC_VARIANTS = ['REOSC', 'Reosc']
                ECHELLE_INSTRUMENT = 'Echelle/SITe2K-1'
                
                # Verificar si es instrumento REOSC
                is_reosc = any(variant in instrument for variant in REOSC_VARIANTS)
                
                if not (is_reosc or instrument == ECHELLE_INSTRUMENT):
                    raise ValueError(f"Instrumento '{instrument}' no implementado en esta función")
                
                # Verificar formato MULTISPEC
                if 'CTYPE1' not in header:
                    raise KeyError("Keyword 'CTYPE1' no encontrada en el header")
                
                if header['CTYPE1'] != 'MULTISPE':
                    raise ValueError(
                        f"CTYPE1='{header.get('CTYPE1')}', se esperaba 'MULTISPE'"
                    )
                
                # Procesar keywords WAT2*
                if not any(key.startswith('WAT2') for key in header):
                    raise KeyError("No se encontraron keywords WAT2* en el header")
                
                # Construir string WAT2
                wat2_string = ''
                wat2_keys = sorted(key for key in header if key.startswith('WAT2'))
                
                for key in wat2_keys:
                    value = str(header[key])
                    # Padding específico para REOSC
                    if len(value) != 68:
                        wat2_string += value + ' '
                    else:
                        wat2_string += value
                
                # Parsear WAT2
                wat2_parts = wat2_string.split('"')
                wat2_parts = [part for part in wat2_parts if 'spec' not in part.lower()]
                
                if not wat2_parts:
                    raise ValueError("No se pudieron extraer parámetros de WAT2")
                
                # Remover último elemento si está vacío
                if wat2_parts[-1].strip() == '':
                    wat2_parts = wat2_parts[:-1]
                
                # Validar número de órdenes
                if len(wat2_parts) < norders:
                    raise ValueError(
                        f"Número insuficiente de parámetros WAT2. "
                        f"Esperados {norders}, encontrados {len(wat2_parts)}"
                    )
                
                # Procesar parámetros WAT2
                wparms = np.zeros((norders, 9), dtype=float)
                
                for i in range(norders):
                    try:
                        # Convertir string a array de floats
                        w1 = np.fromstring(wat2_parts[i], sep=' ', dtype=float)
                        
                        if len(w1) < 9:
                            raise ValueError(
                                f"Parámetros insuficientes en orden {i+1}. "
                                f"Esperados 9, encontrados {len(w1)}"
                            )
                        
                        wparms[i, :] = w1[:9]
                        
                        # Verificar calibración de longitud de onda
                        if w1[2] == -1:
                            raise ValueError(
                                f"Spectrum {i + 1} has no wavelength calibration (type={w1[2]})"
                            )
                            
                    except ValueError as e:
                        logger.error(f"Error procesando orden {i+1}: {e}")
                        raise
                
                # Generar ejes de dispersión
                wavelen = np.zeros((norders, nwave), dtype=float)
                
                for i in range(norders):
                    dtype = wparms[i, 2]  # Tipo de dispersión
                    wstart = wparms[i, 3]  # λ inicial
                    dw = wparms[i, 4]      # Δλ
                    z = wparms[i, 6]       # Redshift
                    
                    if dtype == 0 or dtype == 1:
                        # Dispersión lineal o logarítmica
                        wavelen[i, :] = np.arange(nwave, dtype=float) * dw + wstart
                        
                        if dtype == 1:
                            wavelen[i, :] = 10.0 ** wavelen[i, :]
                            logger.info(f"Orden {i+1}: Dispersión lineal en log(longitud de onda)")
                        else:
                            logger.info(f"Orden {i+1}: Dispersión lineal")
                        
                        # Aplicar corrección de redshift
                        wavelen[i, :] *= 1.0 + z
                        if z != 0:
                            logger.info(f"Orden {i+1}: Corrigiendo redshift z={z}")
                    else:
                        raise ValueError(f"Tipo de dispersión {dtype} no soportado")
                
                # Manejar datos multibanda
                if nbands > 1:
                    logger.info(f"Datos multibanda detectados, usando la primera banda")
                    data = data[0]
                
                return header, wavelen, data
                
            except (KeyError, ValueError) as e:
                logger.error(f"Error procesando archivo: {e}")
                raise
            except Exception as e:
                logger.error(f"Error inesperado: {e}", exc_info=True)
                raise
            
    except FileNotFoundError:
        error_msg = f"Archivo no encontrado: {file_name}"
        logger.error(error_msg)
        raise
    except OSError as e:
        error_msg = f"Error de E/S al leer {file_name}: {e}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Error inesperado al procesar {file_name}: {e}"
        logger.error(error_msg, exc_info=True)
        raise
            
def mask_generator(wavelength, ranges, inclusive=True):
    """
    Genera una máscara booleana con control sobre inclusividad de extremos.

    Parameters
    ----------
    wavelength : array-like
        Array de longitudes de onda (o cualquier valor numérico) a enmascarar.
    ranges : list of list/tuple
        Lista de pares ``[inicio, fin]`` que definen los rangos a seleccionar.
    inclusive : bool or tuple of bool, optional
        Controla si los extremos de cada rango se incluyen:

        * ``True``  (defecto): ambos extremos inclusive (``>=`` y ``<=``).
        * ``False``: ambos extremos exclusivos (``>`` y ``<``).
        * ``(inicio_inc, fin_inc)``: control independiente por extremo.

    Returns
    -------
    numpy.ndarray of bool
        Array booleano con la misma forma que ``wavelength``.
        Es ``True`` en los elementos que caen dentro de alguno de los rangos.

    Examples
    --------
    >>> import numpy as np
    >>> wave = np.arange(4000, 4010)
    >>> mask_generator(wave, [[4002, 4005]])
    array([False, False,  True,  True,  True,  True, False, False, False, False])
    """
    wavelength = np.asarray(wavelength)
    mascara = np.zeros_like(wavelength, dtype=bool)
    
    # Determinar operadores de comparación
    if inclusive is True:
        op_inicio = np.greater_equal  # >=
        op_fin = np.less_equal        # <=
    elif inclusive is False:
        op_inicio = np.greater        # >
        op_fin = np.less              # <
    else:
        # inclusive es una tupla (inicio_inclusive, fin_inclusive)
        inicio_inc, fin_inc = inclusive
        op_inicio = np.greater_equal if inicio_inc else np.greater
        op_fin = np.less_equal if fin_inc else np.less
    
    for rango in ranges:
        inicio, fin = rango
        mascara_rango = op_inicio(wavelength, inicio) & op_fin(wavelength, fin)
        mascara = mascara | mascara_rango
    
    return mascara

def fit_cont(wavelength, flux, model=None, order=2, exclude=None):
    """
    Ejecuta un ajuste de curva de contención utilizando diferentes modelos.

    Args:
        wavelength (array): Longitudes de onda.
        flux (array): Flujo de luz.
        model (str, optional): Tipo de modelo a utilizar. Opciones [legendre, chebyshev]. Defaults to None.
        order (int, optional): Orden del polinomio. Defaults to 2.
        exclude (list of tuples, optional): Longitudes de onda y valores para excluir del ajuste. Defaults to None.

    Returns:
        tuple: Resultado del ajuste y puntos excluidos.

    Raises:
        ValueError: Si se especifica un modelo pero no se proporcionan los parámetros necesarios.
    """
    # Verificar order
    if not isinstance(order, int) or order < 0:
        raise ValueError("El orden debe ser un número entero positivo")
    
    # Verificar si se especifica un modelo
    if model is None:
        raise ValueError("Debe especificar modelo de ajuste. Opciones [legendre, chebyshev]")
    
    # Verificar el tipo del modelo
    if model == 'legendre':
        # Definir modelo
        model = Legendre1D(order)
    elif model == 'chebyshev':
        model = Chebyshev1D(order)
    else:
        raise ValueError("Modelo no válido. Opciones [legendre, chebyshev]")
        
    # Crear una máscara para excluir puntos si se especifican valores de exclusión
    exclude_mask = np.ones_like(wavelength, dtype=bool)
    if exclude is not None:
        for exc in exclude: 
            exclude_mask *= (wavelength < exc[0]) | (wavelength > exc[1])
            
    # Crear un objeto para ajuste lineal con la opción de calcular incertidumbre
    fitter = LinearLSQFitter(calc_uncertainties=True)
    
    # Ejecutar el ajuste utilizando el modelo y los puntos excluidos
    cont_fit = fitter(model, wavelength[exclude_mask], flux[exclude_mask])
            
    # Obtener los puntos excluidos
    excl_pts = [wavelength[~exclude_mask], flux[~exclude_mask]]
    
    return cont_fit, excl_pts

def fit_cont_sigma(wavelength, flux, model=None, order=2, exclude=None,
                    use_sigma_clip=False, sigma_lower=None, sigma_upper=None, niter=5):
    """
    Ejecuta un ajuste de curva de continuo utilizando diferentes modelos,
    con opción de sigma clipping para eliminar outliers.

    Args:
        wavelength (array): Longitudes de onda.
        flux (array): Flujo de luz.
        model (str, optional): Tipo de modelo a utilizar. 
            Opciones ['legendre', 'chebyshev']. Defaults to None.
        order (int, optional): Orden del polinomio. Defaults to 2.
        exclude (list of tuples, optional): 
            Longitudes de onda y valores para excluir del ajuste. 
            Cada tupla debe contener (min_wavelength, max_wavelength).
            Defaults to None.
        use_sigma_clip (bool, optional):
            Si es True, aplica sigma clipping para eliminar outliers.
            Defaults to False.
        sigma_lower (float, optional): 
            Número de sigmas por debajo para rechazar puntos.
            Defaults to None (se usará 3 si use_sigma_clip=True).
        sigma_upper (float, optional): 
            Número de sigmas por encima para rechazar puntos.
            Defaults to None (se usará 3 si use_sigma_clip=True).
        niter (int, optional): 
            Número de iteraciones para el sigma clipping.
            Defaults to 10.

    Returns:
        tuple: 
            - Si use_sigma_clip=False: (cont_fit, excl_pts)
            - Si use_sigma_clip=True: (cont_fit, reject, excl_pts)
            donde:
                cont_fit: Modelo ajustado
                reject: [wavelength_rejected, flux_rejected] - puntos rechazados por sigma clipping
                excl_pts: [wavelength_excluded, flux_excluded] - puntos excluidos manualmente

    Raises:
        ValueError: Si se especifica un modelo pero no se proporcionan los parámetros necesarios.
    """
    # Verificar order
    if not isinstance(order, int) or order < 0:
        raise ValueError("El orden debe ser un número entero positivo")
    
    # Verificar si se especifica un modelo
    if model is None:
        raise ValueError("Debe especificar modelo de ajuste. Opciones ['legendre', 'chebyshev']")
    
    # Verificar el tipo del modelo
    if model == 'legendre':
        model_obj = Legendre1D(order)
    elif model == 'chebyshev':
        model_obj = Chebyshev1D(order)
    else:
        raise ValueError("Modelo no válido. Opciones ['legendre', 'chebyshev']")
    
    # Crear una máscara para excluir puntos si se especifican valores de exclusión
    exclude_mask = np.ones_like(wavelength, dtype=bool)
    if exclude is not None:
        for exc in exclude:
            if len(exc) != 2:
                raise ValueError("Cada elemento de exclude debe ser una tupla de 2 valores (min, max)")
            exclude_mask *= (wavelength < exc[0]) | (wavelength > exc[1])
    
    # Obtener los puntos excluidos manualmente
    excl_pts = [wavelength[~exclude_mask], flux[~exclude_mask]] if np.any(~exclude_mask) else [[], []]
    
    # Preparar datos para el ajuste (excluyendo las regiones especificadas)
    fit_wavelength = wavelength[exclude_mask]
    fit_flux = flux[exclude_mask]
    
    if use_sigma_clip:
        # Configurar valores por defecto para sigma clipping
        if sigma_lower is None:
            sigma_lower = 5
        if sigma_upper is None:
            sigma_upper = 5
            
        # Crear el fitter con sigma clipping
        fitter = FittingWithOutlierRemoval(
            LinearLSQFitter(calc_uncertainties=True), 
            sigma_clip, 
            niter=niter,
            sigma_lower=sigma_lower, 
            sigma_upper=sigma_upper
        )
        
        # Ejecutar el ajuste con sigma clipping
        cont_fit, mask = fitter(model_obj, fit_wavelength, fit_flux)
        
        # Obtener los puntos rechazados por sigma clipping
        # mask es True para puntos BUENOS (no rechazados), False para puntos rechazados
        reject_mask = mask
        reject = [fit_wavelength[reject_mask], fit_flux[reject_mask]]
                
        return cont_fit, reject, excl_pts
    
    else:
        # Ajuste normal sin sigma clipping
        fitter = LinearLSQFitter(calc_uncertainties=True)
        cont_fit = fitter(model_obj, fit_wavelength, fit_flux)
        
        return cont_fit, excl_pts
    
def gaussian(x, center, height, fwhm):
    """
    Función gaussiana definida por centro, altura y FWHM.
    
    Parámetros:
    -----------
    x : array_like
        Valores del eje x donde evaluar la gaussiana
    center : float
        Posición del centro de la gaussiana
    height : float
        Altura máxima de la gaussiana
    fwhm : float
        Ancho a media altura (Full Width at Half Maximum)
    
    Retorna:
    --------
    y : array_like
        Valores de la gaussiana evaluada en x
    """
    # Convertir FWHM a desviación estándar (sigma)
    # FWHM = 2√(2ln2) * sigma ≈ 2.35482 * sigma
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Calcular la gaussiana
    exponent = -((x - center) ** 2) / (2 * sigma ** 2)
    y = 1 + height * np.exp(exponent) # Espectro normalizado
    
    return y

def fit_lines(wavelength, flux, params=None):
    """
    Ajusta múltiples gaussianas a un espectro con parámetros en formato lmfit.
    
    Parameters:
    -----------
    wavelength : array
        Longitudes de onda
    flux : array
        Valores de flujo
    params : dict, optional
        Diccionario con parámetros en formato lmfit.
    
    Returns:
    --------
    result : lmfit.ModelResult
        Resultado del ajuste
    """
    from lmfit.models import GaussianModel, ConstantModel

    # Parámetros por defecto si no se proporcionan
    if params is None:
        params = {}
    
    # Crear lista de gaussianas basada en los parámetros
    gaussian_params = {}
    for key in params.keys():
        if key.startswith('g') and '_' in key:
            # Extraer número de gaussiana y tipo de parámetro
            parts = key.split('_')
            if len(parts) == 2 and parts[0].startswith('g'):
                try:
                    g_num = int(parts[0][1:])  # Extraer número después de 'g'
                    param_type = parts[1]
                    
                    if g_num not in gaussian_params:
                        gaussian_params[g_num] = {}
                    
                    gaussian_params[g_num][param_type] = params[key]
                except ValueError:
                    pass
    
    # Si no se encontraron parámetros de gaussianas, crear una por defecto
    if not gaussian_params:
        n_gauss = 1
        default_center = wavelength[0] + (wavelength[-1] - wavelength[0]) / 2
        default_sigma = 0.85  # FWHM ≈ 2.0
        default_amp = 0.1 * default_sigma * np.sqrt(2 * np.pi)
        
        params['g1_center'] = {'value': default_center, 'min': wavelength[0], 'max': wavelength[-1]}
        params['g1_sigma'] = {'value': default_sigma, 'min': 0.001}
        params['g1_amplitude'] = {'value': default_amp}
        gaussian_params[1] = {'center': {'value': default_center}, 'sigma': {'value': default_sigma}}
    else:
        n_gauss = max(gaussian_params.keys())
    
    # Determinar número de gaussianas basado en los parámetros
    gaussian_indices = sorted(gaussian_params.keys())
    n_gauss = len(gaussian_indices)
    
    # Crear modelo base (fondo constante)
    model = ConstantModel(prefix='bkg_')
    
    # Añadir gaussianas dinámicamente
    for i in range(1, n_gauss + 1):
        prefix = f'g{i}_'
        gaussian = GaussianModel(prefix=prefix)
        model += gaussian
        
        if i == 1:
            pars = model.make_params()
        else:
            pars.update(gaussian.make_params())
    
    # Configurar todos los parámetros del modelo
    for param_name in pars:
        # Primero aplicar cualquier configuración específica del parámetro
        if param_name in params:
            param_config = params[param_name]
            
            # Configurar valor si está especificado
            if 'value' in param_config:
                pars[param_name].set(value=param_config['value'])
            
            # Configurar variabilidad
            if 'vary' in param_config:
                pars[param_name].set(vary=param_config['vary'])
            
            # Configurar límites
            if 'min' in param_config:
                pars[param_name].set(min=param_config['min'])
            if 'max' in param_config:
                pars[param_name].set(max=param_config['max'])
            
            # Configurar expresión si existe
            if 'expr' in param_config:
                pars[param_name].set(expr=param_config['expr'])
        
        # Luego aplicar restricciones globales si existen
        param_parts = param_name.split('_')
        if len(param_parts) >= 2:
            param_type = param_parts[-1]  # 'center', 'sigma', 'amplitude', etc.
            
            # Restricción para todos los parámetros de un tipo
            if f'all_{param_type}' in params:
                global_config = params[f'all_{param_type}']
                
                if 'vary' in global_config:
                    pars[param_name].set(vary=global_config['vary'])
                if 'min' in global_config:
                    pars[param_name].set(min=global_config['min'])
                if 'max' in global_config:
                    pars[param_name].set(max=global_config['max'])
    
    # Configurar parámetros que no tienen valor inicial
    for i in range(1, n_gauss + 1):
        prefix = f'g{i}_'
        
        # Configurar valores por defecto si no están especificados
        if f'{prefix}center' not in params:
            # Establecer rango para centros si no está especificado
            # Verificar si ya tiene límites configurados
            if pars[f'{prefix}center'].min is None:
                pars[f'{prefix}center'].set(min=wavelength[0])
            if pars[f'{prefix}center'].max is None:
                pars[f'{prefix}center'].set(max=wavelength[-1])
        
        # Establecer sigma mínimo si no está especificado
        if f'{prefix}sigma' in pars and pars[f'{prefix}sigma'].min is None:
            pars[f'{prefix}sigma'].set(min=0.001)
    
    # Configurar fondo si no está especificado
    if 'bkg_c' not in params:
        pars['bkg_c'].set(value=1.0, vary=True)
    
    # Opciones de ajuste
    fit_options = params.get('fit_options', {})
    default_fit_options = {
        'method': 'leastsq',
        'max_nfev': 200,
        'nan_policy': 'omit'
    }
    default_fit_options.update(fit_options)
    
    # Realizar el ajuste
    result = model.fit(flux, pars, x=wavelength, **default_fit_options)
    
    return result
        
def resampler(wavelength, flux, resampler_pars):
    """
    Resample spectrum using flux-conserving resampling.
    
    Parameters:
    -----------
    wavelength : array-like
        Original wavelength array
    flux : array-like
        Original flux array
    resampler_pars : tuple
        (start, step, num, ref) parameters for new wavelength grid
    
    Returns:
    --------
    numpy.ndarray
        Resampled flux values
    """
    start, end, num = resampler_pars
    
    # Precompute new wavelength grid
    new_wavelength =  np.logspace(np.log10(start), np.log10(end), num=num) * u.AA
    
    # Create spectrum object
    spec = Spectrum(flux=flux * u.adu, spectral_axis=wavelength * u.AA)
    
    # Use singleton resampler to avoid recreation
    if not hasattr(resampler, '_fluxcon'):
        resampler._fluxcon = FluxConservingResampler(extrapolation_treatment='nan_fill')
    
    return resampler._fluxcon(spec, new_wavelength).flux.value

def concat_flux(flux_matrix):
    """
    Combina espectros superpuestos usando promedio simple.
    Sin algoritmos de rechazo de outliers.
    
    Parámetros:
    -----------
    flux_matrix : numpy.ndarray
        Matriz 2D de forma (n_spectra, n_pixels) donde cada fila es un espectro
        ya remuestreado al mismo eje espectral.
    
    Retorna:
    --------
    flux_combined : numpy.ndarray
        Espectro combinado 1D con n_pixels elementos.
    """
    import numpy as np
    from scipy import interpolate
    
    n_spectra, n_pixels = flux_matrix.shape
    
    if n_spectra == 0:
        raise ValueError("La matriz de flujos está vacía")
    
    if n_spectra == 1:
        # Solo un espectro, devolver tal cual
        return flux_matrix[0].copy()
    
    # Paso 1: Filtrar solo valores finitos (no NaN, no infinitos)
    valid_matrix = flux_matrix.copy()
    valid_matrix[~np.isfinite(valid_matrix)] = np.nan
    
    # Paso 2: Calcular promedio ignorando NaN
    flux_combined = np.nanmean(valid_matrix, axis=0)
    
    # Paso 3: Identificar gaps (NaN) en el resultado
    nan_mask = ~np.isfinite(flux_combined)
    nan_indices = np.where(nan_mask)[0]
    
    if len(nan_indices) == n_pixels:
        # Caso especial: todos los pixeles son NaN
        # Usar valor por defecto para espectro normalizado
        return np.ones(n_pixels)
    
    if len(nan_indices) > 0:
        # Encontrar segmentos contiguos de NaN
        gap_segments = []
        current_segment = [nan_indices[0]]
            
        for i in range(1, len(nan_indices)):
            if nan_indices[i] == nan_indices[i-1] + 1:
                current_segment.append(nan_indices[i])
            else:
                gap_segments.append(current_segment)
                current_segment = [nan_indices[i]]
        gap_segments.append(current_segment)
        
        # Paso 4: Interpolar gaps con spline cúbico
        for segment in gap_segments:
            gap_size = len(segment)
            
            # Solo interpolar gaps pequeños (hasta 15 pixeles)
            if 0 < gap_size <= 15:
                start_idx = segment[0]
                end_idx = segment[-1]
                
                # Buscar puntos válidos antes y después del gap
                before_idx = start_idx - 1
                after_idx = end_idx + 1
                
                while before_idx >= 0 and nan_mask[before_idx]:
                    before_idx -= 1
                while after_idx < n_pixels and nan_mask[after_idx]:
                    after_idx += 1
                
                # Si encontramos puntos a ambos lados, interpolar
                if before_idx >= 0 and after_idx < n_pixels:
                    # Recolectar varios puntos para mejor interpolación
                    x_points = []
                    y_points = []
                    
                    # Puntos antes del gap (hasta 3 puntos)
                    for offset in [-3, -2, -1]:
                        idx = before_idx + offset
                        if 0 <= idx < n_pixels and not nan_mask[idx]:
                            if idx not in x_points:  # Evitar duplicados
                                x_points.append(idx)
                                y_points.append(flux_combined[idx])
                    
                    # Punto inmediatamente antes del gap
                    if before_idx not in x_points:
                        x_points.append(before_idx)
                        y_points.append(flux_combined[before_idx])
                    
                    # Punto inmediatamente después del gap
                    if after_idx not in x_points:
                        x_points.append(after_idx)
                        y_points.append(flux_combined[after_idx])
                    
                    # Puntos después del gap (hasta 3 puntos)
                    for offset in [1, 2, 3]:
                        idx = after_idx + offset
                        if 0 <= idx < n_pixels and not nan_mask[idx]:
                            if idx not in x_points:  # Evitar duplicados
                                x_points.append(idx)
                                y_points.append(flux_combined[idx])
                    
                    # Ordenar puntos por posición
                    x_points = np.array(x_points)
                    y_points = np.array(y_points)
                    if len(x_points) >= 2:
                        sort_idx = np.argsort(x_points)
                        x_points = x_points[sort_idx]
                        y_points = y_points[sort_idx]
                        
                        # Interpolar con spline cúbico si tenemos al menos 4 puntos
                        if len(x_points) >= 4:
                            try:
                                spline = interpolate.CubicSpline(x_points, y_points)
                                flux_combined[segment] = spline(segment)
                                continue  # Gap interpolado, siguiente segmento
                            except:
                                pass  # Fallback a métodos más simples
                        
                        # Interpolación cuadrática si tenemos 3 puntos
                        if len(x_points) >= 3:
                            try:
                                poly = np.polyfit(x_points, y_points, 2)
                                poly_func = np.poly1d(poly)
                                flux_combined[segment] = poly_func(segment)
                                continue
                            except:
                                pass  # Fallback a lineal
                        
                        # Interpolación lineal (mínimo 2 puntos)
                        interp_func = interpolate.interp1d(
                            x_points, y_points,
                            kind='linear',
                            fill_value='extrapolate'
                        )
                        flux_combined[segment] = interp_func(segment)
        
        # Paso 5: Si todavía quedan NaN, usar interpolación lineal simple
        remaining_nan = np.where(~np.isfinite(flux_combined))[0]
        if len(remaining_nan) > 0:
            # Encontrar todos los índices con valores válidos
            good_indices = np.where(np.isfinite(flux_combined))[0]
            
            if len(good_indices) >= 2:
                # Interpolar linealmente
                interp_func = interpolate.interp1d(
                    good_indices,
                    flux_combined[good_indices],
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
                
                all_indices = np.arange(n_pixels)
                flux_combined = interp_func(all_indices)
            elif len(good_indices) == 1:
                # Solo un valor válido, rellenar todo con ese valor
                flux_combined[:] = flux_combined[good_indices[0]]
    
    return flux_combined

def save_concat_fits(filename, header, wavelength, flux):
    """
    Guarda un espectro concatenado en formato FITS 1D compatible con IRAF.

    Elimina todos los keywords MULTISPEC (WAT*, APNUMn, WCSDIM, etc.) y
    reescribe el eje espectral en escala log-lineal (DC-FLAG=1), de forma que
    el archivo resultante pueda ser leído directamente por IRAF/``splot``.

    Parameters
    ----------
    filename : str
        Ruta de salida del archivo FITS (se sobreescribe si existe).
    header : astropy.io.fits.Header
        Cabecera original del espectro MULTISPEC de la que se copian los
        keywords de observación relevantes (OBJECT, DATE-OBS, EXPTIME, etc.).
    wavelength : numpy.ndarray
        Array 1D de longitudes de onda en Ångströms del espectro concatenado.
    flux : numpy.ndarray
        Array 1D de flujo correspondiente a ``wavelength``.

    Notes
    -----
    El flujo se guarda como ``float32`` para compatibilidad con IRAF.
    La escala de longitudes de onda se almacena como log10 (CRVAL1/CDELT1)
    con el flag ``DC-FLAG=1``.
    """
    from astropy.io import fits
    import numpy as np
    
    # Crear una copia del header original
    new_header = header.copy()
    
    # ============================================
    # 1. ELIMINAR TODOS LOS KEYWORDS MULTISPEC
    # ============================================
    
    # Keywords WAT (específicos de multispec)
    for i in range(1, 100):  # WAT0_001 hasta WAT3_100
        for prefix in ['WAT0_', 'WAT1_', 'WAT2_', 'WAT3_']:
            key = f'{prefix}{i:03d}'
            if key in new_header:
                del new_header[key]
    
    # Keywords de órdenes individuales
    for i in range(1, 100):  # Hasta 100 órdenes
        # APNUM keywords
        key = f'APNUM{i}'
        if key in new_header:
            del new_header[key]
        
        # Keywords de parámetros espectrales
        for prefix in ['NPIX', 'PIXLOC', 'W0_', 'W1_', 'DW_', 'TYPE_']:
            key = f'{prefix}{i}'
            if key in new_header:
                del new_header[key]
    
    # Keywords de dimensiones y transformación multispec
    keys_to_remove = [
        'WCSDIM',       # = 3 (multispec)
        'LTM1_1', 'LTM2_2', 'LTM3_3',  # Matrices de transformación
        'CD1_1', 'CD2_2', 'CD3_3',     # Matrices CD
        'CTYPE1', 'CTYPE2', 'CTYPE3',  # Tipos de coordenadas multispec
        
        # Keywords de procesamiento específicos de multispec
        'BANDID1', 'BANDID2',
        'TRIM', 'OVERSCAN', 'CCDSEC', 'CCDMEAN', 'CCDMEANT',
        'CCDPROC', 'ZEROCOR', 'FIXPIX', 'FLATCOR',
        'DCLOG1',
        
        # Keywords que pueden confundir a IRAF para espectro 1D
        'NAXIS2', 'NAXIS3',  # IRAF espera NAXIS=1 para espectro 1D
    ]
    
    for key in keys_to_remove:
        if key in new_header:
            del new_header[key]
    
    # ============================================
    # 2. ACTUALIZAR KEYWORDS PARA ESPECTRO 1D
    # ============================================
    
    # Dimensiones: cambiar de 3D (multispec) a 1D (espectro simple)
    new_header['NAXIS'] = 1
        
    # Información del eje de longitud de onda para espectro 1D
    if len(wavelength) > 1:
        step = np.mean(np.diff(np.log10(wavelength)))
        start = np.log10(wavelength[0])
        ref = 0
        num = len(wavelength)
        new_header['CDELT1'] = step
        new_header['CRVAL1'] = start
        new_header['CRPIX1'] = ref
        new_header['NAXIS1'] = num
        new_header['CTYPE1'] = 'WAVELENGTH'  # Simple, no multispec
        new_header['CUNIT1'] = 'Angstrom'
        new_header['DC-FLAG'] = 1  # 1 = log-linear scale (IRAF convention)
    
    # Cambiar BUNIT si es necesario (de DU/PIXEL a algo más apropiado)
    if 'BUNIT' in new_header and new_header['BUNIT'] == 'DU/PIXEL':
        new_header['BUNIT'] = 'erg/cm2/s/A'  # O el que sea apropiado
    
    # Información sobre la concatenación
    new_header['CONCAT'] = ('YES', 'Spectrum concatenated from multiple orders')
    new_header['HISTORY'] = 'Original multispec spectrum with 61 orders'
    new_header['HISTORY'] = f'Concatenated to 1D spectrum with {len(flux)} points'
    new_header['HISTORY'] = f'Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} A'
    
    # Mantener información importante de observación
    important_keys = [
        'OBJECT', 'RA', 'DEC', 'RA-D', 'DEC-D', 'EQUINOX',
        'DATE-OBS', 'UT', 'EXPTIME', 'AIRMASS',
        'TELESCOP', 'INSTRUME', 'OBSERVER',
        'SITENAME', 'SITELAT', 'SITELONG', 'SITEALT',
        'HJD', 'VHELIO', 'VLSR'
    ]
    
    # Solo mantener los que existen
    for key in important_keys:
        if key not in new_header and key in header:
            new_header[key] = header[key]
    
    # ============================================
    # 3. CREAR Y GUARDAR EL FITS
    # ============================================
    
    # Crear HDU con los datos (array 1D simple)
    # Convertir a float32 para compatibilidad
    data_1d = flux.astype(np.float32)
    hdu = fits.PrimaryHDU(data_1d, header=new_header)
    
    # Opcional: crear extensión con longitudes de onda
    if False:  # Cambiar a True si quieres guardar longitudes de onda por separado
        col1 = fits.Column(name='WAVELENGTH', format='D', array=wavelength)
        col2 = fits.Column(name='FLUX', format='E', array=flux.astype(np.float32))
        cols = fits.ColDefs([col1, col2])
        table_hdu = fits.BinTableHDU.from_columns(cols)
        table_hdu.header['EXTNAME'] = 'SPECTRUM'
        
        hdul = fits.HDUList([hdu, table_hdu])
    else:
        hdul = fits.HDUList([hdu])
    
    # Guardar
    hdul.writeto(filename, overwrite=True, output_verify='fix')

class spectrum:
    """
    Contenedor ligero para un espectro estelar 1D o por órdenes.

    Attributes
    ----------
    wavelength : numpy.ndarray or None
        Array de longitudes de onda en Ångströms.
    flux : numpy.ndarray or None
        Array de flujo correspondiente a ``wavelength``.
    header : astropy.io.fits.Header or None
        Cabecera FITS asociada al espectro.
    """

    def __init__(self, wavelength=None, flux=None, header=None):
        """
        Inicializa el espectro con los datos provistos.

        Parameters
        ----------
        wavelength : array-like, optional
            Longitudes de onda en Ångströms.
        flux : array-like, optional
            Flujo del espectro.
        header : astropy.io.fits.Header, optional
            Cabecera FITS del espectro.
        """
        self.header = header
        self.wavelength = wavelength
        self.flux = flux

    def save(self, file_name, overwrite=False):
        """
        Guarda el espectro en un archivo FITS con los datos en la extensión primaria.

        Los arrays ``wavelength`` y ``flux`` se apilan en un array 2D de forma
        ``(2, n_pixels)`` y se escriben junto con ``header``.

        Parameters
        ----------
        file_name : str
            Ruta del archivo FITS de salida.
        overwrite : bool, optional
            Si es ``True``, sobreescribe el archivo si ya existe. Por defecto ``False``.
        """
        data = np.array([self.wavelength, self.flux])
        fits.writeto(file_name, data, self.header, overwrite=overwrite)
        print(f'Spectrum saved to {file_name}')

def find_closest_line(wavelength, csv_path=None, df=None):
    """
    Encuentra la línea espectral más cercana a la longitud de onda dada.
    
    Parameters:
    wavelength (float): Longitud de onda en Ångströms
    csv_path (str, optional): Ruta al archivo CSV. Si es None, busca 'lines.csv' 
                              en el mismo directorio del script.
    df (DataFrame, optional): DataFrame ya cargado (prioritario sobre csv_path)
    
    Returns:
    dict: Diccionario con la información de la línea más cercana
    """
    # Si no se proporciona un DataFrame, cargarlo desde el archivo
    if df is None:
        # Si no se especifica path, usar el directorio del script
        if csv_path is None:
            # Obtener el directorio donde se encuentra el script actual
            script_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(script_dir, 'lines.csv')
        
        # Verificar que el archivo existe
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")
        
        # Cargar el CSV
        df = pd.read_csv(csv_path)
    
    # Verificar que la columna lambda_rest existe
    if 'lambda_rest' not in df.columns:
        raise KeyError("El CSV debe contener una columna 'lambda_rest'")
    
    # Calcular la diferencia absoluta
    df_copy = df.copy()
    df_copy['diff'] = abs(df_copy['lambda_rest'] - wavelength)
    
    # Encontrar el índice con la mínima diferencia
    idx_min = df_copy['diff'].idxmin()
    
    # Obtener la línea más cercana
    closest_line = df.loc[idx_min].to_dict()

    return closest_line


# Velocidad de la luz en km/s
c_light = 299792.458


def vr(lambda_obs, lambda0, vhelio=0.0):
    """
    Calcula la velocidad radial helocentrica a partir del corrimiento Doppler.

    Parameters
    ----------
    lambda_obs : float
        Longitud de onda observada del centro de la linea (A).
    lambda0 : float
        Longitud de onda en reposo de la linea (A).
    vhelio : float
        Correccion de velocidad heliocentrica en km/s (default 0).

    Returns
    -------
    float
        Velocidad radial en km/s.
    """
    return (lambda_obs - lambda0) / lambda0 * c_light + vhelio


def vrerr(lambda_err, lambda0):
    """
    Propaga el error en la longitud de onda central a error en velocidad radial.

    Parameters
    ----------
    lambda_err : float
        Error en la longitud de onda observada (A).
    lambda0 : float
        Longitud de onda en reposo de la linea (A).

    Returns
    -------
    float
        Error en velocidad radial en km/s.
    """
    return c_light / lambda0 * lambda_err