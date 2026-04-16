#%%
from utils import *
import matplotlib.pyplot as plt
import numpy as np
# %%
file = "../../ssd/HD217312/espectros/845000_s1d.fits"

# %%
with fits.open(file) as hdul:
    hdr = hdul[0].header
    data = hdul[0].data

hdr
# data

# %%
plt.plot(data)
# %%
header, wavelength, flux = read_fits_simple(file)
#%%

header

#%%
plt.plot(wavelength, flux)
# %%

center = 5875
width = 40
region = [[center-width,center+width]]

mask = mask_generator(wavelength, ranges=region)
# %%
plt.plot(wavelength[mask], flux[mask])

# %%
from utils import find_closest_line

line_info = find_closest_line(5876)

linename = '5875'
linename = line_info.get('name', linename)
# print(linename)

if len(linename) > 4:
    print(linename[-4:])