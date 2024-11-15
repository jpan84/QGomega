import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

yrstr = '9999'
a = 6.371e6
ROT = 7.29e-5
g = 9.81
dx, dy, f = None, None, None

lonslc = slice(117, 124)
latslc = slice(25, 55)

def main():
   Z = xr.open_dataset('hgt.%s.nc' % yrstr).hgt
   V = xr.open_dataset('vwnd.%s.nc' % yrstr).vwnd
   Q = xr.open_dataset('shum.%s.nc' % yrstr).shum
   OM = xr.open_dataset('omega.%s.nc' % yrstr).omega

   print(Z)
   print(Z.values.shape)

   f = 2 * ROT * np.sin(np.deg2rad(Z.lat)).values
   dlon = np.abs(Z.lon[1] - Z.lon[0]).values
   dlat = np.abs(Z.lat[1] - Z.lat[0]).values
   dy = a * np.deg2rad(dlat)
   dx = (a * np.cos(np.deg2rad(Z.lat)) * np.deg2rad(dlon)).values
   print(f, dx, dy)

   #TODO: panel b
   #ug, vag, omega
   ug = g / f[None, 1:-1, None] * (Z.values[:, 2:, :] - Z.values[:, :-2, :]) / dy
   ug = xr.DataArray(ug, dims=['level', 'lat', 'lon'], coords=[Z.level, Z.lat[1:-1], Z.lon])
   vg = g / f[None, :, None] * (np.roll(Z.values, -2, axis=2) - Z.values) / dx[None, :, None]
   vag = V.values - vg
   vag = xr.DataArray(vag, dims=['level', 'lat', 'lon'], coords=[Z.level, Z.lat, Z.lon])
   print(ug)
   print(vag)

   pltug = ug.sel(lon=lonslc, lat=latslc).mean(dim='lon')
   plt.contourf(pltug.lat, pltug.level, pltug.values)
   plt.gca().invert_yaxis()
   plt.colorbar()
   plt.show()

   #TODO: panel c
   #shum, inertial stability

if __name__ == '__main__':
   main()
