import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

yrstr = '9999'
a = 6.371e6
ROT = 7.29e-5
g = 9.81
dx, dy, f = None, None, None

lonslc = slice(117, 124)
latslc = slice(55, 30)
levslc = slice(925, 150)
pticks = [150, 200, 300, 400, 500, 600, 700, 850, 925]

def main():
   Z = xr.open_dataset('hgt.%s.nc' % yrstr).hgt
   V = xr.open_dataset('vwnd.%s.nc' % yrstr).vwnd
   U = xr.open_dataset('uwnd.%s.nc' % yrstr).uwnd
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
   ug = g / f[None, 1:-1, None] * (Z.values[:, 2:, :] - Z.values[:, :-2, :]) / 2 / dy
   ug = xr.DataArray(ug, dims=['level', 'lat', 'lon'], coords=[Z.level, Z.lat[1:-1], Z.lon])
   vg = g / f[None, :, None] * (np.roll(Z.values, -2, axis=2) - Z.values) / 2 / dx[None, :, None]
   vag = V.values - vg
   vag = xr.DataArray(vag, dims=['level', 'lat', 'lon'], coords=[Z.level, Z.lat, Z.lon])
   print(ug)
   print(vag)

   plt.rc('font', size=14)
   plt.rcParams['contour.negative_linestyle'] = 'solid'
   clabelkwargs = {'inline': 1, 'fontsize': 12, 'colors': 'black'}
   pltug = ug.sel(lon=lonslc, lat=latslc, level=levslc).mean(dim='lon')
   pltom = OM.sel(lon=lonslc, lat=latslc, level=levslc).mean(dim='lon')
   pltvag = vag.sel(lon=lonslc, lat=latslc, level=levslc).mean(dim='lon')
   cs = plt.contour(pltug.lat, pltug.level, pltug.values, levels=np.arange(8, 33, 4), colors='cyan')
   cs1 = plt.contour(pltom.lat, pltom.level, pltom.values, levels=np.arange(-0.03, -0.12, -0.02)[::-1], colors='red')
   cf = plt.contourf(pltom.lat, pltom.level, pltom.values, levels=np.arange(0.0, 0.0501, 5e-3))
   qv = plt.quiver(pltvag.lat, pltvag.level, pltvag, -1e1 * pltom.values, pivot='mid', scale=15, color='blue', width=3e-3)
   plt.quiverkey(qv, X=.25, Y=-0.1, U=1, label='1 m s$^{-1}$ -0.1 Pa s$^{-1}$', labelpos='E')
   plt.ylim(150, 925)
   plt.yticks(pticks, labels=pticks)
   plt.gca().invert_yaxis()
   plt.clabel(cs, cs.levels[::2], **clabelkwargs, fmt='%d')
   plt.clabel(cs1, [cs1.levels[0], cs1.levels[2]], **clabelkwargs, fmt='%.2f')
   plt.colorbar(cf)
   plt.title('(b) AMax = %.3f     DMax=%.3f' % (pltom.min(), pltom.max()), fontsize=16)
   plt.ylabel('Pressure (hPa)')
   plt.savefig('Fig1b.png')
   plt.close()

   #TODO: panel c
   #shum, inertial stability
   fsq = f[None, 2:-2, None] - (ug.values[:, :-2, :] - ug.values[:, 2:, :]) / 2 / dy
   fsq = xr.DataArray(fsq, dims=['level', 'lat', 'lon'], coords=[Z.level, Z.lat[2:-2], Z.lon])

   pltq = 1000 * Q.sel(lon=lonslc, lat=latslc, level=levslc).mean(dim='lon')
   pltfsq = 1e5 * fsq.sel(lon=lonslc, lat=latslc, level=levslc).mean(dim='lon')
   cs1 = plt.contour(pltq.lat, pltq.level, pltq.values, levels=np.arange(2, 19, 2), colors='black')
   cs = plt.contour(pltug.lat, pltug.level, pltug.values, levels=np.arange(8, 33, 4), colors='cyan')
   plt.contourf(pltfsq.lat, pltfsq.level, pltfsq.values, levels=np.arange(0, 16, 3))
   plt.ylim(150, 925)
   plt.yticks(pticks, labels=pticks)
   plt.gca().invert_yaxis()
   plt.clabel(cs, cs.levels[::2], **clabelkwargs, fmt='%d') 
   plt.clabel(cs1, **clabelkwargs, fmt='%d')
   plt.colorbar()
   plt.title('(c)', fontsize=16)
   plt.ylabel('Pressure (hPa)')
   plt.savefig('Fig1c.png')

if __name__ == '__main__':
   main()
